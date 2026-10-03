# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
"""

# EVOLVE-BLOCK-START

import torch


def _snake_packing(weight: torch.Tensor,
                   num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted items into m packs with exactly n/m items each, minimizing
    max pack load, using the vectorized "sorted snake" heuristic:
    sort items descending per row, split into blocks of size num_packs, and
    alternate block direction so large items pair with small ones.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64,
            device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    order = weight.float().sort(-1, descending=True).indices  # [L, n]
    blocks = order.view(num_layers, groups_per_pack, num_packs)
    # snake: reverse alternating blocks along the block axis
    rev = torch.arange(num_packs - 1, -1, -1,
                       dtype=torch.int64, device=order.device)
    flip_mask = (torch.arange(groups_per_pack,
                              device=order.device) % 2 == 1).view(1, -1, 1)
    snake = torch.where(flip_mask,
                         blocks.index_select(-1, rev),
                         blocks)  # [L, blocks, num_packs]
    # pack_index: for each pack column, its sorted items
    # snake[l, b, p] = item id of slot (b, p)
    # pack p gets slots (b, p) for all b -> slot index within pack = b
    pack_index = torch.empty(num_layers, num_groups,
                              dtype=torch.int64, device=order.device)
    rank_in_pack = torch.empty(num_layers, num_groups,
                                dtype=torch.int64, device=order.device)
    for p in range(num_packs):
        pack_index.scatter_(
            -1, snake[:, :, p].reshape(num_layers, -1),
            torch.full((num_layers, groups_per_pack), p,
                       dtype=torch.int64, device=order.device))
        rank_in_pack.scatter_(
            -1, snake[:, :, p].reshape(num_layers, -1),
            torch.arange(groups_per_pack, dtype=torch.int64,
                         device=order.device).expand(num_layers, -1))
    return pack_index, rank_in_pack


def _waterfilling_counts(weight: torch.Tensor,
                         total: int) -> torch.Tensor:
    """
    Compute replica counts c (c >= 1, sum(c) == total) minimizing
    max(weight / c) via proportional water-filling + integral correction.
    weight: [X, n], returns int64 counts [X, n].
    """
    X, n = weight.shape
    assert total >= n
    w = weight.float().clamp_min(0)
    wsum = w.sum(-1, keepdim=True)
    # proportional ideal counts
    ideal = w * total / wsum.clamp_min(1e-12)
    counts = ideal.ceil()
    # ensure min 1
    counts = counts.clamp_min(1)
    # fix rounding so rows sum to `total`
    cnt = counts.long()
    row_sum = cnt.sum(-1)
    diff = row_sum - total  # [X]
    # rows over budget: decrement smallest fractional (ceil - ideal) first
    frac = (counts - ideal)  # extra amount added by ceiling
    order = frac.sort(-1).indices  # ascending: smallest overhead first
    dev = weight.device
    arange_n = torch.arange(n, dtype=torch.int64, device=dev)
    rowidx = torch.arange(X, dtype=torch.int64, device=dev)
    for _ in range(0):  # placeholder; vectorized fix below
        pass
    # decrement pass (diff > 0): pick entries with largest counts, in
    # ascending-overhead order, only where count > 1
    over = (diff > 0)
    # process rows needing decrements: iterate bounded times (at most n)
    for _ in range(int(over.any().item()) * n + 0):
        active = (diff > 0)
        if not bool(active.any()):
            break
        cand = order  # [X, n] ascending overhead order
        # pick first index in each active row with count > 1
        eligible = active.unsqueeze(1) & (cnt > 1)
        first = torch.where(
            eligible.gather(1, cand),
            cand,
            torch.full_like(cand, -1)).masked_fill(
                ~eligible.gather(1, cand).any(-1, keepdim=True), -1)
        ok = first != -1
        pick = first.clamp_min(0)
        can_dec = ok & active
        # decrement those rows
        cnt = cnt.clone()
        dec_rows = can_dec.nonzero(as_tuple=True)[0]
        if dec_rows.numel() == 0:
            break
        picks = pick[dec_rows]
        cnt[dec_rows, picks] -= 1
        diff = diff.clone()
        diff[dec_rows] -= 1
    # increment pass (diff < 0): add to entries with largest ideal deficit
    under = (diff < 0)
    if bool(under.any()):
        deficit_order = (-ideal).sort(-1).indices  # largest weight first
        add_count = (-diff).clamp_min(0)  # per-row increments needed
        # distribute greedily: repeatedly add to largest-weight entries
        reps = add_count.max().item()
        deficit_pos = 0
        while deficit_pos < reps:
            active_rows = (add_count > 0).nonzero(as_tuple=True)[0]
            if active_rows.numel() == 0:
                break
            # add one to top entry of each active row
            # top entries: entries not yet saturated relative to others;
            # simple: cycle through largest-weight entries repeatedly
            # use ranks so each row's k-th add goes to k-th best entry
            k = deficit_pos
            # rank k among deficit_order per row
            pos = deficit_order[:, k % n]
            ar = active_rows
            p = pos[ar]
            cnt[ar, p] += 1
            add_count[ar] -= 1
            deficit_pos += 1
    return cnt


def _replicate_experts(weight: torch.Tensor, num_phy: int):
    """
    Vectorized replication: counts via water-filling, then build phy2log /
    rank deterministically.
    """
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device
    logcnt = _waterfilling_counts(weight, num_phy)
    total = logcnt.sum(-1)
    assert torch.all(total == num_phy), "water-filling count mismatch"
    phy2log = torch.empty(n, num_phy, dtype=torch.int64, device=device)
    rank = torch.empty(n, num_phy, dtype=torch.int64, device=device)
    # place replicas: sort experts by (count desc, id) then fill
    order = (-logcnt.float()).sort(-1, stable=True).indices
    # offset of each logical expert's replicas
    offs = logcnt.cumsum(-1) - logcnt
    slot_expert = torch.repeat_interleave(
        order, logcnt.gather(-1, order), dim=-1)  # [n, num_phy]
    slot_rank = torch.arange(num_phy, device=device).expand(
        n, -1) - offs.gather(-1, slot_expert)
    # invert to phy layout: need position within phy2log row for each expert
    # we instead compute per-expert starting positions in phy order:
    # phy2log constructed by assigning expert e replicas to consecutive
    # slots determined by stable ordering of experts
    base = offs.gather(-1, order)  # [n, num_log] start offsets in sorted order
    # position of each sorted-order entry
    pos = base + torch.arange(num_log, device=device).expand(n, -1)
    inv = torch.empty_like(pos)
    inv.scatter_(-1, pos, torch.arange(num_log, device=device).expand(n, -1))
    # inv gives, for each consecutive slot position, the rank in `order`
    # slot_expert expects positions per expert; recompute directly:
    # simpler: rebuild phy2log row-major over original expert ids
    starts = torch.zeros(n, dtype=torch.int64, device=device)
    # positions in phy for each logical expert: use sorted stable by expert id
    expert_starts = logcnt.cumsum(-1) - logcnt  # [n, num_log] per original id
    for i in range(n):
        li = logcnt[i]
        # assign each logical expert its replicas
        ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), li)
        r = torch.arange(num_phy, device=device) - expert_starts[i].gather(
            0, ids)
        phy2log[i] = ids
        rank[i] = r
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1,
                                        (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = _snake_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = _replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = _snake_packing(tokens_per_phy,
                                               num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.
    Same inputs/outputs as the original implementation.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

