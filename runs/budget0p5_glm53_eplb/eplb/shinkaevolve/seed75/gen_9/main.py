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

_EPS = 1e-9
_INF = float("inf")


def _lpt_packing(w: torch.Tensor, num_packs: int, cap: int, order: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched greedy LPT seed: insert items (given in `order`, heaviest first)
    into the least-loaded pack that still has capacity.

    Parameters:
        w: [X, n] item weights (float)
        num_packs: number of packs
        cap: exact number of items each pack must contain
        order: [X, n] item indices in descending weight order

    Returns:
        pack_index: [X, n] pack of each item
        pack_weight: [X, num_packs] total weight of each pack
    """
    x, n = w.shape
    rows = torch.arange(x)
    pack_index = torch.zeros(x, n, dtype=torch.int64)
    pack_weight = torch.zeros(x, num_packs, dtype=w.dtype)
    pack_items = torch.zeros(x, num_packs, dtype=torch.int64)
    for t in range(n):
        item = order[:, t]
        cand = pack_weight.masked_fill(pack_items >= cap, _INF)
        pack = cand.argmin(-1)
        pack_index[rows, item] = pack
        pack_items[rows, pack] += 1
        pack_weight[rows, pack] += w[rows, item]
    return pack_index, pack_weight


def _refine_packing(w: torch.Tensor, pack_index: torch.Tensor,
                    pack_weight: torch.Tensor, max_rounds: int = 64
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized local-search refinement of a cardinality-constrained packing.

    Each round, for every row independently, considers swapping one item of
    the heaviest pack with one item of the lightest pack. Two candidate swaps
    are evaluated (the "coarse" swap of the heaviest/lightest items and the
    "fine" swap of the lightest/heaviest items) and the one minimizing the
    resulting spread is applied. Swaps only happen when they strictly reduce
    the max-min spread, so refinement monotonically improves balance.

    Parameters:
        w: [X, n] item weights
        pack_index: [X, n] current pack assignment
        pack_weight: [X, m] current pack loads
        max_rounds: safety bound on refinement iterations

    Returns:
        pack_index, pack_weight after refinement
    """
    x, n = w.shape
    m = pack_weight.size(1)
    if m <= 1 or n <= 1:
        return pack_index, pack_weight
    rows = torch.arange(x)
    pack_index = pack_index.clone()
    pack_weight = pack_weight.clone()
    for _ in range(max_rounds):
        h = pack_weight.argmax(-1)
        l = pack_weight.argmin(-1)
        spread = pack_weight[rows, h] - pack_weight[rows, l]
        if bool((spread <= _EPS).all()):
            break
        in_h = pack_index == h.unsqueeze(1)
        in_l = pack_index == l.unsqueeze(1)
        # Coarse candidate: heaviest item of h vs lightest item of l.
        ha = w.masked_fill(~in_h, -_INF).argmax(-1)
        lb = w.masked_fill(~in_l, _INF).argmin(-1)
        # Fine candidate: lightest item of h vs heaviest item of l.
        hb = w.masked_fill(~in_h, _INF).argmin(-1)
        la = w.masked_fill(~in_l, -_INF).argmax(-1)
        d_coarse = w[rows, ha] - w[rows, lb]
        d_fine = w[rows, hb] - w[rows, la]
        ok_coarse = (d_coarse > _EPS) & (d_coarse < spread)
        ok_fine = (d_fine > _EPS) & (d_fine < spread)
        new_coarse = (spread - 2 * d_coarse).abs()
        new_fine = (spread - 2 * d_fine).abs()
        take_coarse = ok_coarse & (~ok_fine | (new_coarse < new_fine))
        take_fine = ok_fine & (~take_coarse)
        moved = take_coarse | take_fine
        if not bool(moved.any()):
            break
        idx = moved.nonzero(as_tuple=True)[0]
        hi = h[idx]
        li = l[idx]
        ai = torch.where(take_coarse, ha, hb)[idx]
        bi = torch.where(take_coarse, lb, la)[idx]
        di = torch.where(take_coarse, d_coarse, d_fine)[idx]
        pack_index[idx, ai] = li
        pack_index[idx, bi] = hi
        pack_weight[idx, hi] -= di
        pack_weight[idx, li] += di
    return pack_index, pack_weight


def _rank_from_pack_index(pack_index: torch.Tensor) -> torch.Tensor:
    """
    Derive the slot rank of each item within its pack via a stable sort of
    the pack assignment. Any within-pack permutation is valid downstream, so
    this decouples ranks from insertion order and supports refined packings.

    Parameters:
        pack_index: [X, n] pack of each item

    Returns:
        rank_in_pack: [X, n] slot index of each item within its pack
    """
    x, n = pack_index.shape
    order = torch.argsort(pack_index, dim=-1, stable=True)
    sorted_packs = pack_index.gather(-1, order)
    pos = torch.arange(n, dtype=torch.int64).unsqueeze(0).expand(x, n)
    is_start = torch.zeros(x, n, dtype=torch.int64)
    is_start[:, 0] = 1
    if n > 1:
        is_start[:, 1:] = (sorted_packs[:, 1:] !=
                          sorted_packs[:, :-1]).to(torch.int64)
    run_start = torch.cummax(pos * is_start, dim=-1).values
    rank_sorted = pos - run_start
    rank = torch.empty_like(pack_index)
    rank.scatter_(-1, order, rank_sorted)
    return rank


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses a batched greedy LPT seed followed by a vectorized local-search
    swap refinement between the heaviest and lightest packs of each row.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1 or num_packs <= 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64).expand(
                                      num_layers, num_groups).clone()
        rank_in_pack = torch.zeros(num_layers,
                                   num_groups,
                                   dtype=torch.int64)
        return pack_index, rank_in_pack

    w = weight.float()
    order = w.sort(-1, descending=True).indices
    pack_index, pack_weight = _lpt_packing(w, num_packs, groups_per_pack, order)
    pack_index, pack_weight = _refine_packing(w, pack_index, pack_weight)
    rank_in_pack = _rank_from_pack_index(pack_index)
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
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
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
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

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
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
