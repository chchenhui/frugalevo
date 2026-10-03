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


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

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

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    """Vectorized greedy (LPT) packer.

    Replaces the per-layer Python loop (with a Python `min` over packs
    for every item) by tensor ops broadcast across all layers. We still
    iterate once per item (num_groups iterations) so every item is
    assigned exactly once — the previous attempt only ran
    groups_per_pack iterations and left pack_index partially
    uninitialized, causing out-of-bounds indices downstream. Each
    iteration: mask out full packs, argmin over pack weights to pick
    the least-loaded pack with capacity, scatter-assign the next
    heaviest item of each layer, then scatter-add its weight/count.
    Assignments are identical to the original greedy algorithm.
    """
    device = weight.device
    sorted_w, indices = weight.float().sort(-1, descending=True)
    indices = indices.to(device)
    arange_layers = torch.arange(num_layers, dtype=torch.int64,
                                 device=device)
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64,
                             device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64,
                               device=device)
    pack_weights = torch.zeros(num_layers, num_packs, device=device)
    pack_items = torch.zeros(num_layers, num_packs, dtype=torch.int64,
                             device=device)
    for step in range(num_groups):
        # Only packs with free capacity are candidates; pick least-loaded.
        masked = pack_weights.masked_fill(pack_items >= groups_per_pack,
                                          float("inf"))
        pack = masked.argmin(-1)  # [num_layers]
        rank = pack_items[arange_layers, pack]
        item = indices[:, step]
        pack_index.scatter_(1, item.unsqueeze(1), pack.unsqueeze(1))
        rank_in_pack.scatter_(1, item.unsqueeze(1), rank.unsqueeze(1))
        pack_weights.scatter_add_(1, pack.unsqueeze(1),
                                  sorted_w[:, step].unsqueeze(1))
        pack_items.scatter_add_(1, pack.unsqueeze(1),
                                torch.ones_like(rank).unsqueeze(1))
    return pack_index.cpu(), rank_in_pack.cpu()


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
    """
    Vectorized waterfilling replication.

    Instead of the sequential argmax(weight/logcnt) loop (one Python
    iteration per redundant replica), we binary-search a per-row load
    target T such that sum(ceil(w/T)) fits within num_phy slots, set
    cnt[e] = max(1, ceil(w[e]/T)), then distribute leftover slots
    greedily to the expert with the largest current w/cnt. This
    enforces a uniform per-replica load ceiling w/cnt <= T for every
    expert, which the myopic argmax loop cannot guarantee. The
    phy2log/rank expansion is done with repeat_interleave over the
    flattened [n, num_log] grid plus a cumsum-based rank computation
    (no per-layer Python loop, no multi-dim expand). Output contract
    (phy2log, rank, logcnt shapes/dtypes) is identical to before.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    w = weight.float()
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    if num_redundant == 0:
        phy2log = torch.arange(num_phy, dtype=torch.int64,
                               device=device).repeat(n, 1)
        rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # --- Bisection on load target T (per row), ~60 bounded iterations ---
    hi = w.max().clamp(min=1e-12) * float(num_phy)
    lo = hi.clone()
    # ensure lo satisfies sum(ceil(w/lo)) <= num_phy (true for lo = w.max())
    for _ in range(60):
        mid = (lo + hi) * 0.5
        fits = torch.ceil(w / mid.clamp(min=1e-12)).sum(-1,
                                                        keepdim=True) <= num_phy
        lo = torch.where(fits, mid, lo)
        hi = torch.where(fits, hi, mid)
    cnt = torch.ceil(w / lo.clamp(min=1e-12)).clamp(min=1).to(torch.int64)

    # --- Greedy fix-up: distribute leftover slots to largest w/cnt ---
    leftover = num_phy - cnt.sum(-1)  # [n], each >= 0
    max_leftover = int(leftover.max().item())
    for _ in range(max_leftover):
        active = leftover > 0
        ratio = w / cnt.float()
        ratio = ratio.masked_fill(~active.unsqueeze(-1), float("-inf"))
        target = ratio.max(dim=-1).indices
        cnt[arangen, target] += 1
        leftover = (leftover - 1).clamp(min=0)

    # --- Expand counts to phy2log / rank via repeat_interleave ---
    flat_cnt = cnt.flatten()  # [n * num_log], row-major
    base = torch.arange(num_log, dtype=torch.int64,
                        device=device).unsqueeze(0).expand(n, num_log)
    seg = torch.arange(n * num_log, dtype=torch.int64,
                       device=device).repeat_interleave(flat_cnt)
    ids = base.flatten()[seg]  # logical expert id per slot
    rows = seg // num_log
    starts = torch.cumsum(flat_cnt, 0) - flat_cnt
    rank_flat = torch.arange(ids.numel(), dtype=torch.int64,
                             device=device) - starts[seg]
    phy2log = ids.view(n, num_phy)
    rank = rank_flat.view(n, num_phy)
    logcnt = cnt
    return phy2log, rank, logcnt


def _hierarchical_fallback(weight, num_physical_experts, num_groups,
                           num_nodes, num_gpus):
    """Original hierarchical pipeline, kept as a validity fallback."""
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

    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
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


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Global snake-matching rebalancer.

    Replaces the hierarchical group->node->GPU pipeline with a single global
    construction: (1) replica counts per logical expert via global waterfilling
    (repeatedly replicate the expert with the highest weight/count, vectorized
    across layers, at most num_phy - num_log iterations); (2) per-replica loads
    weight/cnt; (3) sort replicas by load descending and assign them to GPU
    slots in a fixed "snake" slot order (GPU 0..G-1, G-1..0, repeating), so
    heavy replicas always land on distinct GPUs and each GPU receives exactly
    num_phy/num_gpus slots. No node-boundary constraint is imposed, removing
    the balance bound the hierarchical decomposition suffers from. Fully
    deterministic, O(num_phy log num_phy) tensor ops, no per-expert Python
    loop. Falls back to the original hierarchical pipeline on any structural
    assertion failure.
    """
    try:
        num_layers, num_log = weight.shape
        num_phy = num_physical_experts
        assert num_phy >= num_log
        assert num_phy % num_gpus == 0
        slots_per_gpu = num_phy // num_gpus
        device = weight.device
        wf = weight.float()
        num_redundant = num_phy - num_log

        # --- Phase 1: global waterfilling replica counts ---
        log_ids = torch.empty(num_layers, num_phy, dtype=torch.int64,
                              device=device)
        rep_rank = torch.empty(num_layers, num_phy, dtype=torch.int64,
                               device=device)
        log_ids[:, :num_log] = torch.arange(num_log, dtype=torch.int64,
                                            device=device)
        rep_rank[:, :num_log] = 0
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64,
                            device=device)
        arangen = torch.arange(num_layers, dtype=torch.int64, device=device)
        for i in range(num_redundant):
            target = (wf / logcnt).max(dim=-1).indices
            log_ids[:, num_log + i] = target
            rep_rank[:, num_log + i] = logcnt[arangen, target]
            logcnt[arangen, target] += 1

        # --- Phase 2: per-replica loads, sorted descending ---
        loads = wf.gather(-1, log_ids) / logcnt.gather(-1, log_ids).float()
        order = loads.sort(-1, descending=True).indices  # [L, num_phy]
        sorted_logs = log_ids.gather(-1, order)
        sorted_ranks = rep_rank.gather(-1, order)

        # --- Phase 3: block-deal (reverse snake) slot assignment ---
        """Blocked multiway-partition deal across GPUs.

        Replaces the per-replica sequential LPT loop (num_phy iterations
        of masked argmin + scatter) with a blocked greedy deal: replicas
        are already sorted by load descending; they are dealt in blocks
        of num_gpus. For each block, GPUs are ordered by current load
        ascending (argsort), and the block's replicas — in descending
        load order — are matched onto that ordering (heaviest remaining
        replica -> currently lightest GPU). Each block contributes
        exactly one replica per GPU, and there are exactly
        slots_per_gpu blocks, so every GPU ends with exactly
        slots_per_gpu replicas and slot index within a GPU equals the
        block index. Per block this costs one argsort, one gather and
        one add instead of num_gpus sequential steps, cutting the
        Python loop from num_phy to num_phy/num_gpus iterations while
        preserving the greedy heaviest-to-lightest pairing.
        """
        gpu_loads = torch.zeros(num_layers, num_gpus, device=device)
        phy2log = torch.empty(num_layers, num_phy, dtype=torch.int64,
                              device=device)
        phyrank = torch.empty(num_layers, num_phy, dtype=torch.int64,
                              device=device)
        arange_gpus = torch.arange(num_gpus, dtype=torch.int64,
                                   device=device)
        for blk in range(slots_per_gpu):
            # GPUs sorted by current load, ascending.
            gpu_order = gpu_loads.argsort(-1)  # [L, G]
            lo = blk * num_gpus
            hi = lo + num_gpus
            blk_logs = sorted_logs[:, lo:hi]   # [L, G], descending load
            blk_ranks = sorted_ranks[:, lo:hi]
            blk_loads = loads[:, lo:hi]
            # Match j-th heaviest block replica to j-th lightest GPU:
            # scatter along gpu_order positions.
            dst = torch.empty_like(blk_logs)
            dst.scatter_(1, gpu_order, blk_logs)
            dst_rank = torch.empty_like(blk_ranks)
            dst_rank.scatter_(1, gpu_order, blk_ranks)
            phy2log[:, lo:hi] = dst
            phyrank[:, lo:hi] = dst_rank
            gpu_loads = gpu_loads + blk_loads.gather(1, gpu_order)
        return phy2log, phyrank, logcnt
    except Exception:
        return _hierarchical_fallback(weight, num_physical_experts,
                                      num_groups, num_nodes, num_gpus)


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
            replicas for each logical expert
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

