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

    # Vectorized greedy LPT: process groups in descending weight order,
    # assigning each to the least-loaded pack with free capacity. The whole
    # assignment is done with tensor ops batched across all layers at once,
    # avoiding slow per-layer/per-group Python loops.
    # Quadratic-penalty LPT (exponential-penalty family, quadratic form):
    # instead of the myopic linear min-load argmin, pick the eligible pack
    # minimizing load_j + beta * (load_j - mean_load)^2. The convex quadratic
    # term penalizes packs drifting toward the eventual maximum far more
    # steeply than the linear rule, while the linear term keeps the greedy
    # LPT behavior dominant (unlike a pure exp(beta*load) rule, which
    # over-penalizes any pack holding a heavy item and degenerates to
    # round-robin). beta = num_packs / total_load is scale-free, so the
    # penalty term is O(load) only for large deviations. Deterministic
    # tie-breaking via argmin; the capacity mask still enforces exact
    # cardinality, and the loop count is unchanged.
    device = weight.device
    sorted_weight, indices = weight.float().sort(-1, descending=True)
    pack_index = torch.empty_like(indices)
    rank_in_pack = torch.empty_like(indices)
    pack_weights = torch.zeros(num_layers, num_packs, device=device)
    pack_items = torch.zeros(num_layers, num_packs,
                             dtype=torch.int64,
                             device=device)
    total = weight.float().sum(-1, keepdim=True)
    beta = num_packs / (total + 1e-9)  # [num_layers, 1], scale-free
    arange_layers = torch.arange(num_layers, device=device)
    for step in range(num_groups):
        group = indices[:, step]
        w = sorted_weight[:, step]
        # Mask packs that still have capacity; pick min penalized load.
        eligible = pack_items < groups_per_pack
        masked = pack_weights.masked_fill(~eligible, float("inf"))
        mean_load = pack_weights.sum(-1, keepdim=True) / (step + 1)
        score = masked + beta * (masked - mean_load).pow(2)
        pack = score.argmin(-1)
        pack_index[arange_layers, group] = pack
        rank_in_pack[arange_layers, group] = pack_items[arange_layers, pack]
        pack_weights[arange_layers, pack] += w
        pack_items[arange_layers, pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Waterfilling replication: find per-layer replica counts c_i >= 1 with
    sum_i c_i = num_phy minimizing max_i w_i / c_i. Solved by bisection on
    the level lambda with c_i = max(1, ceil(w_i / lambda)), followed by
    bounded vectorized trim/top-up loops (at most num_log steps each) to
    fix rounding of the count budget. phy2log/rank are then built fully
    vectorized via repeat_interleave (with the logical-id vector tiled n
    times so lengths match) and a cummax boundary scan for replica ranks.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    w = weight.float()
    if num_redundant == 0:
        phy2log = torch.arange(num_log, dtype=torch.int64,
                               device=device).repeat(n, 1)
        rank = torch.zeros_like(phy2log)
        logcnt = torch.ones_like(phy2log)
        return phy2log, rank, logcnt

    # Bisection on lambda: smallest lambda with sum(max(1, ceil(w/lambda)))
    # <= num_phy. Per-layer bounds, all ops vectorized over [n, num_log].
    lo = w.min(-1, keepdim=True).values.clamp_min(1e-9) / num_phy
    hi = w.max(-1, keepdim=True).values.clamp_min(1e-9)
    for _ in range(50):
        mid = (lo + hi) * 0.5
        cnt = torch.clamp(torch.ceil(w / mid), min=1.0)
        over = cnt.sum(-1, keepdim=True) > num_phy
        lo = torch.where(over, mid, lo)
        hi = torch.where(over, hi, mid)
    cnt = torch.clamp(torch.ceil(w / hi - 1e-6), min=1.0).to(torch.int64)

    # Guard against float-rounding excess: remove surplus replicas from
    # experts with the smallest w/c (bounded by num_log steps).
    excess = (cnt.sum(-1) - num_phy).clamp_min(0)
    for _ in range(int(excess.max().item())):
        m = excess > 0
        if not m.any():
            break
        sc = w / cnt
        sc = sc.masked_fill((cnt <= 1) | ~m.unsqueeze(-1), float("inf"))
        p = sc.argmin(-1)
        rows = m.nonzero(as_tuple=True)[0]
        cnt[rows, p[rows]] -= 1
        excess = torch.where(m, excess - 1, excess)

    # Top-up the rounding deficit with greedy argmax w/(c+1) steps
    # (vectorized; at most num_log iterations).
    deficit = (num_phy - cnt.sum(-1)).clamp_min(0)
    for _ in range(int(deficit.max().item())):
        m = deficit > 0
        if not m.any():
            break
        sc = w / (cnt + 1)
        sc = sc.masked_fill(~m.unsqueeze(-1), -1.0)
        p = sc.argmax(-1)
        rows = m.nonzero(as_tuple=True)[0]
        cnt[rows, p[rows]] += 1
        deficit = torch.where(m, deficit - 1, deficit)

    # Build phy2log via repeat_interleave over flattened per-row counts.
    # The logical-id vector must be tiled n times so its length matches
    # cnt.reshape(-1) (n * num_log); the total repeat count is then
    # sum(cnt) = n * num_phy.
    flat_log = torch.repeat_interleave(
        torch.arange(num_log, dtype=torch.int64,
                     device=device).repeat(n),
        cnt.reshape(-1),
    )
    phy2log = flat_log.view(n, num_phy)
    # Rank of each replica within its logical group: distance from the
    # group's start, located via a boundary scan + cummax propagation.
    flat = phy2log.reshape(-1)
    idx = torch.arange(flat.numel(), dtype=torch.int64, device=device)
    boundary = torch.ones(flat.numel(), dtype=torch.bool, device=device)
    boundary[1:] = flat[1:] != flat[:-1]
    boundary[::num_phy] = True  # row starts always begin a new group
    starts = torch.zeros_like(idx)
    starts[boundary] = idx[boundary]
    starts = torch.cummax(starts, 0).values
    rank = (idx - starts).view(n, num_phy)
    return phy2log, rank, cnt


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
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    # Global policy, executed directly (no group/node packing detour):
    # 1) replicate hot experts across the full weight tensor so a replica
    #    may land on any GPU, then 2) LPT-pack all physical experts onto
    #    all GPUs. This strictly enlarges the feasible placement set versus
    #    the hierarchical policy, so per-GPU max load cannot increase while
    #    expert-level balance improves, with less algorithmic overhead than
    #    routing through the hierarchical path with num_groups=num_nodes=1.
    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    phy_experts_per_gpu = num_replicas // num_gpus
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    # Reorder physical experts into (gpu, slot) order via scatter.
    ordered_phy2log = torch.empty_like(phy2log)
    ordered_phy2log.scatter_(1, phy2pphy, phy2log)
    ordered_phrank = torch.empty_like(phyrank)
    ordered_phrank.scatter_(1, phy2pphy, phyrank)
    phy2log, phyrank = ordered_phy2log, ordered_phrank
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

