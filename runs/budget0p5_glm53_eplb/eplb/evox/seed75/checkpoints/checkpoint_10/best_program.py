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
import numpy as np


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, each containing exactly n/m objects,
    minimizing the maximum pack weight.

    Approach: LPT greedy (sort descending, place each item in the lightest
    pack with remaining capacity, tie-breaking toward emptier packs), followed
    by a local swap-refinement that repeatedly exchanges items between the
    heaviest and lightest packs when it reduces the maximum pack weight.
    Implemented in NumPy for speed; ranks are recomputed after refinement so
    the (pack, rank) assignment remains a valid bijection.
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

    w = weight.float().cpu().numpy().astype(np.float64)
    order = np.argsort(-w, axis=1)
    pack_index = np.full((num_layers, num_groups), -1, dtype=np.int64)
    rank_in_pack = np.full((num_layers, num_groups), -1, dtype=np.int64)

    for i in range(num_layers):
        pack_w = np.zeros(num_packs, dtype=np.float64)
        pack_n = np.zeros(num_packs, dtype=np.int64)
        # LPT greedy with tie-break toward emptier packs
        for g in order[i]:
            cand = np.flatnonzero(pack_n < groups_per_pack)
            p = cand[np.lexsort((pack_n[cand], pack_w[cand]))[0]]
            pack_index[i, g] = p
            rank_in_pack[i, g] = pack_n[p]
            pack_w[p] += w[i, g]
            pack_n[p] += 1
        # Local swap refinement between heaviest and lightest packs:
        # evaluate all pairwise swaps between the two packs and take the
        # one minimizing max(new_hi, new_lo) if it beats the current max.
        for _ in range(2 * num_groups):
            hi = int(pack_w.argmax())
            lo = int(pack_w.argmin())
            if pack_w[hi] - pack_w[lo] <= 1e-12:
                break
            hi_items = np.flatnonzero(pack_index[i] == hi)
            lo_items = np.flatnonzero(pack_index[i] == lo)
            wi = w[i]
            diff = wi[hi_items][:, None] - wi[lo_items][None, :]
            cost = np.maximum(pack_w[hi] - diff, pack_w[lo] + diff)
            # Only swaps that move weight from hi to lo are useful
            cost = np.where(diff > 1e-12, cost, np.inf)
            a, b = np.unravel_index(int(np.argmin(cost)), cost.shape)
            if not np.isfinite(cost[a, b]) or cost[a, b] >= pack_w[hi]:
                break
            ga, gb = int(hi_items[a]), int(lo_items[b])
            pack_index[i, ga], pack_index[i, gb] = lo, hi
            pack_w[hi] += -wi[ga] + wi[gb]
            pack_w[lo] += -wi[gb] + wi[ga]
        # Recompute ranks so (pack, rank) stays a bijection
        for p in range(num_packs):
            items = np.flatnonzero(pack_index[i] == p)
            rank_in_pack[i, items] = np.arange(len(items), dtype=np.int64)

    return torch.from_numpy(pack_index), torch.from_numpy(rank_in_pack)


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, minimizing the maximum
    per-replica load.

    Approach: water-filling — repeatedly grant an extra replica to the expert
    with the highest load-per-replica (weight / count). Implemented in NumPy
    with a running per-row argmax over weight/count for speed.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    w = weight.float().cpu().numpy().astype(np.float64)
    phy2log = np.tile(np.arange(num_phy, dtype=np.int64), (n, 1))
    rank = np.zeros((n, num_phy), dtype=np.int64)
    logcnt = np.ones((n, num_log), dtype=np.int64)
    next_slot = np.full(n, num_log, dtype=np.int64)
    rows = np.arange(n)
    for _ in range(num_redundant):
        # Water-filling: give an extra replica to the expert with the
        # highest load-per-replica, fully vectorized across rows.
        redundant_indices = (w / logcnt).argmax(axis=-1)
        slots = next_slot.copy()
        phy2log[rows, slots] = redundant_indices
        rank[rows, slots] = logcnt[rows, redundant_indices]
        logcnt[rows, redundant_indices] += 1
        next_slot += 1
    return (torch.from_numpy(phy2log), torch.from_numpy(rank),
            torch.from_numpy(logcnt))


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

