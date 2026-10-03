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

import heapq

import torch


def _refine_packing(
    bins: list[list[int]],
    values: list[float],
    loads: list[float],
) -> None:
    """Bounded equal-cardinality swap refinement for a greedy packing."""
    num_packs = len(bins)
    if num_packs <= 1:
        return

    # A few improving swaps recover cases where cardinality-constrained LPT
    # leaves a heavy item paired with an unnecessarily light item.
    for _ in range(min(2 * num_packs, 16)):
        high = max(range(num_packs), key=loads.__getitem__)
        low = min(range(num_packs), key=loads.__getitem__)
        current_max = max(loads)
        current_range = current_max - min(loads)

        if high == low or not bins[high] or not bins[low]:
            break

        other_loads = [
            loads[pack] for pack in range(num_packs)
            if pack != high and pack != low
        ]
        best_pair: tuple[int, int, float, float] | None = None
        best_objective = (current_max, current_range)

        # Test swaps only between the most and least loaded packs. This keeps
        # refinement inexpensive while targeting the maximum-load objective.
        for high_pos, high_item in enumerate(bins[high]):
            high_weight = values[high_item]
            for low_pos, low_item in enumerate(bins[low]):
                low_weight = values[low_item]
                new_high = loads[high] - high_weight + low_weight
                new_low = loads[low] - low_weight + high_weight
                candidate_loads = other_loads + [new_high, new_low]
                candidate_max = max(candidate_loads)
                candidate_range = candidate_max - min(candidate_loads)
                objective = (candidate_max, candidate_range)

                if objective < best_objective:
                    best_objective = objective
                    best_pair = (high_pos, low_pos, new_high, new_low)

        if best_pair is None:
            break

        high_pos, low_pos, new_high, new_low = best_pair
        bins[high][high_pos], bins[low][low_pos] = (
            bins[low][low_pos],
            bins[high][high_pos],
        )
        loads[high] = new_high
        loads[low] = new_low


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
        pack_index = torch.arange(
            weight.size(-1), dtype=torch.int64,
            device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Sorting remains vectorized. The following packing loop operates on
    # ordinary Python numbers, avoiding repeated scalar tensor synchronization
    # and O(num_packs) scans from the original implementation.
    cpu_weight = weight.float().cpu()
    sorted_indices = cpu_weight.sort(-1, descending=True).indices.tolist()
    weight_rows = cpu_weight.tolist()

    all_pack_indices: list[list[int]] = []
    all_ranks: list[list[int]] = []

    for layer in range(num_layers):
        values = weight_rows[layer]
        bins: list[list[int]] = [[] for _ in range(num_packs)]
        loads = [0.0] * num_packs

        # Heap entries are (load, pack_id), which also gives deterministic
        # low-index tie breaking and O(log(num_packs)) pack selection.
        available = [(0.0, pack) for pack in range(num_packs)]
        heapq.heapify(available)

        for item in sorted_indices[layer]:
            current_load, pack = heapq.heappop(available)
            bins[pack].append(item)
            current_load += values[item]
            loads[pack] = current_load

            if len(bins[pack]) < groups_per_pack:
                heapq.heappush(available, (current_load, pack))

        _refine_packing(bins, values, loads)

        pack_index = [0] * num_groups
        rank_in_pack = [0] * num_groups
        for pack, items in enumerate(bins):
            for rank, item in enumerate(items):
                pack_index[item] = pack
                rank_in_pack[item] = rank

        all_pack_indices.append(pack_index)
        all_ranks.append(rank_in_pack)

    return (
        torch.tensor(all_pack_indices, dtype=torch.int64),
        torch.tensor(all_ranks, dtype=torch.int64),
    )


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
    phy2log = torch.arange(
        num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    rows = torch.arange(n, dtype=torch.int64, device=device)

    # This greedy allocation is optimal for the discrete minimax replica-load
    # objective: each additional slot goes to the largest current load/share.
    for phy_index in range(num_log, num_phy):
        logical_index = torch.div(
            weight, logcnt, rounding_mode=None).max(dim=-1).indices
        phy2log[:, phy_index] = logical_index
        rank[:, phy_index] = logcnt[rows, logical_index]
        logcnt[rows, logical_index] += 1

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
            torch.arange(
                perm.size(1), dtype=torch.int64,
                device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack logical expert groups to nodes.
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (
        ((group_pack_index * groups_per_node + group_rank_in_pack) *
         group_size).unsqueeze(-1) +
        torch.arange(
            group_size, dtype=torch.int64,
            device=group_pack_index.device)
    ).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts inside each node.
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs inside every node.
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy, num_gpus // num_nodes)
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
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_replicas, dtype=torch.int64,
            device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

