# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.
"""

# EVOLVE-BLOCK-START

import itertools
import numpy as np

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack weighted objects into equal-cardinality packs using deterministic LPT.

    The general path batches the independent per-layer LPT recurrences.  It
    retains the original LPT rule: descending item order, least loaded
    non-full pack, and lowest pack index on ties.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if num_packs == 1:
        pack_index = torch.zeros(weight.shape,
                                 dtype=torch.int64,
                                 device=weight.device)
        rank_in_pack = torch.arange(num_groups,
                                    dtype=torch.int64,
                                    device=weight.device).expand(
                                        num_layers, -1)
        return pack_index, rank_in_pack

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Callers use CPU load metrics.  Keep the batched recurrence in NumPy:
    # stable sorting preserves deterministic LPT order, while float64 loads
    # retain the scalar accumulation precision of the original implementation.
    values = weight.detach().float().cpu().contiguous().numpy()
    order = np.argsort(-values, axis=1, kind="stable")
    loads = np.zeros((num_layers, num_packs), dtype=values.dtype)
    counts = np.zeros((num_layers, num_packs), dtype=np.int64)
    pack_index = np.empty((num_layers, num_groups), dtype=np.int64)
    rank_in_pack = np.empty((num_layers, num_groups), dtype=np.int64)
    rows = np.arange(num_layers)

    for position in range(num_groups):
        group = order[:, position]
        # Mask only the capacity-ineligible entries.  The temporary is
        # contiguous and remains entirely batched across layers, preserving
        # lowest-index tie breaking without Python per-layer heap operations.
        eligible_loads = np.where(counts < groups_per_pack, loads,
                                  np.finfo(values.dtype).max)
        pack = eligible_loads.argmin(axis=1)
        old_count = counts[rows, pack]
        pack_index[rows, group] = pack
        rank_in_pack[rows, group] = old_count
        loads[rows, pack] += values[rows, group]
        counts[rows, pack] = old_count + 1

    return (torch.from_numpy(pack_index).to(weight.device),
            torch.from_numpy(rank_in_pack).to(weight.device))


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate redundant replicas greedily by largest current per-replica load.
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

    for phy in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, phy] = redundant_indices
        rank[:, phy] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """Pack groups, replicate experts, then pack replicas onto GPUs."""
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

    tokens_per_group = weight.unflatten(
        -1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64,
                             device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    def repaired_packing(
            values: torch.Tensor,
            packs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply exact pair repairs, then two bounded global critical-GPU swaps."""
        packed, ranks = balanced_packing(values, packs)
        capacity = values.shape[1] // packs
        if capacity > 8 or packs < 2:
            return packed, ranks

        assignment = packed.numpy().copy()
        weights = values.numpy()
        rank_array = ranks.numpy().copy()
        subsets = np.asarray(
            tuple(itertools.combinations(range(2 * capacity), capacity)),
            dtype=np.int64,
        )
        positions = np.arange(capacity, dtype=np.int64)

        for row in range(values.shape[0]):
            loads = np.zeros(packs, dtype=weights.dtype)
            members_by_gpu = []
            for gpu in range(packs):
                members = np.flatnonzero(assignment[row] == gpu)
                members_by_gpu.append(members)
                loads[gpu] = weights[row, members].sum(dtype=weights.dtype)

            heavy = int(np.argmax(loads))
            for light in np.argsort(loads, kind="stable")[:2]:
                light = int(light)
                if light == heavy:
                    continue
                joined = np.concatenate((members_by_gpu[heavy],
                                         members_by_gpu[light]))
                joined_weights = weights[row, joined]
                selected_load = joined_weights[subsets].sum(axis=1)
                other_load = joined_weights.sum(dtype=weights.dtype) - selected_load
                pair_peak = np.maximum(selected_load, other_load)
                pair_spread = selected_load * selected_load + other_load * other_load
                incumbent_peak = max(loads[heavy], loads[light])
                incumbent_spread = loads[heavy] * loads[heavy] + loads[light] * loads[light]
                best = int(np.lexsort((pair_spread, pair_peak))[0])
                if (pair_peak[best] > incumbent_peak or
                        (pair_peak[best] == incumbent_peak and
                         pair_spread[best] >= incumbent_spread)):
                    continue
                selected = np.zeros(2 * capacity, dtype=bool)
                selected[subsets[best]] = True
                assignment[row, joined[selected]] = heavy
                assignment[row, joined[~selected]] = light
                loads[heavy] = selected_load[best]
                loads[light] = other_load[best]
                for gpu in (heavy, light):
                    members = np.flatnonzero(assignment[row] == gpu)
                    members_by_gpu[gpu] = members
                    order = np.argsort(-weights[row, members], kind="stable")
                    rank_array[row, members[order]] = positions

            # Each pass examines every destination GPU and every pair of
            # equal-cardinality physical-expert positions.  Only a swap that
            # improves the complete (peak, squared-spread) objective is kept.
            for _ in range(2):
                heavy = int(np.argmax(loads))
                source = members_by_gpu[heavy]
                source_weights = weights[row, source]
                incumbent = (float(loads.max()), float(np.square(loads).sum()))
                best_key = incumbent
                best_gpu = best_i = best_j = -1

                for gpu in range(packs):
                    if gpu == heavy:
                        continue
                    target = members_by_gpu[gpu]
                    delta = weights[row, target][None, :] - source_weights[:, None]
                    new_heavy = loads[heavy] + delta
                    new_target = loads[gpu] - delta
                    fixed = loads.copy()
                    fixed[heavy] = -np.inf
                    fixed[gpu] = -np.inf
                    fixed_peak = fixed.max()
                    candidate_peak = np.maximum(
                        fixed_peak, np.maximum(new_heavy, new_target))
                    candidate_spread = (
                        np.square(loads).sum() - loads[heavy] ** 2 -
                        loads[gpu] ** 2 + new_heavy ** 2 + new_target ** 2)
                    flat = np.lexsort(
                        (candidate_spread.ravel(), candidate_peak.ravel()))[0]
                    candidate = (float(candidate_peak.ravel()[flat]),
                                 float(candidate_spread.ravel()[flat]))
                    if candidate < best_key:
                        best_key = candidate
                        best_gpu = gpu
                        best_i, best_j = np.unravel_index(
                            flat, candidate_peak.shape)

                if best_gpu < 0:
                    break

                target = members_by_gpu[best_gpu]
                a, b = int(source[best_i]), int(target[best_j])
                wa, wb = weights[row, a], weights[row, b]
                assignment[row, a], assignment[row, b] = best_gpu, heavy
                loads[heavy] += wb - wa
                loads[best_gpu] += wa - wb
                members_by_gpu[heavy][best_i] = b
                members_by_gpu[best_gpu][best_j] = a

                for gpu in (heavy, best_gpu):
                    members = members_by_gpu[gpu]
                    order = np.argsort(-weights[row, members], kind="stable")
                    rank_array[row, members[order]] = positions

        return (torch.from_numpy(assignment).to(values.device),
                torch.from_numpy(rank_array).to(values.device))

    pack_index, rank_in_pack = repaired_packing(
        tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(
                     0,
                     num_logical_experts,
                     num_logical_experts // num_nodes,
                     dtype=torch.int64,
                     device=weight.device,
                 ).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)

    pphyrank = torch.gather(phyrank, -1, pphy2phy).view(num_layers, -1)
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
    Select the lowest-peak feasible topology per layer.  Equal peaks are
    resolved by full GPU-load dispersion, then by retaining node locality.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.detach().float().cpu()

    if num_groups % num_nodes == 0:
        hierarchical_layout = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
        global_layout = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

        def load_statistics(layout):
            candidate_phy2log, _, candidate_logcnt = layout
            physical_load = weight.gather(
                1, candidate_phy2log
            ) / candidate_logcnt.gather(1, candidate_phy2log)
            gpu_load = physical_load.reshape(
                num_layers, num_gpus, -1).sum(dim=-1)
            return gpu_load.amax(dim=-1), gpu_load.square().sum(dim=-1)

        hierarchical_peak, hierarchical_spread = load_statistics(
            hierarchical_layout)
        global_peak, global_spread = load_statistics(global_layout)

        use_global = ((global_peak < hierarchical_peak) |
                      ((global_peak == hierarchical_peak) &
                       (global_spread < hierarchical_spread)))
        phy2log = torch.where(
            use_global[:, None], global_layout[0], hierarchical_layout[0])
        phyrank = torch.where(
            use_global[:, None], global_layout[1], hierarchical_layout[1])
        logcnt = torch.where(
            use_global[:, None], global_layout[2], hierarchical_layout[2])
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    maxlogcnt = num_replicas - num_logical_experts + 1
    log2phy = torch.full(
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