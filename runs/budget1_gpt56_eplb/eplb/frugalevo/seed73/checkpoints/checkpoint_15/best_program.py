# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.
"""

# EVOLVE-BLOCK-START

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
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                 num_gpus // num_nodes)
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