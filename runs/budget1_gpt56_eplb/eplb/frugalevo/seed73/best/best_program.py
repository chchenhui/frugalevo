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
    """Globally allocate replicas, then LPT-pack every copy into GPU slots."""
    del num_groups, num_nodes
    num_layers, num_logical_experts = weight.shape
    assert num_physical_experts >= num_logical_experts
    assert num_physical_experts % num_gpus == 0

    # Replica allocation is global: hot experts receive copies based on their
    # current per-copy load, without first constraining them to a node.
    copy2log, copyrank, logcnt = replicate_experts(
        weight, num_physical_experts)

    copy_load = weight.gather(1, copy2log) / logcnt.gather(1, copy2log)
    gpu_index, slot_rank = balanced_packing(copy_load, num_gpus)

    slots_per_gpu = num_physical_experts // num_gpus
    physical_slot = gpu_index * slots_per_gpu + slot_rank

    pphy2log = torch.empty_like(copy2log)
    pphyrank = torch.empty_like(copyrank)
    pphy2log.scatter_(1, physical_slot, copy2log)
    pphyrank.scatter_(1, physical_slot, copyrank)

    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one topology because the hierarchical solver is node-independent."""
    del num_groups, num_nodes
    num_layers, num_logical_experts = weight.shape
    weight = weight.detach().float().cpu()

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