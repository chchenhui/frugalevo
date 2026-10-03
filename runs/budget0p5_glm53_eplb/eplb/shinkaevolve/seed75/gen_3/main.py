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

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Vectorized greedy building blocks
# ---------------------------------------------------------------------------

def _balanced_packing_np(weight: np.ndarray,
                         num_packs: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack n weighted objects to m packs, such that each pack contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    All rows (layers) are processed in lockstep: for each item (in sorted,
    descending order), every row greedily picks its least-loaded pack that
    still has capacity. Tie-breaking (lowest pack index) matches the original
    sequential algorithm.

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
        pack_index = np.tile(np.arange(num_groups, dtype=np.int64),
                             (num_layers, 1))
        rank_in_pack = np.zeros((num_layers, num_groups), dtype=np.int64)
        return pack_index, rank_in_pack

    # Descending sort per row (stable -> smallest original index first on
    # ties, matching the original min-scan behavior).
    order = np.argsort(-weight, axis=1, kind="stable")
    w_sorted = np.take_along_axis(weight, order, axis=1)

    loads = np.zeros((num_layers, num_packs), dtype=weight.dtype)
    counts = np.zeros((num_layers, num_packs), dtype=np.int64)
    pack_index = np.full((num_layers, num_groups), -1, dtype=np.int64)
    rank_in_pack = np.full((num_layers, num_groups), -1, dtype=np.int64)

    rows = np.arange(num_layers)
    for t in range(num_groups):
        # Packs at capacity are masked out with +inf.
        eff = np.where(counts >= groups_per_pack, np.inf, loads)
        p = eff.argmin(axis=1)  # ties -> lowest pack index
        c = counts[rows, p]
        tgt = order[:, t]
        pack_index[rows, tgt] = p
        rank_in_pack[rows, tgt] = c
        loads[rows, p] += w_sorted[:, t]
        counts[rows, p] += 1
    return pack_index, rank_in_pack


def _replicate_experts_np(weight: np.ndarray,
                          num_phy: int) -> tuple[np.ndarray, np.ndarray,
                                                 np.ndarray]:
    """
    Replicate `num_log` experts to `num_phy` replicas, greedily replicating
    the expert with the highest per-replica load at each step.

    The per-replica load matrix (weight / count) is maintained incrementally
    so each redundant-expert step costs one vectorized argmax instead of a
    full re-division and max reduction.

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

    phy2log = np.tile(np.arange(num_phy, dtype=np.int64), (n, 1))
    rank = np.zeros((n, num_phy), dtype=np.int64)
    logcnt = np.ones((n, num_log), dtype=np.int64)

    rows = np.arange(n)
    ratio = weight.astype(weight.dtype, copy=True)  # weight / 1
    for i in range(num_log, num_phy):
        idx = ratio.argmax(axis=1)
        phy2log[:, i] = idx
        rank[:, i] = logcnt[rows, idx]
        logcnt[rows, idx] += 1
        ratio[rows, idx] = weight[rows, idx] / logcnt[rows, idx]
    return phy2log, rank, logcnt


# ---------------------------------------------------------------------------
# Permutation helpers (row-wise scatter inverse)
# ---------------------------------------------------------------------------

def _inverse_perm_np(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    rows = np.arange(perm.shape[0])
    inv[rows[:, None], perm] = np.arange(perm.shape[1],
                                         dtype=np.int64)[None, :]
    return inv


# ---------------------------------------------------------------------------
# Hierarchical (node-aware) policy
# ---------------------------------------------------------------------------

def _rebalance_experts_hierarchical_np(
    weight: np.ndarray,
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
    gpus_per_node = num_gpus // num_nodes
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes
    tokens_per_group = weight.reshape(num_layers, num_groups,
                                       group_size).sum(-1)
    group_pack_index, group_rank_in_pack = _balanced_packing_np(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size)[:, :, None] +
                np.arange(group_size, dtype=np.int64)[None, None,
                                                      :]).reshape(
                                                          num_layers, -1)
    mlog2log = _inverse_perm_np(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = np.take_along_axis(
        weight, mlog2log, axis=1).reshape(-1,
                                          num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = _replicate_experts_np(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = np.take_along_axis(tokens_per_mlog / mlogcnt,
                                        phy2mlog, axis=1)
    pack_index, rank_in_pack = _balanced_packing_np(tokens_per_phy,
                                                    gpus_per_node)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = _inverse_perm_np(phy2pphy)

    pphy2mlog = np.take_along_axis(phy2mlog, pphy2phy, axis=1)
    pphy2mlog = (
        pphy2mlog.reshape(num_layers, num_nodes, -1) +
        np.arange(0, num_logical_experts, num_logical_experts // num_nodes,
                  dtype=np.int64)[None, :, None]).reshape(num_layers, -1)
    pphy2log = np.take_along_axis(log2mlog, pphy2mlog, axis=1)
    pphyrank = np.take_along_axis(phyrank, pphy2phy,
                                  axis=1).reshape(num_layers, -1)
    logcnt = np.take_along_axis(mlogcnt.reshape(num_layers, -1), log2mlog,
                                axis=1)
    return pphy2log, pphyrank, logcnt


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    weight_np = weight.float().cpu().numpy()

    if num_groups % num_nodes == 0:
        # hierarchical (node-aware) load-balance policy
        phy2log, phyrank, logcnt = _rebalance_experts_hierarchical_np(
            weight_np, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global load-balance policy
        phy2log, phyrank, logcnt = _rebalance_experts_hierarchical_np(
            weight_np, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = np.full((num_layers, num_logical_experts, maxlogcnt), -1,
                      dtype=np.int64)
    log2phy.reshape(num_layers, -1)[
        np.arange(num_layers)[:, None],
        phy2log * maxlogcnt + phyrank,
    ] = np.arange(num_replicas, dtype=np.int64)[None, :]

    return (
        torch.from_numpy(phy2log),
        torch.from_numpy(log2phy),
        torch.from_numpy(logcnt),
    )

# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

