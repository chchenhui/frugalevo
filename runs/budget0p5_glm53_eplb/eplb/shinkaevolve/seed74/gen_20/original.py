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

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Stage 1: weighted balanced packing (heap-based LPT with capacity)
# ---------------------------------------------------------------------------

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects into m packs, each holding exactly n/m objects,
    minimizing load imbalance (LPT with cardinality constraint).

    Returns pack_index [X, n] and rank_in_pack [X, n].
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                   dtype=torch.int64).expand(num_layers, -1)
        rank_in_pack = torch.zeros(num_layers, num_groups, dtype=torch.int64)
        return pack_index, rank_in_pack

    w = weight.float().cpu().numpy()
    order = np.argsort(-w, axis=1, kind="stable")
    pack_index = np.empty((num_layers, num_groups), dtype=np.int64)
    rank_in_pack = np.empty((num_layers, num_groups), dtype=np.int64)

    for i in range(num_layers):
        w_i = w[i]
        # heap of (load, tiebreak_id, pack, items)
        heap = [(0.0, p, p, 0) for p in range(num_packs)]
        heapq.heapify(heap)
        counts = [0] * num_packs
        loads = [0.0] * num_packs
        for group in order[i]:
            # pop until a non-full pack is at the top
            while heap and heap[0][3] >= groups_per_pack:
                heapq.heappop(heap)
            _, _, pack, _ = heap[0]
            pack_index[i, group] = pack
            rank_in_pack[i, group] = counts[pack]
            counts[pack] += 1
            loads[pack] += w_i[group]
            if counts[pack] < groups_per_pack:
                heapq.heapreplace(heap, (loads[pack], pack, pack, counts[pack]))
            else:
                heapq.heappop(heap)

    return torch.from_numpy(pack_index), torch.from_numpy(rank_in_pack)


# ---------------------------------------------------------------------------
# Stage 2: replication via water-filling greedy
# ---------------------------------------------------------------------------

def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate num_log experts to num_phy physical slots minimizing max load
    per logical expert (each replica serves weight/count).

    Returns phy2log [X, num_phy], rank [X, num_phy], logcnt [X, num_log].
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    w = weight.float().cpu().numpy()
    phy2log = np.tile(np.arange(num_phy, dtype=np.int64), (n, 1))
    rank = np.zeros((n, num_phy), dtype=np.int64)
    logcnt = np.ones((n, num_log), dtype=np.int64)

    if num_redundant > 0:
        # per-load for each logical expert, replicated across rows
        per_load = w.copy()
        rows = np.arange(n)
        for i in range(num_log, num_phy):
            # pick the expert with the highest per-replica load in each row
            best = per_load.argmax(axis=1)
            r = logcnt[rows, best]
            phy2log[:, i] = best
            rank[:, i] = r
            logcnt[rows, best] = r + 1
            per_load[rows, best] = w[rows, best] / (r + 1)

    return (torch.from_numpy(phy2log), torch.from_numpy(rank),
            torch.from_numpy(logcnt))


# ---------------------------------------------------------------------------
# Stage 3: hierarchical assembly
# ---------------------------------------------------------------------------

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
            1, perm,
            torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return inv

    # 3a. replicate experts globally across the full physical pool, so any
    # logical expert can receive replicas regardless of node membership
    phy2log, phyrank, logcnt = replicate_experts(
        weight, num_physical_experts)

    # 3b. pack physical experts (with replica-shared loads) to GPUs globally
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2log = phy2log.gather(-1, pphy2phy)
    pphyrank = phyrank.gather(-1, pphy2phy)
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
        -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64).expand(num_layers, -1))
    return phy2log, log2phy, logcnt

# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
