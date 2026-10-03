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


def replicate_experts_global(weight: np.ndarray,
                            num_phy: int):
    """
    Globally replicate num_log logical experts into num_phy physical slots.

    Parameters:
        weight: [L, num_log] per-layer token load of each logical expert

    Returns:
        phy2log: [L, num_phy] logical expert id of each physical slot
        rank: [L, num_phy] replica rank of each physical slot
        logcnt: [L, num_log] number of replicas per logical expert
    """
    L, num_log = weight.shape
    redundant = num_phy - num_log
    assert redundant >= 0

    phy2log = np.tile(np.arange(num_phy, dtype=np.int64), (L, 1))
    rank = np.zeros((L, num_phy), dtype=np.int64)
    logcnt = np.ones((L, num_log), dtype=np.int64)

    if redundant > 0:
        per_load = weight.copy()
        rows = np.arange(L)
        for i in range(num_log, num_phy):
            best = per_load.argmax(axis=1)
            r = logcnt[rows, best]
            phy2log[:, i] = best
            rank[:, i] = r
            logcnt[rows, best] = r + 1
            per_load[rows, best] = weight[rows, best] / (r + 1)

    return phy2log, rank, logcnt


def pack_to_gpus(load: np.ndarray,
                 num_gpus: int,
                 per_gpu: int) -> np.ndarray:
    """
    Cardinality-constrained LPT packing of physical experts onto GPUs,
    followed by swap local search between the heaviest/lightest GPU.

    Parameters:
        load: [L, n] load of each physical expert slot (n = num_gpus*per_gpu)
        num_gpus: number of GPUs
        per_gpu: physical experts per GPU

    Returns:
        gpu: [L, n] gpu index assigned to each slot
    """
    L, n = load.shape
    gpu = np.empty((L, n), dtype=np.int64)

    for i in range(L):
        li = load[i]
        order = np.argsort(-li, kind="stable")

        # --- greedy LPT with exact cardinality via heap ---
        heap = [(0.0, p, p, 0) for p in range(num_gpus)]
        heapq.heapify(heap)
        loads = [0.0] * num_gpus
        cnt = [0] * num_gpus
        for idx in order:
            while heap[0][3] >= per_gpu:
                heapq.heappop(heap)
            _, _, p, _ = heap[0]
            gpu[i, idx] = p
            cnt[p] += 1
            loads[p] += li[idx]
            if cnt[p] < per_gpu:
                heapq.heapreplace(heap, (loads[p], p, p, cnt[p]))
            else:
                heapq.heappop(heap)

        # --- pairwise swap refinement (heaviest <-> lightest GPU) ---
        if num_gpus > 1 and per_gpu >= 1:
            row_gpu = gpu[i]
            for _ in range(4 * num_gpus):
                h = int(np.argmax(loads))
                l = int(np.argmin(loads))
                if h == l or loads[h] - loads[l] < 1e-12:
                    break
                ih = np.flatnonzero(row_gpu == h)
                il = np.flatnonzero(row_gpu == l)
                wa = li[ih][:, None]  # items leaving heavy
                wb = li[il][None, :]  # items entering heavy
                new_h = loads[h] - wa + wb
                new_l = loads[l] + wa - wb
                worst = np.maximum(new_h, new_l)
                k = int(np.argmin(worst))
                a, b = divmod(k, len(il))
                if worst.flat[k] >= max(loads[h], loads[l]) - 1e-12:
                    break
                A, B = ih[a], il[b]
                row_gpu[A] = l
                row_gpu[B] = h
                loads[h] = float(new_h[a, b])
                loads[l] = float(new_l[a, b])

    return gpu


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
        num_nodes: number of server nodes
        num_gpus: number of GPUs

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_replicas >= num_logical_experts
    assert num_replicas % num_gpus == 0

    w = weight.float().cpu().numpy().astype(np.float64)
    num_phy = num_replicas
    per_gpu = num_phy // num_gpus

    # Stage 1: global replication (water-filling over all layers at once)
    phy2log, rep_rank, logcnt = replicate_experts_global(w, num_phy)

    # Stage 2: pack physical expert slots onto GPUs
    per_slot_load = w / logcnt  # [L, num_log], then expand to slots
    slot_load = np.take_along_axis(per_slot_load, phy2log, axis=1)

    gpu = pack_to_gpus(slot_load, num_gpus, per_gpu)

    # rank (slot within GPU) for every expert slot
    srt = np.argsort(gpu, axis=1, kind="stable")
    sorted_gpu = np.take_along_axis(gpu, srt, axis=1)
    pos = np.arange(num_phy, dtype=np.int64)[None, :].repeat(num_layers, 0)
    ranks_sorted = pos - sorted_gpu * per_gpu  # each gpu fills a contiguous block
    slot_in_gpu = np.empty_like(gpu)
    np.put_along_axis(slot_in_gpu, srt, ranks_sorted, axis=1)

    # physical expert id = gpu * per_gpu + slot_in_gpu
    phys_id = gpu * per_gpu + slot_in_gpu

    # invert: for each physical id, which slot index it came from
    inv = np.empty_like(phys_id)
    rows = np.arange(num_layers)[:, None]
    inv[rows, phys_id] = np.arange(num_phy, dtype=np.int64)[None, :]

    phy2log_final = np.take_along_axis(phy2log, inv, axis=1)
    phyrank_final = np.take_along_axis(rep_rank, inv, axis=1)

    physical_to_logical_map = torch.from_numpy(phy2log_final.copy())
    phyrank = torch.from_numpy(phyrank_final.copy())
    expert_count = torch.from_numpy(logcnt.copy())

    # build logical -> physical map
    num_redundant_experts = num_phy - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        physical_to_logical_map * maxlogcnt + phyrank,
        torch.arange(num_phy, dtype=torch.int64).expand(num_layers, -1),
    )
    return physical_to_logical_map, log2phy, expert_count

# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

