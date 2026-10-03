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
    n/m objects and the weights of all packs are as balanced as possible
    (cardinality-constrained LPT, vectorized across rows).

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
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    order = weight.float().argsort(-1, descending=True)  # [L, G]
    sorted_w = weight.float().gather(-1, order)  # [L, G]
    pack_weights = torch.zeros((num_layers, num_packs),
                               dtype=torch.float32,
                               device=weight.device)
    pack_items = torch.zeros((num_layers, num_packs),
                             dtype=torch.int64,
                             device=weight.device)
    pack_index_sorted = torch.empty((num_layers, num_groups),
                                    dtype=torch.int64,
                                    device=weight.device)
    rank_sorted = torch.empty_like(pack_index_sorted)
    INF = float("inf")
    for t in range(num_groups):
        # pick the least-loaded pack that still has capacity (per row)
        cand = torch.where(pack_items < groups_per_pack, pack_weights,
                           torch.full_like(pack_weights, INF))
        pack = cand.argmin(-1)  # [L]
        rank = pack_items.gather(1, pack.unsqueeze(1)).squeeze(1)
        pack_index_sorted[:, t] = pack
        rank_sorted[:, t] = rank
        pack_weights.scatter_add_(1, pack.unsqueeze(1),
                                  sorted_w[:, t].unsqueeze(1))
        pack_items.scatter_add_(
            1, pack.unsqueeze(1),
            torch.ones_like(pack).unsqueeze(1))
    pack_index = torch.empty_like(pack_index_sorted)
    pack_index.scatter_(1, order, pack_index_sorted)
    rank_in_pack = torch.empty_like(rank_sorted)
    rank_in_pack.scatter_(1, order, rank_sorted)
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas via water-filling on
    `weight / count`, such that the maximum per-replica load is minimized.

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
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    w = weight.float()
    for i in range(num_log, num_phy):
        redundant_indices = (w / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def _refine_gpu_balance(phy2log: torch.Tensor, phyrank: torch.Tensor,
                        slot_load: torch.Tensor, num_gpus: int,
                        max_rounds: int = 64):
    """
    Local search: per layer, swap one slot between the heaviest and the
    lightest GPU to minimize the max of the two resulting GPU loads.
    Fully vectorized across layers; all per_gpu x per_gpu candidate swaps
    are evaluated in a single batched tensor op each round.
    """
    L, num_phy = phy2log.shape
    P = num_phy // num_gpus
    if num_gpus < 2 or P < 1:
        return phy2log, phyrank
    dev = phy2log.device
    rows = torch.arange(L, device=dev)
    base = torch.arange(P, device=dev)
    phy2log = phy2log.clone()
    phyrank = phyrank.clone()
    slot_load = slot_load.clone()

    for _ in range(max_rounds):
        gpu_load = slot_load.view(L, num_gpus, P).sum(-1)  # [L, G]
        h = gpu_load.argmax(-1)  # [L]
        l = gpu_load.argmin(-1)  # [L]
        lh = gpu_load[rows, h]
        ll = gpu_load[rows, l]
        if bool((lh - ll <= 1e-12).all()):
            break
        hi = h.unsqueeze(1) * P + base  # [L, P] heavy slot ids
        li = l.unsqueeze(1) * P + base  # [L, P] light slot ids
        wa = slot_load.gather(1, hi).unsqueeze(2)  # [L, P, 1]
        wb = slot_load.gather(1, li).unsqueeze(1)  # [L, 1, P]
        new_h = lh.view(L, 1, 1) - wa + wb
        new_l = ll.view(L, 1, 1) + wa - wb
        worst = torch.maximum(new_h, new_l).flatten(1)  # [L, P*P]
        best = worst.argmin(1)  # [L]
        bv = worst[rows, best]
        cur = torch.maximum(lh, ll)
        mask = bv < cur - 1e-12
        if not bool(mask.any()):
            break
        a = best // P
        b = best % P
        # no-op self swap where the layer does not improve
        zero = torch.zeros_like(h)
        ia = torch.where(mask, h * P + a, zero)
        ib = torch.where(mask, l * P + b, zero)
        for buf in (phy2log, phyrank, slot_load):
            va = buf[rows, ia].clone()
            vb = buf[rows, ib].clone()
            buf[rows, ia] = vb
            buf[rows, ib] = va
    return phy2log, phyrank


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Global policy: water-filling replication of all logical experts, then
    cardinality-constrained LPT packing of physical slots onto GPUs, then
    a heavy<->light GPU swap local search. All computation stays on the
    input device (no CPU round-trip).

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
    device = weight.device
    w = weight.float()

    # Stage 1: global water-filling replication
    phy2log, phyrank, logcnt = replicate_experts(w, num_replicas)

    # Stage 2: LPT packing of physical slots onto GPUs
    slot_load = (w / logcnt).gather(1, phy2log)  # [L, num_phy]
    pack_index, rank_in_pack = balanced_packing(slot_load, num_gpus)
    per_gpu = num_replicas // num_gpus
    pos = pack_index * per_gpu + rank_in_pack  # destination physical slot
    inv = torch.empty_like(pos)
    inv.scatter_(1, pos,
                 torch.arange(num_replicas, dtype=torch.int64,
                              device=device).expand(pos.shape))
    phy2log = phy2log.gather(1, inv)
    phyrank = phyrank.gather(1, inv)
    slot_load = slot_load.gather(1, inv)

    # Stage 3: heavy/light GPU swap local search
    phy2log, phyrank = _refine_gpu_balance(phy2log, phyrank, slot_load,
                                           num_gpus)

    # Build logical -> physical map
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
