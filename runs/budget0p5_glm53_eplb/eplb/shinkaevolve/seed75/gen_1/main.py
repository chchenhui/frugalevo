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


def _replicate_experts(weight: torch.Tensor, num_phy: int):
    """Vectorized greedy replication: repeatedly clone the expert with the
    highest load/replicas ratio. Returns per-replica logical ids, ranks,
    counts and per-replica loads."""
    num_layers, num_log = weight.shape
    device = weight.device
    logcnt = torch.ones(num_layers, num_log, dtype=torch.int64,
                        device=device)
    arangen = torch.arange(num_layers, device=device)
    for _ in range(num_log, num_phy):
        idx = (weight / logcnt).argmax(dim=-1)
        logcnt[arangen, idx] += 1
    # Build physical list: each logical expert repeated logcnt times,
    # along with its replica rank.
    phy2log = torch.empty(num_layers, num_phy, dtype=torch.int64,
                          device=device)
    phyrank = torch.empty(num_layers, num_phy, dtype=torch.int64,
                          device=device)
    arange_log = torch.arange(num_log, dtype=torch.int64, device=device)
    for l in range(num_layers):
        cnt = logcnt[l]
        phy2log[l] = arange_log.repeat_interleave(cnt)
        # rank within duplicates
        phyrank[l] = torch.cat(
            [torch.arange(c, dtype=torch.int64, device=device)
             for c in cnt.tolist()])
    tokens_phy = weight.gather(-1, phy2log) / logcnt.gather(-1, phy2log)
    return phy2log, phyrank, logcnt, tokens_phy


def _snake_assign(tokens_phy: torch.Tensor, num_gpus: int, epg: int):
    """Snake-draft LPT: sort desc, deal rounds alternating direction so each
    GPU gets exactly epg experts. Fully vectorized."""
    num_layers, num_phy = tokens_phy.shape
    order = tokens_phy.sort(-1, descending=True).indices
    pos = torch.arange(num_phy, device=tokens_phy.device)
    pos = pos.unsqueeze(0).expand(num_layers, -1)
    rnd = pos // epg
    within = pos % epg
    gpu = torch.where(rnd % 2 == 1, num_gpus - 1 - within, within)
    gpu_of = torch.empty_like(gpu)
    gpu_of.scatter_(1, order, gpu)
    return gpu_of


def _refine(tokens_phy: torch.Tensor, gpu_of: torch.Tensor, num_gpus: int,
            epg: int, iters: int = 40):
    """Vectorized pairwise-swap local search: swap the heaviest item of the
    max-load GPU with the lightest item of the min-load GPU whenever the
    maximum load strictly decreases."""
    num_layers, num_phy = tokens_phy.shape
    device = tokens_phy.device
    for _ in range(iters):
        loads = torch.zeros(num_layers, num_gpus, device=device)
        loads.scatter_add_(1, gpu_of, tokens_phy)
        worst = loads.argmax(-1, keepdim=True)          # [L,1]
        best = loads.argmin(-1, keepdim=True)           # [L,1]
        max_load = loads.max(-1, keepdim=True).values
        min_load = loads.min(-1, keepdim=True).values
        # heaviest item on worst gpu
        mask_w = gpu_of == worst
        w_big = torch.where(mask_w, tokens_phy,
                            torch.full_like(tokens_phy, float("-inf")))
        big_idx = w_big.argmax(-1)                      # [L]
        w_big_val = tokens_phy.gather(1, big_idx.unsqueeze(1)).squeeze(1)
        # lightest item on best gpu
        mask_b = gpu_of == best
        w_small = torch.where(mask_b, tokens_phy,
                              torch.full_like(tokens_phy, float("inf")))
        small_idx = w_small.argmin(-1)                  # [L]
        w_small_val = tokens_phy.gather(1, small_idx.unsqueeze(1)).squeeze(1)
        # hypothetical loads after swap
        new_worst = max_load.squeeze(1) - w_big_val + w_small_val
        new_best = min_load.squeeze(1) - w_small_val + w_big_val
        improve = (new_worst < max_load.squeeze(1)) & \
                  (new_best < max_load.squeeze(1)) & \
                  (w_big_val > w_small_val)
        if not improve.any():
            break
        gi = torch.arange(num_layers, device=device)
        # swap gpus of the two items
        gw = gpu_of[gi, big_idx].clone()
        gb = gpu_of[gi, small_idx].clone()
        gpu_of[gi, big_idx] = torch.where(improve, gb, gw)
        gpu_of[gi, small_idx] = torch.where(improve, gw, gb)
    return gpu_of


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
    num_redundant = num_replicas - num_logical_experts
    assert num_redundant >= 0
    assert num_replicas % num_gpus == 0
    epg = num_replicas // num_gpus
    device = weight.device

    # Stage 1: replicate experts (vectorized greedy by load/replicas).
    phy2log, phyrank, logcnt, tokens_phy = _replicate_experts(
        weight, num_replicas)

    # Stage 2: snake-draft LPT assignment to GPUs (exactly epg per GPU).
    gpu_of = _snake_assign(tokens_phy, num_gpus, epg)

    # Stage 3: vectorized pairwise-swap refinement.
    gpu_of = _refine(tokens_phy, gpu_of, num_gpus, epg)

    # Compact physical slots: slot = gpu * epg + rank-within-gpu.
    order = gpu_of.sort(-1, stable=True).indices         # [L, P]
    sorted_gpu = gpu_of.gather(1, order)
    within_rank = (torch.arange(num_replicas, device=device)
                   .unsqueeze(0).expand(num_layers, -1)) % epg
    slot_sorted = sorted_gpu * epg + within_rank         # [L, P]
    pphy2log = torch.empty_like(phy2log)
    pphy2log.scatter_(1, order, slot_sorted)
    # Remap to contiguous ids: slot_sorted is already a permutation of
    # 0..P-1 within each row, so no remap needed; derive final maps.
    final_phy2log = phy2log.gather(1, pphy2log) is None
    # Build final per-slot maps.
    final_phy2log = torch.empty(num_layers, num_replicas,
                                dtype=torch.int64, device=device)
    final_phyrank = torch.empty(num_layers, num_replicas,
                                 dtype=torch.int64, device=device)
    final_phy2log.scatter_(1, pphy2log, phy2log)
    final_phyrank.scatter_(1, pphy2log, phyrank)

    maxlogcnt = num_redundant + 1
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt), -1,
                         dtype=torch.int64, device=device)
    log2phy.view(num_layers, -1).scatter_(
        -1,
        final_phy2log * maxlogcnt + final_phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=device).expand(num_layers, -1),
    )
    return final_phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

