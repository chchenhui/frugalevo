# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.
"""

# EVOLVE-BLOCK-START

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorize exact LPT packing across layers with batched state updates."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            num_groups, dtype=torch.int64,
            device=weight.device).expand(weight.shape)
        return pack_index, torch.zeros_like(weight, dtype=torch.int64)

    # Sorting and all mutable packing state stay on CPU.  Float64 accumulation
    # preserves stable greedy comparisons while avoiding per-layer Python loops.
    sorted_weight, indices = weight.float().cpu().sort(-1, descending=True)
    sorted_weight = sorted_weight.to(torch.float64)

    pack_index = torch.empty(
        (num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty_like(pack_index)
    pack_load = torch.zeros(
        (num_layers, num_packs), dtype=torch.float64, device="cpu")
    pack_count = torch.zeros(
        (num_layers, num_packs), dtype=torch.int64, device="cpu")
    inf = float("inf")

    # One transition updates every layer.  Clone followed by in-place masking
    # avoids constructing the masked result through an additional dispatch.
    for position in range(num_groups):
        available_load = pack_load.clone()
        available_load.masked_fill_(
            pack_count >= groups_per_pack, inf)
        chosen_pack = available_load.argmin(dim=1, keepdim=True)
        group = indices[:, position].unsqueeze(1)

        chosen_rank = pack_count.gather(1, chosen_pack)
        pack_index.scatter_(1, group, chosen_pack)
        rank_in_pack.scatter_(1, group, chosen_rank)
        pack_count.scatter_(1, chosen_pack, chosen_rank + 1)
        pack_load.scatter_add_(
            1, chosen_pack, sorted_weight[:, position].unsqueeze(1))

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(
        num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


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
            torch.arange(
                perm.size(1), dtype=torch.int64,
                device=perm.device).expand(perm.shape))
        return inv

    tokens_per_group = weight.unflatten(
        -1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (
        ((group_pack_index * groups_per_node + group_rank_in_pack) *
         group_size).unsqueeze(-1) +
        torch.arange(
            group_size, dtype=torch.int64,
            device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(
                     0, num_logical_experts,
                     num_logical_experts // num_nodes,
                     device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
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
    num_layers, num_logical_experts = weight.shape
    weight = weight.detach().float().cpu().contiguous()

    def placement_proxy(candidate):
        candidate_phy2log, _, candidate_logcnt = candidate
        physical_load = weight.gather(1, candidate_phy2log)
        replica_count = candidate_logcnt.gather(1, candidate_phy2log)
        physical_load = physical_load / replica_count.clamp_min(1)
        gpu_load = physical_load.reshape(
            num_layers, num_gpus, num_replicas // num_gpus).sum(-1)
        return (gpu_load.amax(-1), gpu_load.square().sum(-1),
                physical_load.amax(-1))

    if num_groups % num_nodes == 0:
        hierarchical = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)

        # With one node these constructions are identical; avoid duplicating
        # the complete replica and packing calculation.
        if num_nodes == 1:
            phy2log, phyrank, logcnt = hierarchical
        else:
            global_pool = rebalance_experts_hierarchical(
                weight, num_replicas, 1, 1, num_gpus)
            h_peak, h_sq, h_expert = placement_proxy(hierarchical)
            g_peak, g_sq, g_expert = placement_proxy(global_pool)

            use_global = g_peak < h_peak
            use_global |= (g_peak == h_peak) & (g_sq < h_sq)
            use_global |= ((g_peak == h_peak) & (g_sq == h_sq) &
                           (g_expert < h_expert))

            phy2log = torch.where(
                use_global.unsqueeze(1), global_pool[0], hierarchical[0])
            phyrank = torch.where(
                use_global.unsqueeze(1), global_pool[1], hierarchical[1])
            logcnt = torch.where(
                use_global.unsqueeze(1), global_pool[2], hierarchical[2])
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    maxlogcnt = num_replicas - num_logical_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt), -1,
        dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1, phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_replicas, dtype=torch.int64,
            device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]