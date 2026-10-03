# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.
"""

# EVOLVE-BLOCK-START

import heapq

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack weighted objects into equal-cardinality packs using deterministic LPT.
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

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for layer in range(num_layers):
        available = [(0, pack) for pack in range(num_packs)]
        heapq.heapify(available)
        pack_items = [0] * num_packs
        row_weight = weight[layer].tolist()

        for group in indices[layer].tolist():
            current_weight, pack = heapq.heappop(available)
            rank = pack_items[pack]
            pack_index[layer, group] = pack
            rank_in_pack[layer, group] = rank
            pack_items[pack] = rank + 1
            if rank + 1 < groups_per_pack:
                heapq.heappush(available,
                               (current_weight + row_weight[group], pack))
    return pack_index, rank_in_pack


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

    # Step 1: pack logical groups to nodes.
    tokens_per_group = weight.unflatten(-1,
                                        (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=weight.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate independently within each node.
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical replicas to GPUs within each node.
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

    # A final physical position maps backward through pphy2phy.  Replica rank
    # must follow that same inverse mapping to remain paired with its expert.
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
    Compare node-local and global layouts by each layer's maximum GPU load.

    When node-local packing is feasible, construct both the locality-preserving
    and unrestricted layouts and select the one with the lower per-layer GPU
    peak, retaining locality on exact ties.
    """
    num_layers, num_logical_experts = weight.shape
    # Load metrics are inference data; avoid retaining an autograd graph while
    # transferring them to the CPU for the deterministic packing routines.
    weight = weight.detach().float().cpu()

    if num_groups % num_nodes == 0:
        hierarchical_layout = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
        global_layout = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

        def peak_load(layout):
            """Return each layer's maximum predicted GPU load."""
            candidate_phy2log, _, candidate_logcnt = layout
            physical_load = weight.gather(
                1, candidate_phy2log
            ) / candidate_logcnt.gather(1, candidate_phy2log)
            # reshape is valid because each candidate has a fixed number of
            # contiguous physical slots per GPU.
            return physical_load.reshape(num_layers, num_gpus, -1).sum(
                dim=-1).amax(dim=-1)

        hierarchical_peak = peak_load(hierarchical_layout)
        global_peak = peak_load(global_layout)

        # Strict comparison preserves node locality on exact ties.
        use_global = global_peak < hierarchical_peak
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