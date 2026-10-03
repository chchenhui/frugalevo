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

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack descending-weight items with LPT using list-backed heap assignment.

    Each layer is independently processed by a min-heap of non-full packs.
    Python lists avoid repeated scalar tensor writes in the hot greedy loop;
    tensors are materialized only once after all placements are complete.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            num_groups, dtype=torch.int64,
            device=weight.device).expand(weight.shape)
        return pack_index, torch.zeros_like(pack_index)

    sorted_weight, indices = weight.float().sort(-1, descending=True)
    sorted_weight = sorted_weight.tolist()
    indices = indices.tolist()
    pack_index = [[-1] * num_groups for _ in range(num_layers)]
    rank_in_pack = [[-1] * num_groups for _ in range(num_layers)]

    for layer, (layer_indices, layer_weights) in enumerate(
            zip(indices, sorted_weight)):
        heap = [(0.0, pack) for pack in range(num_packs)]
        heapq.heapify(heap)
        counts = [0] * num_packs
        packs = pack_index[layer]
        ranks = rank_in_pack[layer]
        for group, item_weight in zip(layer_indices, layer_weights):
            load, pack = heapq.heappop(heap)
            packs[group] = pack
            ranks[group] = counts[pack]
            counts[pack] += 1
            if counts[pack] < groups_per_pack:
                heapq.heappush(heap, (load + item_weight, pack))

    return (torch.tensor(pack_index, dtype=torch.int64),
            torch.tensor(rank_in_pack, dtype=torch.int64))


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily add replicas by updating only each selected expert's load.

    `replica_load[e]` is maintained as ``weight[e] / replica_count[e]``.
    Therefore each iteration has the same choice as recomputing the complete
    division, but avoids a full matrix division for every redundant replica.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.empty(n, num_phy, dtype=torch.int64, device=device)
    phy2log[:, :num_log] = torch.arange(
        num_log, dtype=torch.int64, device=device)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    rows = torch.arange(n, dtype=torch.int64, device=device)
    replica_load = weight.clone()

    for i in range(num_log, num_phy):
        expert = replica_load.max(dim=-1).indices
        old_count = logcnt[rows, expert]
        phy2log[:, i] = expert
        rank[:, i] = old_count
        new_count = old_count + 1
        logcnt[rows, expert] = new_count
        replica_load[rows, expert] = weight[rows, expert] / new_count

    return phy2log, rank, logcnt


def _placement_cost(weight: torch.Tensor, phy2log: torch.Tensor,
                    logcnt: torch.Tensor, num_nodes: int,
                    num_gpus: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute realized per-layer node and GPU bottleneck loads."""
    layers, replicas = phy2log.shape
    slots_per_gpu = replicas // num_gpus
    copy_load = weight.gather(1, phy2log) / logcnt.gather(1, phy2log)
    gpu_load = copy_load.view(layers, num_gpus, slots_per_gpu).sum(-1)
    node_load = gpu_load.view(layers, num_nodes,
                              num_gpus // num_nodes).sum(-1)
    return node_load.max(-1).values, gpu_load.max(-1).values


def _striped_group_permutation(weight: torch.Tensor,
                               num_groups: int) -> torch.Tensor | None:
    """Spread high-average-demand experts across fixed-size hierarchy groups."""
    _, num_experts = weight.shape
    if num_groups <= 1 or num_experts % num_groups:
        return None

    # Normalize each layer so a high-token layer cannot dominate the grouping
    # decision, then stripe descending-demand experts across all groups.
    normalized = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
    order = normalized.mean(0).argsort(descending=True)
    group_size = num_experts // num_groups
    return order.view(group_size, num_groups).transpose(0, 1).reshape(-1)


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
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

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
    ).view(1, -1, 1)).flatten(-2)
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
    if num_groups % num_nodes == 0:
        # Keep the original fixed groups as the incumbent. A striped ordering
        # is a low-cost second candidate that prevents persistent hot experts
        # from being concentrated in the same node-local group.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)

        permutation = _striped_group_permutation(weight, num_groups)
        if permutation is not None and num_nodes > 1:
            expanded_permutation = permutation.expand(num_layers, -1)
            candidate_weight = weight.gather(1, expanded_permutation)
            cand_phy, cand_rank, cand_cnt = rebalance_experts_hierarchical(
                candidate_weight, num_replicas, num_groups, num_nodes,
                num_gpus)

            # Translate candidate logical IDs and replica counts back to the
            # caller's original expert numbering before measuring its result.
            cand_phy = permutation[cand_phy]
            original_cnt = torch.empty_like(cand_cnt)
            original_cnt.scatter_(1, expanded_permutation, cand_cnt)

            base_node, base_gpu = _placement_cost(
                weight, phy2log, logcnt, num_nodes, num_gpus)
            cand_node, cand_gpu = _placement_cost(
                weight, cand_phy, original_cnt, num_nodes, num_gpus)

            # Node balance is the hierarchy's primary objective; GPU balance
            # breaks node-load ties. Thus the candidate cannot worsen either.
            use_candidate = ((cand_node < base_node) |
                             ((cand_node == base_node) &
                              (cand_gpu < base_gpu)))
            if use_candidate.any():
                selected = use_candidate.nonzero(as_tuple=False).flatten()
                phy2log = phy2log.clone()
                phyrank = phyrank.clone()
                logcnt = logcnt.clone()
                phy2log[selected] = cand_phy[selected]
                phyrank[selected] = cand_rank[selected]
                logcnt[selected] = original_cnt[selected]
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
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

