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


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return the inverse of every permutation in a batch."""
    inverse = torch.empty_like(perm)
    positions = torch.arange(
        perm.size(1), dtype=torch.int64, device=perm.device
    ).expand_as(perm)
    inverse.scatter_(1, perm, positions)
    return inverse


def balanced_packing(
    weight: torch.Tensor,
    num_packs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Assign equal-cardinality item sets to packs with LPT scheduling.

    Items are processed from greatest to smallest weight. Each item is assigned
    to the least-loaded pack which still has a free slot. The capacity-aware
    heap guarantees exactly equal cardinality without a later repair pass.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    if items_per_pack == 1:
        pack_index = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand(num_layers, -1)
        return pack_index, torch.zeros_like(pack_index)

    cpu_weight = weight.float().cpu()
    try:
        sorted_items = cpu_weight.sort(
            dim=-1, descending=True, stable=True
        ).indices
    except TypeError:
        sorted_items = cpu_weight.sort(dim=-1, descending=True).indices

    # Heap scheduling is sequential per layer, but using Python lists for the
    # bookkeeping avoids one PyTorch scalar write per physical assignment.
    pack_rows: list[list[int]] = []
    rank_rows: list[list[int]] = []
    for row_weight, ordered_items in zip(
        cpu_weight.tolist(), sorted_items.tolist()
    ):
        heap = [(0.0, pack) for pack in range(num_packs)]
        heapq.heapify(heap)
        item_count = [0] * num_packs
        pack_row = [0] * num_items
        rank_row = [0] * num_items

        for item in ordered_items:
            current_load, pack = heapq.heappop(heap)
            rank = item_count[pack]
            pack_row[item] = pack
            rank_row[item] = rank
            rank += 1
            item_count[pack] = rank

            if rank < items_per_pack:
                heapq.heappush(
                    heap, (current_load + row_weight[item], pack)
                )

        pack_rows.append(pack_row)
        rank_rows.append(rank_row)

    return (
        torch.tensor(pack_rows, dtype=torch.int64, device=weight.device),
        torch.tensor(rank_rows, dtype=torch.int64, device=weight.device),
    )


def replicate_experts(
    weight: torch.Tensor,
    num_phy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate replicas by selecting the largest discrete waterline reductions.

    The candidate ``weight[e] / k`` denotes the load after creating the
    ``k``-th total copy of expert ``e``. Selecting the largest redundant-copy
    candidates minimizes the maximum replica load independently per layer.
    """
    num_layers, num_logical = weight.shape
    num_redundant = num_phy - num_logical
    assert num_redundant >= 0

    device = weight.device
    base_experts = torch.arange(
        num_logical, dtype=torch.int64, device=device
    ).expand(num_layers, -1)

    if num_redundant == 0:
        return (
            base_experts,
            torch.zeros_like(base_experts),
            torch.ones_like(base_experts),
        )

    # Candidate position r represents the extra replica whose resulting total
    # count is r + 2. A deterministic negative fallback keeps zero-load
    # selections prefix ordered as well.
    replica_number = torch.arange(
        2, num_redundant + 2, dtype=weight.dtype, device=device
    )
    candidate = weight.unsqueeze(-1) / replica_number
    candidate = torch.where(
        weight.unsqueeze(-1) > 0,
        candidate,
        -replica_number.view(1, 1, -1),
    )

    _, selected = candidate.flatten(1).topk(
        num_redundant, dim=1, largest=True, sorted=False
    )
    extra_logical = selected // num_redundant
    extra_rank = selected.remainder(num_redundant) + 1

    logical_count = torch.ones(
        (num_layers, num_logical), dtype=torch.int64, device=device
    )
    logical_count.scatter_add_(
        1, extra_logical, torch.ones_like(extra_logical)
    )

    return (
        torch.cat((base_experts, extra_logical), dim=1),
        torch.cat((torch.zeros_like(base_experts), extra_rank), dim=1),
        logical_count,
    )


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform group-to-node, replica, then physical-slot balancing."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    logical_per_node = num_logical_experts // num_nodes
    physical_per_node = num_physical_experts // num_nodes
    physical_per_gpu = num_physical_experts // num_gpus

    # Step 1: assign complete routing groups to nodes.
    group_load = weight.reshape(
        num_layers, num_groups, group_size
    ).sum(dim=-1)
    group_pack, group_rank = balanced_packing(group_load, num_nodes)

    local_to_global = (
        (group_pack * groups_per_node + group_rank) * group_size
    ).unsqueeze(-1) + torch.arange(
        group_size, dtype=torch.int64, device=weight.device
    )
    local_to_global = local_to_global.flatten(1)
    global_to_local = _inverse_permutation(local_to_global)

    # Step 2: allocate redundant experts independently within every node.
    local_weight = weight.gather(1, global_to_local).reshape(
        num_layers * num_nodes, logical_per_node
    )
    phy_to_local, phy_rank, local_count = replicate_experts(
        local_weight, physical_per_node
    )

    # Step 3: place equally-loaded physical experts onto local GPU slots.
    replica_load = (
        local_weight / local_count.to(local_weight.dtype)
    ).gather(1, phy_to_local)
    gpu_pack, gpu_rank = balanced_packing(replica_load, gpus_per_node)
    physical_to_slot = gpu_pack * physical_per_gpu + gpu_rank
    slot_to_physical = _inverse_permutation(physical_to_slot)

    ordered_local = phy_to_local.gather(1, slot_to_physical)
    ordered_rank = phy_rank.gather(1, slot_to_physical)

    node_offsets = torch.arange(
        0,
        num_logical_experts,
        logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    ).view(1, num_nodes, 1)
    ordered_global_local = (
        ordered_local.reshape(num_layers, num_nodes, physical_per_node)
        + node_offsets
    ).flatten(1)

    physical_to_logical = global_to_local.gather(1, ordered_global_local)
    physical_rank = ordered_rank.reshape(num_layers, num_physical_experts)
    logical_count = local_count.reshape(
        num_layers, num_logical_experts
    ).gather(1, local_to_global)

    return physical_to_logical, physical_rank, logical_count


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange and replicate experts.

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

    max_replica_count = num_replicas - num_logical_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replica_count),
        -1,
        dtype=torch.int64,
        device=weight.device,
    )
    physical_ids = torch.arange(
        num_replicas, dtype=torch.int64, device=weight.device
    ).expand(num_layers, -1)
    log2phy.flatten(1).scatter_(
        1,
        phy2log * max_replica_count + phyrank,
        physical_ids,
    )

    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
