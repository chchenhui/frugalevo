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
    Pack equally-sized collections of weighted items into packs.

    The sorted items are processed in capacity-sized blocks. Every block
    contributes exactly one item to every pack, while the heaviest item in a
    block is assigned to the currently lightest pack. This folded assignment
    avoids the per-item Python scheduling loop while retaining the important
    load-aware behavior and exact cardinality constraint.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    if items_per_pack == 1:
        pack_index = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand_as(weight)
        return pack_index, torch.zeros_like(pack_index)

    # Each block has one descending-load item for each pack.
    sorted_items = torch.argsort(weight.float(), dim=-1, descending=True)
    sorted_weight = weight.gather(-1, sorted_items).reshape(
        num_layers, items_per_pack, num_packs
    )

    sorted_pack = torch.empty(
        (num_layers, items_per_pack, num_packs),
        dtype=torch.int64,
        device=weight.device,
    )
    pack_load = torch.zeros(
        (num_layers, num_packs), dtype=weight.dtype, device=weight.device
    )

    # The loop is over capacity blocks, normally much smaller than the number
    # of experts. All layers and all packs are processed together.
    for block in range(items_per_pack):
        lightest_packs = torch.argsort(pack_load, dim=-1)
        sorted_pack[:, block] = lightest_packs
        pack_load.scatter_add_(1, lightest_packs, sorted_weight[:, block])

    item_to_pack = torch.empty(
        (num_layers, num_items), dtype=torch.int64, device=weight.device
    )
    item_to_rank = torch.empty_like(item_to_pack)

    item_to_pack.scatter_(
        1,
        sorted_items,
        sorted_pack.reshape(num_layers, num_items),
    )
    ranks = torch.arange(
        items_per_pack, dtype=torch.int64, device=weight.device
    ).view(1, items_per_pack, 1).expand(num_layers, -1, num_packs)
    item_to_rank.scatter_(1, sorted_items, ranks.reshape(num_layers, num_items))
    return item_to_pack, item_to_rank


def replicate_experts(
    weight: torch.Tensor,
    num_phy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate redundant copies using discrete marginal water filling.

    Adding the next replica of logical expert e has priority weight[e] / c[e].
    Selecting the largest priorities is equivalent to repeatedly splitting the
    currently most overloaded replica, but the batched top-k implementation
    removes most sequential scheduling overhead.
    """
    num_layers, num_logical = weight.shape
    num_extra = num_phy - num_logical
    assert num_extra >= 0

    device = weight.device
    base = torch.arange(num_logical, dtype=torch.int64, device=device)
    phy2log = torch.empty(
        (num_layers, num_phy), dtype=torch.int64, device=device
    )
    phy2log[:, :num_logical] = base
    rank = torch.zeros((num_layers, num_phy), dtype=torch.int64, device=device)
    logcnt = torch.ones(
        (num_layers, num_logical), dtype=torch.int64, device=device
    )

    if num_extra == 0:
        return phy2log, rank, logcnt

    # A dense marginal table is much faster for normal EPLB shapes. Keep an
    # iterative path for extreme shapes to bound temporary memory.
    candidate_count = num_layers * num_logical * num_extra
    if candidate_count <= 8_000_000:
        divisors = torch.arange(
            1, num_extra + 1, dtype=weight.dtype, device=device
        )
        marginal = (weight.unsqueeze(-1) / divisors).flatten(1)
        selected = torch.topk(marginal, num_extra, dim=-1).indices
        selected_expert = selected.remainder(num_logical)

        phy2log[:, num_logical:] = selected_expert
        logcnt.scatter_add_(
            1,
            selected_expert,
            torch.ones_like(selected_expert, dtype=torch.int64),
        )

        # Assign unique replica ranks for each selected logical expert without
        # a quadratic comparison of duplicate selections.
        order = torch.argsort(selected_expert, dim=-1)
        ordered_expert = selected_expert.gather(1, order)
        positions = torch.arange(
            num_extra, dtype=torch.int64, device=device
        ).view(1, -1).expand(num_layers, -1)
        starts = torch.where(
            torch.ones_like(ordered_expert, dtype=torch.bool),
            positions,
            torch.zeros_like(positions),
        )
        starts[:, 1:] = torch.where(
            ordered_expert[:, 1:] != ordered_expert[:, :-1],
            positions[:, 1:],
            torch.zeros_like(positions[:, 1:]),
        )
        group_start = torch.cummax(starts, dim=-1).values
        selected_rank = positions - group_start + 1
        rank[:, num_logical:].scatter_(1, order, selected_rank)
    else:
        # Memory-safe fallback preserving the same discrete water-filling rule.
        layer_index = torch.arange(num_layers, dtype=torch.int64, device=device)
        for physical in range(num_logical, num_phy):
            expert = (weight / logcnt).argmax(dim=-1)
            phy2log[:, physical] = expert
            rank[:, physical] = logcnt[layer_index, expert]
            logcnt[layer_index, expert] += 1

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a node-local replica layout and balance it across local GPUs."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_physical_experts // num_gpus

    def inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
        inverse = torch.empty_like(perm)
        source = torch.arange(
            perm.size(1), dtype=torch.int64, device=perm.device
        ).expand_as(perm)
        inverse.scatter_(1, perm, source)
        return inverse

    # 1. Fold complete expert groups onto nodes according to aggregate load.
    group_load = weight.reshape(num_layers, num_groups, group_size).sum(dim=-1)
    group_node, group_rank = balanced_packing(group_load, num_nodes)

    logical_to_node_order = (
        (group_node * groups_per_node + group_rank).unsqueeze(-1) * group_size
        + torch.arange(group_size, dtype=torch.int64, device=weight.device)
    ).flatten(1)
    node_order_to_logical = inverse_permutation(logical_to_node_order)

    logical_per_node = num_logical_experts // num_nodes
    node_weight = weight.gather(1, node_order_to_logical).reshape(
        num_layers * num_nodes, logical_per_node
    )

    # 2. Allocate replicas independently inside each node.
    phy_per_node = num_physical_experts // num_nodes
    phy_to_node_logical, phy_rank, node_count = replicate_experts(
        node_weight, phy_per_node
    )

    # 3. Fold node-local physical experts onto node-local GPU slots.
    physical_load = (node_weight / node_count).gather(
        1, phy_to_node_logical
    )
    gpu_index, gpu_rank = balanced_packing(physical_load, gpus_per_node)
    phy_to_gpu_slot = gpu_index * phy_per_gpu + gpu_rank
    gpu_slot_to_phy = inverse_permutation(phy_to_gpu_slot)

    gpu_slot_to_node_logical = phy_to_node_logical.gather(
        1, gpu_slot_to_phy
    )
    node_offsets = torch.arange(
        0,
        num_logical_experts,
        logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    ).view(1, num_nodes, 1)

    gpu_slot_to_node_order = (
        gpu_slot_to_node_logical.reshape(num_layers, num_nodes, -1)
        + node_offsets
    ).flatten(1)

    physical_to_logical = node_order_to_logical.gather(
        1, gpu_slot_to_node_order
    )
    physical_rank = phy_rank.gather(1, gpu_slot_to_phy).reshape(
        num_layers, num_physical_experts
    )
    logical_count = node_count.reshape(num_layers, -1).gather(
        1, logical_to_node_order
    )
    return physical_to_logical, physical_rank, logical_count


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange and replicate experts according to observed routing load.

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        )
    else:
        # Fall back to global balancing when group boundaries cannot be split
        # evenly among nodes.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight,
            num_replicas,
            1,
            1,
            num_gpus,
        )

    max_replica_count = num_replicas - num_logical_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replica_count),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    physical_index = torch.arange(
        num_replicas, dtype=torch.int64, device=logcnt.device
    ).expand(num_layers, -1)
    log2phy.reshape(num_layers, -1).scatter_(
        1,
        phy2log * max_replica_count + phyrank,
        physical_index,
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

