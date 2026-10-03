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

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class _Topology:
    num_layers: int
    num_logical_experts: int
    num_physical_experts: int
    num_groups: int
    num_nodes: int
    num_gpus: int

    @property
    def group_size(self) -> int:
        return self.num_logical_experts // self.num_groups

    @property
    def groups_per_node(self) -> int:
        return self.num_groups // self.num_nodes

    @property
    def logical_per_node(self) -> int:
        return self.num_logical_experts // self.num_nodes

    @property
    def physical_per_node(self) -> int:
        return self.num_physical_experts // self.num_nodes

    @property
    def gpus_per_node(self) -> int:
        return self.num_gpus // self.num_nodes

    @property
    def physical_per_gpu(self) -> int:
        return self.num_physical_experts // self.num_gpus


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return the row-wise inverse of a dense row-wise permutation."""
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
    Assign equally sized item sets to packs using LPT scheduling.

    Every pack receives exactly ``weight.size(1) // num_packs`` items. Items
    are considered in descending load order and assigned to the currently
    least-loaded non-full pack. A heap makes the least-loaded-pack lookup
    logarithmic instead of linearly scanning every pack for every item.
    """
    num_rows, num_items = weight.shape
    assert num_packs > 0
    assert num_items % num_packs == 0

    items_per_pack = num_items // num_packs
    if items_per_pack == 1:
        pack_index = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand(num_rows, -1)
        return pack_index, torch.zeros_like(pack_index)

    # Sorting remains tensor-native. Only the scheduling state is moved into
    # compact Python scalar structures, avoiding repeated tensor scalar ops.
    sorted_indices = torch.argsort(weight, dim=-1, descending=True)
    sorted_rows = sorted_indices.tolist()
    weight_rows = weight.tolist()

    pack_rows: list[list[int]] = []
    rank_rows: list[list[int]] = []

    for row_indices, row_weights in zip(sorted_rows, weight_rows):
        assignments = [-1] * num_items
        ranks = [-1] * num_items
        item_counts = [0] * num_packs

        # (accumulated_load, pack_id). pack_id provides deterministic ties.
        available: list[tuple[float, int]] = [
            (0.0, pack) for pack in range(num_packs)
        ]
        heapq.heapify(available)

        for item in row_indices:
            current_load, pack = heapq.heappop(available)
            rank = item_counts[pack]
            assignments[item] = pack
            ranks[item] = rank
            item_counts[pack] = rank + 1

            if item_counts[pack] < items_per_pack:
                heapq.heappush(
                    available, (current_load + float(row_weights[item]), pack)
                )

        pack_rows.append(assignments)
        rank_rows.append(ranks)

    return (
        torch.tensor(pack_rows, dtype=torch.int64, device=weight.device),
        torch.tensor(rank_rows, dtype=torch.int64, device=weight.device),
    )


def replicate_experts(
    weight: torch.Tensor,
    num_phy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate redundant physical experts with greedy water-filling.

    The next replica always goes to the logical expert with maximum current
    per-replica load. Priority queues avoid repeatedly evaluating
    ``weight / replica_count`` for every logical expert.
    """
    num_rows, num_logical = weight.shape
    assert num_phy >= num_logical

    redundant_count = num_phy - num_logical
    if redundant_count == 0:
        logical_ids = torch.arange(
            num_logical, dtype=torch.int64, device=weight.device
        ).expand(num_rows, -1)
        ranks = torch.zeros_like(logical_ids)
        counts = torch.ones_like(logical_ids)
        return logical_ids, ranks, counts

    physical_rows: list[list[int]] = []
    rank_rows: list[list[int]] = []
    count_rows: list[list[int]] = []

    for loads in weight.tolist():
        counts = [1] * num_logical
        physical = list(range(num_logical))
        ranks = [0] * num_logical

        # Python's heap is a min-heap, hence negative effective load.
        # Expert id is included for deterministic tie breaking.
        candidates = [(-float(load), expert) for expert, load in enumerate(loads)]
        heapq.heapify(candidates)

        for _ in range(redundant_count):
            _, expert = heapq.heappop(candidates)
            physical.append(expert)
            ranks.append(counts[expert])
            counts[expert] += 1
            heapq.heappush(
                candidates, (-float(loads[expert]) / counts[expert], expert)
            )

        physical_rows.append(physical)
        rank_rows.append(ranks)
        count_rows.append(counts)

    return (
        torch.tensor(physical_rows, dtype=torch.int64, device=weight.device),
        torch.tensor(rank_rows, dtype=torch.int64, device=weight.device),
        torch.tensor(count_rows, dtype=torch.int64, device=weight.device),
    )


def _validate_topology(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> _Topology:
    num_layers, num_logical_experts = weight.shape
    assert num_physical_experts >= num_logical_experts
    assert num_groups > 0 and num_nodes > 0 and num_gpus > 0
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    return _Topology(
        num_layers=num_layers,
        num_logical_experts=num_logical_experts,
        num_physical_experts=num_physical_experts,
        num_groups=num_groups,
        num_nodes=num_nodes,
        num_gpus=num_gpus,
    )


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a node-aware physical placement plan.

    Returns:
        physical_to_logical_map: [layers, num_physical_experts]
        physical_replica_rank: [layers, num_physical_experts]
        logical_replica_count: [layers, num_logical_experts]
    """
    topology = _validate_topology(
        weight, num_physical_experts, num_groups, num_nodes, num_gpus
    )

    # Stage 1: group logical experts by node using aggregate group load.
    group_load = weight.unflatten(
        -1, (topology.num_groups, topology.group_size)
    ).sum(dim=-1)
    group_node, group_rank = balanced_packing(group_load, topology.num_nodes)

    logical_to_node_order = (
        (
            group_node * topology.groups_per_node + group_rank
        ).unsqueeze(-1)
        * topology.group_size
        + torch.arange(
            topology.group_size, dtype=torch.int64, device=weight.device
        )
    ).flatten(-2)
    node_order_to_logical = _inverse_permutation(logical_to_node_order)

    # Stage 2: allocate replicas independently within each node.
    node_logical_load = weight.gather(-1, node_order_to_logical).reshape(
        topology.num_layers * topology.num_nodes,
        topology.logical_per_node,
    )
    physical_to_node_logical, replica_rank, node_logical_count = (
        replicate_experts(node_logical_load, topology.physical_per_node)
    )

    # Stage 3: place equalized physical replica loads on GPUs within a node.
    physical_load = (node_logical_load / node_logical_count).gather(
        -1, physical_to_node_logical
    )
    gpu_index, gpu_rank = balanced_packing(
        physical_load, topology.gpus_per_node
    )
    physical_to_gpu_slot = (
        gpu_index * topology.physical_per_gpu + gpu_rank
    )
    gpu_slot_to_physical = _inverse_permutation(physical_to_gpu_slot)

    # Stage 4: translate local/node-order ids back to original logical ids.
    gpu_slot_to_node_logical = physical_to_node_logical.gather(
        -1, gpu_slot_to_physical
    )
    node_offsets = torch.arange(
        0,
        topology.num_logical_experts,
        topology.logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    ).view(1, topology.num_nodes, 1)

    gpu_slot_to_node_order = (
        gpu_slot_to_node_logical.view(
            topology.num_layers, topology.num_nodes, -1
        )
        + node_offsets
    ).flatten(1, 2)

    physical_to_logical = node_order_to_logical.gather(
        -1, gpu_slot_to_node_order
    )
    physical_replica_rank = replica_rank.gather(
        -1, gpu_slot_to_physical
    ).view(topology.num_layers, -1)
    logical_replica_count = node_logical_count.view(
        topology.num_layers, -1
    ).gather(-1, logical_to_node_order)

    return (
        physical_to_logical,
        physical_replica_rank,
        logical_replica_count,
    )


def _materialize_logical_to_physical(
    physical_to_logical: torch.Tensor,
    physical_replica_rank: torch.Tensor,
    logical_replica_count: torch.Tensor,
) -> torch.Tensor:
    """Build the padded inverse expert map expected by the EPLB caller."""
    num_layers, num_physical = physical_to_logical.shape
    num_logical = logical_replica_count.size(1)
    max_replica_count = num_physical - num_logical + 1

    logical_to_physical = torch.full(
        (num_layers, num_logical, max_replica_count),
        -1,
        dtype=torch.int64,
        device=physical_to_logical.device,
    )
    destination = physical_to_logical * max_replica_count + physical_replica_rank
    source = torch.arange(
        num_physical, dtype=torch.int64, device=physical_to_logical.device
    ).expand(num_layers, -1)

    logical_to_physical.view(num_layers, -1).scatter_(
        1, destination, source
    )
    return logical_to_physical


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancing.

    Args:
        weight: [layers, num_logical_experts] expert load statistics.
        num_replicas: Total physical expert slots.
        num_groups: Number of logical expert groups.
        num_nodes: Number of server nodes.
        num_gpus: Total number of GPUs.

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    # Planning is intentionally CPU resident: it is infrequent control-plane
    # work, avoids GPU synchronization, and returns compact integer maps.
    planning_weight = weight.to(device="cpu", dtype=torch.float32)
    _, num_logical_experts = planning_weight.shape
    assert num_replicas >= num_logical_experts

    if num_groups % num_nodes == 0:
        hierarchy = (num_groups, num_nodes)
    else:
        # Preserve the original global-policy fallback when group boundaries
        # cannot be evenly distributed across nodes.
        hierarchy = (1, 1)

    physical_to_logical, physical_replica_rank, logical_replica_count = (
        rebalance_experts_hierarchical(
            planning_weight,
            num_replicas,
            hierarchy[0],
            hierarchy[1],
            num_gpus,
        )
    )
    logical_to_physical = _materialize_logical_to_physical(
        physical_to_logical,
        physical_replica_rank,
        logical_replica_count,
    )

    return physical_to_logical, logical_to_physical, logical_replica_count


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

