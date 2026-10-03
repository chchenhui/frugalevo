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

import torch


def _inverse_permutation(permutation: torch.Tensor) -> torch.Tensor:
    """Return the row-wise inverse of a row-wise permutation."""
    inverse = torch.empty_like(permutation)
    source = torch.arange(
        permutation.size(1),
        dtype=torch.int64,
        device=permutation.device,
    ).expand_as(permutation)
    inverse.scatter_(1, permutation, source)
    return inverse


def _ranks_from_pack_ids(pack_ids: torch.Tensor,
                         num_packs: int) -> torch.Tensor:
    """Return dense zero-based ranks within every assigned pack."""
    ranks = torch.empty_like(pack_ids)
    for pack in range(num_packs):
        members = pack_ids.eq(pack)
        member_ranks = members.to(torch.int64).cumsum(dim=1) - 1
        ranks.masked_scatter_(members, member_ranks.masked_select(members))
    return ranks


def _refine_extreme_packs(
    weight: torch.Tensor,
    pack_ids: torch.Tensor,
    num_packs: int,
    capacity: int,
    rounds: int = 2,
) -> torch.Tensor:
    """
    Improve LPT packing through cardinality-preserving heavy/light swaps.

    For a heavy pack H and light pack L, a swap transferring d changes their
    loads to H-d and L+d. The optimum is therefore d closest to (H-L)/2,
    rather than simply the largest permissible transfer.
    """
    num_layers, num_items = weight.shape
    if num_packs <= 1 or capacity <= 1 or num_items > 128:
        return pack_ids

    refined = pack_ids.clone()
    rows = torch.arange(num_layers, device=weight.device)
    loads = torch.zeros(
        (num_layers, num_packs),
        dtype=weight.dtype,
        device=weight.device,
    )
    loads.scatter_add_(1, refined, weight)

    for _ in range(rounds):
        heavy_pack = loads.argmax(dim=1)
        light_pack = loads.argmin(dim=1)
        gap = loads[rows, heavy_pack] - loads[rows, light_pack]

        heavy_members = refined.eq(heavy_pack.unsqueeze(1))
        light_members = refined.eq(light_pack.unsqueeze(1))

        # transfer[i, a, b] is load moved by exchanging heavy item a and
        # light item b. A valid exchange must reduce the old heavy load.
        transfer = weight.unsqueeze(2) - weight.unsqueeze(1)
        valid = (heavy_members.unsqueeze(2) & light_members.unsqueeze(1)
                 & transfer.gt(0) & transfer.lt(gap[:, None, None]))

        # Minimize the resulting maximum of the two extreme pack loads.
        error = (gap[:, None, None] - 2 * transfer).abs()
        error.masked_fill_(~valid, torch.inf)
        flat_error = error.flatten(1)
        best_error, flat_choice = flat_error.min(dim=1)
        accepted = torch.isfinite(best_error)

        heavy_item = flat_choice // num_items
        light_item = flat_choice % num_items
        selected_rows = rows[accepted]

        selected_heavy = heavy_pack[accepted]
        selected_light = light_pack[accepted]
        moved = transfer[
            selected_rows,
            heavy_item[accepted],
            light_item[accepted],
        ]

        # Empty advanced-index updates are valid, allowing fixed refinement
        # rounds without a device-to-host synchronization.
        refined[selected_rows, heavy_item[accepted]] = selected_light
        refined[selected_rows, light_item[accepted]] = selected_heavy
        loads[selected_rows, selected_heavy] -= moved
        loads[selected_rows, selected_light] += moved

    return refined


def balanced_packing(
    weight: torch.Tensor,
    num_packs: int,
    refine: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack equal-cardinality objects with vectorized cardinality-constrained LPT.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    capacity = num_items // num_packs

    if capacity == 1:
        pack_ids = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand(num_layers, -1)
        return pack_ids, torch.zeros_like(pack_ids)

    values = weight.float()
    sorted_items = values.argsort(dim=1, descending=True)
    pack_ids = torch.empty(
        (num_layers, num_items), dtype=torch.int64, device=weight.device
    )
    ranks = torch.empty_like(pack_ids)
    pack_load = torch.zeros(
        (num_layers, num_packs), dtype=values.dtype, device=weight.device
    )
    pack_count = torch.zeros(
        (num_layers, num_packs), dtype=torch.int64, device=weight.device
    )
    rows = torch.arange(num_layers, device=weight.device)
    unavailable = torch.finfo(values.dtype).max

    for position in range(num_items):
        item = sorted_items[:, position]
        target = pack_load.masked_fill(
            pack_count.ge(capacity), unavailable
        ).argmin(dim=1)
        pack_ids[rows, item] = target
        ranks[rows, item] = pack_count[rows, target]
        pack_load[rows, target] += values[rows, item]
        pack_count[rows, target] += 1

    if not refine:
        return pack_ids, ranks

    pack_ids = _refine_extreme_packs(values, pack_ids, num_packs, capacity)
    return pack_ids, _ranks_from_pack_ids(pack_ids, num_packs)


def replicate_experts(
    weight: torch.Tensor,
    num_physical: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate redundant expert slots by greedy per-replica-load water filling."""
    num_rows, num_logical = weight.shape
    assert num_physical >= num_logical

    device = weight.device
    physical_to_logical = torch.arange(
        num_physical, dtype=torch.int64, device=device
    ).expand(num_rows, -1).clone()
    replica_rank = torch.zeros_like(physical_to_logical)
    logical_count = torch.ones(
        (num_rows, num_logical), dtype=torch.int64, device=device
    )

    if num_physical == num_logical:
        return physical_to_logical, replica_rank, logical_count

    rows = torch.arange(num_rows, device=device)
    for slot in range(num_logical, num_physical):
        selected = (weight / logical_count).argmax(dim=1)
        physical_to_logical[:, slot] = selected
        replica_rank[:, slot] = logical_count[rows, selected]
        logical_count[rows, selected] += 1

    return physical_to_logical, replica_rank, logical_count


def _plan_hierarchical(
    weight: torch.Tensor,
    num_physical: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a node-aware and GPU-aware physical expert layout."""
    num_layers, num_logical = weight.shape
    assert num_logical % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical % num_gpus == 0

    group_size = num_logical // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    physical_per_gpu = num_physical // num_gpus
    logical_per_node = num_logical // num_nodes
    physical_per_node = num_physical // num_nodes

    group_load = weight.reshape(num_layers, num_groups, group_size).sum(dim=-1)
    group_node, group_rank = balanced_packing(group_load, num_nodes)

    group_offsets = torch.arange(
        group_size, dtype=torch.int64, device=weight.device
    )
    logical_to_node_order = (
        (group_node * groups_per_node + group_rank).unsqueeze(-1)
        * group_size + group_offsets
    ).flatten(1)
    node_order_to_logical = _inverse_permutation(logical_to_node_order)

    node_logical_load = weight.gather(1, node_order_to_logical).reshape(
        num_layers * num_nodes, logical_per_node
    )
    physical_to_node_logical, replica_rank, node_logical_count = (
        replicate_experts(node_logical_load, physical_per_node)
    )

    physical_load = (node_logical_load / node_logical_count).gather(
        1, physical_to_node_logical
    )
    gpu_id, gpu_rank = balanced_packing(
        physical_load, gpus_per_node, refine=True
    )
    physical_to_gpu_slot = gpu_id * physical_per_gpu + gpu_rank
    gpu_slot_to_physical = _inverse_permutation(physical_to_gpu_slot)

    ordered_node_logical = physical_to_node_logical.gather(
        1, gpu_slot_to_physical
    )
    ordered_rank = replica_rank.gather(1, gpu_slot_to_physical)

    node_offsets = torch.arange(
        0,
        num_logical,
        logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    ).view(1, num_nodes, 1)
    ordered_global_node_logical = (
        ordered_node_logical.reshape(num_layers, num_nodes, physical_per_node)
        + node_offsets
    ).flatten(1)

    physical_to_logical = node_order_to_logical.gather(
        1, ordered_global_node_logical
    )
    physical_rank = ordered_rank.reshape(num_layers, num_physical)
    logical_count = node_logical_count.reshape(
        num_layers, num_logical
    ).gather(1, logical_to_node_order)

    return physical_to_logical, physical_rank, logical_count


def _build_logical_to_physical(
    physical_to_logical: torch.Tensor,
    physical_rank: torch.Tensor,
    logical_count: torch.Tensor,
) -> torch.Tensor:
    """Materialize the padded inverse EPLB mapping."""
    num_layers, num_physical = physical_to_logical.shape
    num_logical = logical_count.size(1)
    max_count = num_physical - num_logical + 1

    logical_to_physical = torch.full(
        (num_layers, num_logical, max_count),
        -1,
        dtype=torch.int64,
        device=physical_to_logical.device,
    )
    physical_ids = torch.arange(
        num_physical,
        dtype=torch.int64,
        device=physical_to_logical.device,
    ).expand(num_layers, -1)
    logical_to_physical.reshape(num_layers, -1).scatter_(
        1,
        physical_to_logical * max_count + physical_rank,
        physical_ids,
    )
    return logical_to_physical


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rebalance logical experts into the requested physical topology."""
    _, num_logical = weight.shape
    assert num_replicas >= num_logical

    # Preserve device locality: float() is a no-op for float32 input and all
    # planner intermediates and returned mappings remain on weight.device.
    planner_weight = weight.float()

    if num_groups % num_nodes == 0:
        physical_to_logical, physical_rank, logical_count = (
            _plan_hierarchical(
                planner_weight,
                num_replicas,
                num_groups,
                num_nodes,
                num_gpus,
            )
        )
    else:
        physical_to_logical, physical_rank, logical_count = (
            _plan_hierarchical(
                planner_weight,
                num_replicas,
                1,
                1,
                num_gpus,
            )
        )

    logical_to_physical = _build_logical_to_physical(
        physical_to_logical,
        physical_rank,
        logical_count,
    )
    return physical_to_logical, logical_to_physical, logical_count


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
