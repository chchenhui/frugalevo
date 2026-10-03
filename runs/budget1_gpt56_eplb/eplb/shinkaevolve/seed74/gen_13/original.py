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


def _inverse_permutation(perm: torch.Tensor) -> torch.Tensor:
    """Return inverse permutations for a batch of permutation rows."""
    inverse = torch.empty_like(perm)
    positions = torch.arange(
        perm.size(1), dtype=torch.int64, device=perm.device
    ).expand_as(perm)
    inverse.scatter_(1, perm, positions)
    return inverse


def _capped_lpt_packing(
    weight: torch.Tensor,
    num_packs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched longest-processing-time packing with a fixed item count per pack.

    Every row is scheduled independently, but all rows are updated by the same
    tensor operations.  Items are visited from high to low load and placed on
    the least-loaded pack which still has capacity.
    """
    num_rows, num_items = weight.shape
    assert num_items % num_packs == 0
    capacity = num_items // num_packs

    if capacity == 1:
        pack = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand(num_rows, -1)
        return pack, torch.zeros_like(pack)

    # Sorting is the only ordering operation.  The subsequent loop is over
    # items, not layers or packs, and each iteration updates all rows at once.
    order = torch.argsort(weight.float(), dim=1, descending=True)
    pack_index = torch.empty(
        (num_rows, num_items), dtype=torch.int64, device=weight.device
    )
    rank_in_pack = torch.empty_like(pack_index)

    loads = torch.zeros(
        (num_rows, num_packs), dtype=torch.float32, device=weight.device
    )
    occupancy = torch.zeros(
        (num_rows, num_packs), dtype=torch.int64, device=weight.device
    )
    row_ids = torch.arange(num_rows, device=weight.device)

    for position in range(num_items):
        item = order[:, position]
        unavailable = occupancy >= capacity
        candidate_loads = loads.masked_fill(unavailable, float("inf"))
        destination = candidate_loads.argmin(dim=1)

        item_load = weight[row_ids, item].float()
        item_rank = occupancy.gather(1, destination.unsqueeze(1)).squeeze(1)

        pack_index.scatter_(1, item.unsqueeze(1), destination.unsqueeze(1))
        rank_in_pack.scatter_(1, item.unsqueeze(1), item_rank.unsqueeze(1))

        loads.scatter_add_(1, destination.unsqueeze(1), item_load.unsqueeze(1))
        occupancy.scatter_add_(
            1,
            destination.unsqueeze(1),
            torch.ones((num_rows, 1), dtype=torch.int64, device=weight.device),
        )

    # LPT is a good initial solution, but its irrevocable early choices can
    # leave a heavy pack with a beneficial item exchange.  For the small
    # node/GPU packing instances that dominate placement quality, perform a
    # few batched max-load-reducing swaps.  Each swap keeps pack cardinalities
    # exact and therefore preserves the layout contract of this function.
    if num_items <= 64 and num_packs > 1:
        negative_infinity = float("-inf")
        for _ in range(min(num_packs, 4)):
            source_pack = loads.argmax(dim=1)
            current_max = loads.gather(
                1, source_pack.unsqueeze(1)
            ).squeeze(1)

            # The largest load excluding the source and a prospective
            # destination pack can be obtained from the top two remaining
            # packs, avoiding an additional item-by-pack expansion.
            remaining_loads = loads.clone()
            remaining_loads.scatter_(
                1, source_pack.unsqueeze(1), negative_infinity
            )
            top_loads, top_packs = remaining_loads.topk(
                min(2, num_packs - 1), dim=1
            )
            other_max_default = top_loads[:, 0]
            if num_packs > 2:
                other_max_if_top_is_destination = top_loads[:, 1]
            else:
                other_max_if_top_is_destination = torch.full_like(
                    other_max_default, negative_infinity
                )

            item_loads = weight.float()
            source_items = pack_index == source_pack.unsqueeze(1)
            destination_packs = pack_index
            destination_is_top = destination_packs == top_packs[:, :1]
            other_max = torch.where(
                destination_is_top,
                other_max_if_top_is_destination.unsqueeze(1),
                other_max_default.unsqueeze(1),
            )

            # Row i, source item a, destination item b represents swapping
            # a and b. Invalid pairs are masked rather than materialized as
            # Python per-row searches.
            source_after = (
                current_max[:, None, None]
                - item_loads[:, :, None]
                + item_loads[:, None, :]
            )
            destination_load = loads.gather(1, destination_packs)
            destination_after = (
                destination_load[:, None, :]
                - item_loads[:, None, :]
                + item_loads[:, :, None]
            )
            candidate_max = torch.maximum(
                torch.maximum(source_after, destination_after),
                other_max[:, None, :],
            )
            valid_pair = (
                source_items[:, :, None]
                & ~source_items[:, None, :]
            )
            candidate_max.masked_fill_(~valid_pair, float("inf"))

            best_value, best_pair = candidate_max.flatten(1).min(dim=1)
            improve = best_value < current_max
            if not bool(improve.any()):
                break

            source_item = best_pair // num_items
            destination_item = best_pair % num_items
            rows = torch.arange(num_rows, device=weight.device)

            source_slot = rank_in_pack[rows, source_item]
            destination_slot = rank_in_pack[rows, destination_item]
            destination_pack = pack_index[rows, destination_item]

            # Only rows with a strict improvement are changed.
            selected_rows = rows[improve]
            selected_source = source_item[improve]
            selected_destination = destination_item[improve]
            pack_index[selected_rows, selected_source] = destination_pack[improve]
            pack_index[selected_rows, selected_destination] = source_pack[improve]
            rank_in_pack[selected_rows, selected_source] = destination_slot[improve]
            rank_in_pack[selected_rows, selected_destination] = source_slot[improve]

            delta = (
                item_loads[rows, destination_item]
                - item_loads[rows, source_item]
            )
            loads[rows[improve], source_pack[improve]] += delta[improve]
            loads[rows[improve], destination_pack[improve]] -= delta[improve]

    return pack_index, rank_in_pack


def _replicate_by_pressure(
    weight: torch.Tensor,
    num_physical: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate replicas by repeatedly reducing the greatest shard pressure.

    At each step an additional copy goes to the expert maximizing
    load / existing_copies.  This is a discrete water-filling policy for the
    maximum per-replica load objective and is evaluated for all rows at once.
    """
    num_rows, num_logical = weight.shape
    assert num_physical >= num_logical

    device = weight.device
    physical_to_logical = torch.empty(
        (num_rows, num_physical), dtype=torch.int64, device=device
    )
    replica_rank = torch.zeros_like(physical_to_logical)
    logical_count = torch.ones(
        (num_rows, num_logical), dtype=torch.int64, device=device
    )

    initial = torch.arange(
        num_logical, dtype=torch.int64, device=device
    ).expand(num_rows, -1)
    physical_to_logical[:, :num_logical] = initial

    row_ids = torch.arange(num_rows, device=device)
    for physical_id in range(num_logical, num_physical):
        pressure = weight.float() / logical_count.float()
        chosen = pressure.argmax(dim=1)
        current_rank = logical_count[row_ids, chosen]

        physical_to_logical[:, physical_id] = chosen
        replica_rank[:, physical_id] = current_rank
        logical_count[row_ids, chosen] += 1

    return physical_to_logical, replica_rank, logical_count


def _hierarchical_rebalance(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform node-local replication followed by capacity-aware GPU packing."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    logical_per_node = num_logical_experts // num_nodes
    physical_per_node = num_physical_experts // num_nodes
    gpus_per_node = num_gpus // num_nodes
    physical_per_gpu = num_physical_experts // num_gpus

    # First reserve node locality for logical expert groups.
    group_load = weight.reshape(num_layers, num_groups, group_size).sum(dim=-1)
    group_node, group_slot = _capped_lpt_packing(group_load, num_nodes)

    logical_to_node_order = (
        (group_node * groups_per_node + group_slot).unsqueeze(-1) * group_size
        + torch.arange(group_size, dtype=torch.int64, device=weight.device)
    ).reshape(num_layers, num_logical_experts)
    node_order_to_logical = _inverse_permutation(logical_to_node_order)

    # Each node receives its contiguous portion of the node-order namespace.
    node_local_weight = weight.gather(1, node_order_to_logical).reshape(
        num_layers * num_nodes, logical_per_node
    )

    local_phy_to_logical, local_rank, local_count = _replicate_by_pressure(
        node_local_weight, physical_per_node
    )

    # Each generated replica has the load of one shard of its logical expert.
    replica_load = (
        node_local_weight / local_count.float()
    ).gather(1, local_phy_to_logical)

    gpu_index, gpu_rank = _capped_lpt_packing(replica_load, gpus_per_node)
    packed_to_local_physical = gpu_index * physical_per_gpu + gpu_rank
    local_physical_to_packed = _inverse_permutation(packed_to_local_physical)

    packed_to_local_logical = local_phy_to_logical.gather(
        1, local_physical_to_packed
    )
    packed_rank = local_rank.gather(1, local_physical_to_packed)

    node_offsets = torch.arange(
        num_nodes, dtype=torch.int64, device=weight.device
    ).view(1, num_nodes, 1) * logical_per_node

    packed_to_node_order = (
        packed_to_local_logical.reshape(num_layers, num_nodes, physical_per_node)
        + node_offsets
    ).reshape(num_layers, num_physical_experts)

    physical_to_logical = node_order_to_logical.gather(
        1, packed_to_node_order
    )
    replica_rank = packed_rank.reshape(num_layers, num_physical_experts)

    logical_count = local_count.reshape(
        num_layers, num_nodes * logical_per_node
    ).gather(1, logical_to_node_order)

    return physical_to_logical, replica_rank, logical_count


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rearrange and replicate experts to balance their observed routing load.

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_replicas >= num_logical_experts

    # Rebalancing runs infrequently and returns CPU metadata, matching the
    # original interface.  float32 also provides stable scheduling comparisons.
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        physical_to_logical, replica_rank, logical_count = _hierarchical_rebalance(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        )
    else:
        # If group boundaries cannot be distributed evenly among nodes, use one
        # global node; GPU packing remains capacity constrained.
        physical_to_logical, replica_rank, logical_count = _hierarchical_rebalance(
            weight,
            num_replicas,
            1,
            1,
            num_gpus,
        )

    max_replica_count = num_replicas - num_logical_experts + 1
    logical_to_physical = torch.full(
        (num_layers, num_logical_experts, max_replica_count),
        -1,
        dtype=torch.int64,
        device=weight.device,
    )

    physical_ids = torch.arange(
        num_replicas, dtype=torch.int64, device=weight.device
    ).expand(num_layers, -1)
    flattened_slots = physical_to_logical * max_replica_count + replica_rank

    logical_to_physical.reshape(num_layers, -1).scatter_(
        1, flattened_slots, physical_ids
    )

    return physical_to_logical, logical_to_physical, logical_count


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]