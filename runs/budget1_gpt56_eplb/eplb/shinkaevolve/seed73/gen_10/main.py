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
    inverse = torch.empty_like(perm)
    inverse.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64,
                     device=perm.device).expand_as(perm),
    )
    return inverse


def _balanced_lpt_packing(
    weight: torch.Tensor,
    num_packs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Capacity-constrained LPT packing with one best cross-pack exchange."""
    num_rows, num_items = weight.shape
    assert num_items % num_packs == 0
    capacity = num_items // num_packs

    if capacity == 1:
        pack = torch.arange(num_items, dtype=torch.int64,
                            device=weight.device).expand_as(weight)
        return pack, torch.zeros_like(pack)

    weight = weight.float().cpu()
    order = weight.argsort(dim=-1, descending=True)
    pack_index = torch.empty((num_rows, num_items), dtype=torch.int64)
    rank_in_pack = torch.empty_like(pack_index)

    for row in range(num_rows):
        values = weight[row].tolist()
        loads = [0.0] * num_packs
        sizes = [0] * num_packs
        members: list[list[tuple[int, int, float]]] = [
            [] for _ in range(num_packs)
        ]

        available = [(0.0, pack) for pack in range(num_packs)]
        heapq.heapify(available)

        for item in order[row].tolist():
            _, pack = heapq.heappop(available)
            slot = sizes[pack]
            item_weight = values[item]

            pack_index[row, item] = pack
            rank_in_pack[row, item] = slot
            members[pack].append((item, slot, item_weight))
            sizes[pack] += 1
            loads[pack] += item_weight

            if sizes[pack] < capacity:
                heapq.heappush(available, (loads[pack], pack))

        # LPT is fast but can leave a poor pair of full packs. Search the
        # compact swap neighborhood once and retain only an improving move.
        if num_packs > 1:
            old_max = max(loads)
            best_max = old_max
            best_move = None

            for left in range(num_packs):
                for right in range(left + 1, num_packs):
                    other_max = max(
                        (loads[p] for p in range(num_packs)
                         if p != left and p != right),
                        default=float("-inf"),
                    )
                    for left_item, left_slot, left_weight in members[left]:
                        for right_item, right_slot, right_weight in members[
                                right]:
                            new_left = (loads[left] - left_weight +
                                        right_weight)
                            new_right = (loads[right] - right_weight +
                                         left_weight)
                            candidate = max(other_max, new_left, new_right)
                            if candidate < best_max:
                                best_max = candidate
                                best_move = (
                                    left,
                                    right,
                                    left_item,
                                    left_slot,
                                    right_item,
                                    right_slot,
                                    new_left,
                                    new_right,
                                )

            if best_move is not None:
                (left, right, left_item, left_slot, right_item, right_slot,
                 new_left, new_right) = best_move
                pack_index[row, left_item] = right
                rank_in_pack[row, left_item] = right_slot
                pack_index[row, right_item] = left
                rank_in_pack[row, right_item] = left_slot
                loads[left] = new_left
                loads[right] = new_right

    return pack_index, rank_in_pack


def _waterfill_replicas(
    weight: torch.Tensor,
    num_physical: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate replicas by selecting the largest discrete water-filling gains.

    The kth copy of expert e has load weight[e] / k. Selecting the largest
    available gains yields the optimal integer replica counts for minimizing
    the largest per-replica expert load under a fixed replica count.
    """
    num_rows, num_logical = weight.shape
    extra = num_physical - num_logical
    assert extra >= 0

    device = weight.device
    counts = torch.ones((num_rows, num_logical),
                        dtype=torch.int64,
                        device=device)

    if extra:
        # Candidate k corresponds to adding replica rank k, where the first
        # physical copy is rank zero and has already been reserved.
        divisors = torch.arange(2, extra + 2,
                                dtype=weight.dtype,
                                device=device)
        gains = (weight.unsqueeze(-1) / divisors).flatten(1)
        chosen = gains.topk(extra, dim=-1, largest=True, sorted=False).indices
        chosen_experts = torch.div(chosen,
                                   extra,
                                   rounding_mode="floor")
        counts.scatter_add_(1, chosen_experts,
                            torch.ones_like(chosen_experts,
                                            dtype=torch.int64))

    max_count = int(counts.max().item())
    experts = torch.arange(num_logical, dtype=torch.int64,
                           device=device).view(1, num_logical, 1)
    ranks = torch.arange(max_count, dtype=torch.int64,
                         device=device).view(1, 1, max_count)

    valid = ranks < counts.unsqueeze(-1)
    physical_to_logical = experts.expand(num_rows, -1,
                                         max_count)[valid].view(
                                             num_rows, num_physical)
    physical_rank = ranks.expand(num_rows, num_logical,
                                 -1)[valid].view(num_rows, num_physical)
    return physical_to_logical, physical_rank, counts


def _hierarchical_rebalance(
    weight: torch.Tensor,
    num_physical: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # Stage 1: locality-preserving group placement on nodes.
    group_load = weight.view(num_layers, num_groups, group_size).sum(-1)
    group_node, group_slot = _balanced_lpt_packing(group_load, num_nodes)
    logical_to_node_order = (
        (group_node * groups_per_node + group_slot).unsqueeze(-1) *
        group_size + torch.arange(group_size, dtype=torch.int64)
    ).flatten(-2)
    node_order_to_logical = _inverse_permutation(logical_to_node_order)

    # Stage 2: exact discrete water filling independently within each node.
    node_weight = weight.gather(1, node_order_to_logical).reshape(
        num_layers * num_nodes, logical_per_node)
    node_phy_to_logical, node_rank, node_count = _waterfill_replicas(
        node_weight, physical_per_node)

    # Stage 3: place replica jobs onto equal-capacity GPUs in every node.
    replica_load = (node_weight / node_count).gather(
        1, node_phy_to_logical)
    gpu_index, gpu_slot = _balanced_lpt_packing(replica_load, gpus_per_node)
    physical_position = gpu_index * physical_per_gpu + gpu_slot
    position_to_replica = _inverse_permutation(physical_position)

    ordered_node_logical = node_phy_to_logical.gather(1,
                                                       position_to_replica)
    ordered_rank = node_rank.gather(1, position_to_replica)

    node_offsets = torch.arange(
        0, num_logical, logical_per_node, dtype=torch.int64).view(1, -1, 1)
    merged_logical = (ordered_node_logical.view(num_layers, num_nodes, -1) +
                      node_offsets).flatten(1)
    physical_to_logical = node_order_to_logical.gather(1, merged_logical)
    physical_rank = ordered_rank.view(num_layers, -1)

    logical_count = node_count.view(num_layers, -1).gather(
        1, logical_to_node_order)
    return physical_to_logical, physical_rank, logical_count


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Rebalance logical experts across physical expert slots.

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, logical_experts, max_replicas]
        expert_count: [layers, logical_experts]
    """
    num_layers, num_logical = weight.shape
    assert num_replicas >= num_logical

    # EPLB metadata is consumed on CPU; moving once also avoids GPU scalar
    # synchronization during scheduling.
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = _hierarchical_rebalance(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # A global policy is the safe fallback when whole groups cannot be
        # assigned evenly to nodes.
        phy2log, phyrank, logcnt = _hierarchical_rebalance(
            weight, num_replicas, 1, 1, num_gpus)

    max_replicas_per_logical = num_replicas - num_logical + 1
    log2phy = torch.full(
        (num_layers, num_logical, max_replicas_per_logical),
        -1,
        dtype=torch.int64,
    )
    log2phy.view(num_layers, -1).scatter_(
        1,
        phy2log * max_replicas_per_logical + phyrank,
        torch.arange(num_replicas, dtype=torch.int64).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
