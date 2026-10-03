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
    """Return the inverse of a batch of permutations."""
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
    Assign equal-cardinality item sets to packs.

    Items are considered from largest to smallest.  A pack-sized batch of
    items is paired with packs ordered from currently least loaded to most
    loaded.  This is the batched antitone form of greedy list scheduling:
    every round fills exactly one slot in every pack, so capacities are
    guaranteed without an item-at-a-time capacity search.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    if items_per_pack == 1:
        pack_index = torch.arange(
            num_items, dtype=torch.int64, device=weight.device
        ).expand(num_layers, -1)
        return pack_index, torch.zeros_like(pack_index)

    # Stable ordering makes equal-load layouts reproducible where supported.
    try:
        sorted_items = weight.sort(dim=-1, descending=True, stable=True).indices
    except TypeError:
        sorted_items = weight.sort(dim=-1, descending=True).indices
    sorted_weight = weight.gather(1, sorted_items)

    pack_index = torch.empty(
        (num_layers, num_items), dtype=torch.int64, device=weight.device
    )
    rank_in_pack = torch.empty_like(pack_index)
    pack_load = torch.zeros(
        (num_layers, num_packs), dtype=weight.dtype, device=weight.device
    )

    # The loop is over physical slots per pack, not layers or individual
    # experts.  All layers and all packs in a round are handled together.
    for slot in range(items_per_pack):
        begin = slot * num_packs
        end = begin + num_packs
        item_ids = sorted_items[:, begin:end]
        item_weight = sorted_weight[:, begin:end]

        # Largest remaining item goes to the least-loaded available pack.
        least_loaded_packs = pack_load.argsort(dim=1)
        pack_index.scatter_(1, item_ids, least_loaded_packs)
        rank_in_pack.scatter_(
            1,
            item_ids,
            torch.full_like(item_ids, slot),
        )
        pack_load.scatter_add_(1, least_loaded_packs, item_weight)

    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor,
    num_phy: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate replicas using a discrete waterline selection.

    For a logical expert of load w, its k-th physical copy gives a waterline
    w / k.  Selecting the largest `num_phy - num_log` candidate waterlines
    selects the replica counts that minimize the maximum load of any replica.
    """
    num_layers, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0

    device = weight.device
    base_experts = torch.arange(
        num_log, dtype=torch.int64, device=device
    ).expand(num_layers, -1)

    if num_redundant == 0:
        return (
            base_experts,
            torch.zeros_like(base_experts),
            torch.ones_like(base_experts),
        )

    # Candidate column j represents replica rank j + 1, i.e. the
    # (j + 2)-th total copy of the corresponding logical expert.
    replica_number = torch.arange(
        2, num_redundant + 2, dtype=weight.dtype, device=device
    )
    candidate = weight.unsqueeze(-1) / replica_number

    # Zero-load experts have equal useful waterlines.  Give them decreasing
    # negative fallback values so selected candidates are always a prefix
    # (rank 1 before rank 2), which keeps logical-to-physical maps compact.
    fallback = -replica_number.view(1, 1, -1)
    candidate = torch.where(
        weight.unsqueeze(-1) > 0,
        candidate,
        fallback,
    )

    _, selected = candidate.flatten(1).topk(
        num_redundant, dim=1, largest=True, sorted=False
    )
    extra_logical = selected // num_redundant
    extra_rank = selected.remainder(num_redundant) + 1

    logical_count = torch.ones(
        (num_layers, num_log), dtype=torch.int64, device=device
    )
    logical_count.scatter_add_(1, extra_logical, torch.ones_like(extra_logical))

    phy2log = torch.cat((base_experts, extra_logical), dim=1)
    phy_rank = torch.cat(
        (torch.zeros_like(base_experts), extra_rank),
        dim=1,
    )
    return phy2log, phy_rank, logical_count


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hierarchical node-local replication followed by GPU slot packing."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    gpus_per_node = num_gpus // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus
    logical_per_node = num_logical_experts // num_nodes
    physical_per_node = num_physical_experts // num_nodes

    # 1. Balance complete routing groups across nodes.
    tokens_per_group = weight.reshape(
        num_layers, num_groups, group_size
    ).sum(dim=-1)
    group_pack, group_rank = balanced_packing(tokens_per_group, num_nodes)

    local_logical_to_global = (
        (group_pack * groups_per_node + group_rank) * group_size
    ).unsqueeze(-1) + torch.arange(
        group_size, dtype=torch.int64, device=weight.device
    )
    local_logical_to_global = local_logical_to_global.flatten(1)
    global_logical_to_local = _inverse_permutation(local_logical_to_global)

    # 2. Allocate redundant copies independently inside each node.
    node_local_weight = weight.gather(
        1, global_logical_to_local
    ).reshape(num_layers * num_nodes, logical_per_node)
    phy2local, phy_rank, local_count = replicate_experts(
        node_local_weight, physical_per_node
    )

    # 3. Balance equal-load physical-expert slots among local GPUs.
    replica_load = (
        node_local_weight / local_count.to(node_local_weight.dtype)
    ).gather(1, phy2local)
    gpu_pack, gpu_rank = balanced_packing(replica_load, gpus_per_node)
    physical_to_gpu_slot = gpu_pack * phy_experts_per_gpu + gpu_rank
    gpu_slot_to_physical = _inverse_permutation(physical_to_gpu_slot)

    ordered_local = phy2local.gather(1, gpu_slot_to_physical)
    ordered_rank = phy_rank.gather(1, gpu_slot_to_physical)

    # Convert local node expert ids back to original global logical ids.
    ordered_local = ordered_local.reshape(
        num_layers, num_nodes, physical_per_node
    )
    ordered_local += torch.arange(
        0,
        num_logical_experts,
        logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    ).view(1, num_nodes, 1)
    ordered_local = ordered_local.flatten(1)

    physical_to_logical = global_logical_to_local.gather(1, ordered_local)
    physical_rank = ordered_rank.reshape(num_layers, num_physical_experts)
    logical_count = local_count.reshape(
        num_layers, num_logical_experts
    ).gather(1, local_logical_to_global)

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

    # EPLB layout metadata is CPU-side and small relative to model tensors.
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phy_rank, logical_count = rebalance_experts_hierarchical(
            weight,
            num_replicas,
            num_groups,
            num_nodes,
            num_gpus,
        )
    else:
        # A single global group preserves the original fallback behavior.
        phy2log, phy_rank, logical_count = rebalance_experts_hierarchical(
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
    logical_to_physical.flatten(1).scatter_(
        1,
        phy2log * max_replica_count + phy_rank,
        physical_ids,
    )

    return phy2log, logical_to_physical, logical_count


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]

