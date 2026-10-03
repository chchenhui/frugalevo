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
    """
    Assign equally many weighted items to each pack.

    A largest-first heap assignment provides the initial schedule. A bounded
    exchange refinement then improves the maximum/minimum pack-load range
    without changing pack cardinalities.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs
    device = weight.device

    # These cases have no balancing decision to make. Keeping the input order
    # also avoids needless remapping when load statistics are uniform.
    if num_packs == 1:
        return (torch.zeros_like(weight, dtype=torch.int64),
                torch.arange(num_items, dtype=torch.int64,
                             device=device).expand(num_layers, -1))

    if items_per_pack == 1:
        return (torch.arange(num_items, dtype=torch.int64,
                             device=device).expand(num_layers, -1),
                torch.zeros_like(weight, dtype=torch.int64))

    # Exact uniformity is a strictly safe shortcut: every capacity-respecting
    # assignment has identical pack load.
    if weight.numel() == 0 or torch.all(weight == weight.reshape(-1)[0]):
        item_ids = torch.arange(num_items, dtype=torch.int64, device=device)
        return (item_ids.div(items_per_pack, rounding_mode="floor").expand(
            num_layers, -1),
                item_ids.remainder(items_per_pack).expand(num_layers, -1))

    cpu_weight = weight.float().cpu()
    order = cpu_weight.sort(dim=-1, descending=True).indices
    pack_index = torch.empty((num_layers, num_items),
                             dtype=torch.int64,
                             device="cpu")
    rank_in_pack = torch.empty_like(pack_index)

    # Refinement is deliberately bounded. It is especially useful for the
    # relatively small node/GPU pack counts used by EPLB, while retaining the
    # low runtime of greedy scheduling for large expert counts.
    max_refinement_rounds = min(8, num_packs)

    for layer in range(num_layers):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        members = [[] for _ in range(num_packs)]
        available = [(0.0, pack) for pack in range(num_packs)]
        heapq.heapify(available)

        for item in order[layer].tolist():
            current_load, pack = heapq.heappop(available)
            slot = counts[pack]
            value = cpu_weight[layer, item].item()

            pack_index[layer, item] = pack
            rank_in_pack[layer, item] = slot
            members[pack].append(item)
            counts[pack] += 1
            loads[pack] = current_load + value

            if counts[pack] < items_per_pack:
                heapq.heappush(available, (loads[pack], pack))

        # Pairwise exchanges between the extremal packs repair common greedy
        # overshoots. Members retain their destination slot/rank, so the
        # resulting assignment remains a valid exact-capacity permutation.
        for _ in range(max_refinement_rounds):
            heavy = max(range(num_packs), key=loads.__getitem__)
            light = min(range(num_packs), key=loads.__getitem__)
            old_range = loads[heavy] - loads[light]
            if old_range <= 0.0:
                break

            old_square = sum(x * x for x in loads)
            best = None
            best_key = (old_range, old_square)

            for heavy_pos, heavy_item in enumerate(members[heavy]):
                heavy_value = cpu_weight[layer, heavy_item].item()
                for light_pos, light_item in enumerate(members[light]):
                    light_value = cpu_weight[layer, light_item].item()
                    new_heavy = loads[heavy] - heavy_value + light_value
                    new_light = loads[light] - light_value + heavy_value

                    candidate_loads = loads.copy()
                    candidate_loads[heavy] = new_heavy
                    candidate_loads[light] = new_light
                    candidate_range = (max(candidate_loads) -
                                       min(candidate_loads))
                    candidate_square = sum(x * x for x in candidate_loads)
                    candidate_key = (candidate_range, candidate_square)

                    if candidate_key < best_key:
                        best_key = candidate_key
                        best = (heavy_pos, light_pos, heavy_item, light_item,
                                new_heavy, new_light)

            if best is None:
                break

            (heavy_pos, light_pos, heavy_item, light_item, new_heavy,
             new_light) = best

            heavy_rank = rank_in_pack[layer, heavy_item].item()
            light_rank = rank_in_pack[layer, light_item].item()
            members[heavy][heavy_pos] = light_item
            members[light][light_pos] = heavy_item
            pack_index[layer, heavy_item] = light
            pack_index[layer, light_item] = heavy
            rank_in_pack[layer, heavy_item] = light_rank
            rank_in_pack[layer, light_item] = heavy_rank
            loads[heavy] = new_heavy
            loads[light] = new_light

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate redundant replicas by repeatedly splitting the largest current
    per-replica logical-expert load.
    """
    num_rows, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0

    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).expand(num_rows, -1).clone()
    rank = torch.zeros((num_rows, num_phy), dtype=torch.int64, device=device)
    logcnt = torch.ones((num_rows, num_log), dtype=torch.int64, device=device)

    if num_redundant == 0:
        return phy2log, rank, logcnt

    rows = torch.arange(num_rows, dtype=torch.int64, device=device)
    for physical_id in range(num_log, num_phy):
        selected = (weight / logcnt).max(dim=-1).indices
        phy2log[:, physical_id] = selected
        rank[:, physical_id] = logcnt[rows, selected]
        logcnt[rows, selected] += 1

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perform node-aware replication followed by GPU-aware placement."""
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0

    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    physical_per_gpu = num_physical_experts // num_gpus
    logical_per_node = num_logical_experts // num_nodes
    physical_per_node = num_physical_experts // num_nodes

    def inverse(permutation: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(permutation)
        result.scatter_(
            1,
            permutation,
            torch.arange(permutation.size(1), dtype=torch.int64,
                         device=permutation.device).expand_as(permutation),
        )
        return result

    tokens_per_group = weight.view(num_layers, num_groups,
                                   group_size).sum(dim=-1)
    node_of_group, group_slot = balanced_packing(tokens_per_group, num_nodes)

    logical_offsets = torch.arange(group_size, dtype=torch.int64,
                                   device=weight.device)
    log2mlog = (((node_of_group * groups_per_node + group_slot) * group_size)
                .unsqueeze(-1) + logical_offsets).flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).reshape(
        num_layers * num_nodes, logical_per_node)
    phy2mlog, replica_rank, mlog_count = replicate_experts(
        tokens_per_mlog, physical_per_node)

    replica_load = (tokens_per_mlog / mlog_count).gather(-1, phy2mlog)
    gpu_of_phy, gpu_slot = balanced_packing(replica_load,
                                            num_gpus // num_nodes)
    phy2packed = gpu_of_phy * physical_per_gpu + gpu_slot
    packed2phy = inverse(phy2packed)

    packed2mlog = phy2mlog.gather(-1, packed2phy)
    node_offsets = torch.arange(
        0,
        num_logical_experts,
        logical_per_node,
        dtype=torch.int64,
        device=weight.device,
    )
    packed2mlog = (packed2mlog.reshape(num_layers, num_nodes, -1) +
                   node_offsets.view(1, num_nodes, 1)).flatten(-2)

    phy2log = mlog2log.gather(-1, packed2mlog)
    phy_rank = replica_rank.gather(-1, packed2phy).reshape(
        num_layers, num_physical_experts)
    log_count = mlog_count.reshape(num_layers, -1).gather(-1, log2mlog)
    return phy2log, phy_rank, log_count


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute physical-to-logical and logical-to-physical expert mappings.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # If declared group boundaries cannot be evenly assigned to nodes,
        # global placement is the topology-safe fallback.
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    max_logical_replicas = num_replicas - num_logical_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_logical_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    physical_ids = torch.arange(num_replicas, dtype=torch.int64,
                                device=logcnt.device).expand(num_layers, -1)
    log2phy.view(num_layers, -1).scatter_(
        1,
        phy2log * max_logical_replicas + phyrank,
        physical_ids,
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]
