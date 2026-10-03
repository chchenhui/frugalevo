# SPDX-License-Identifier: Apache-2.0
"""
Expert parallelism load balancer (EPLB) for vLLM.
"""

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign descending loads to the least-loaded capacity-constrained pack."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(
            num_groups, dtype=torch.int64, device=weight.device
        ).expand_as(weight)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    values = weight.detach().float().cpu()
    indices = values.sort(dim=-1, descending=True).indices

    pack_index = torch.full(
        (num_layers, num_groups), -1, dtype=torch.int64, device="cpu"
    )
    rank_in_pack = torch.full_like(pack_index, -1)
    pack_weights = torch.zeros(
        (num_layers, num_packs), dtype=values.dtype, device="cpu"
    )
    pack_items = torch.zeros(
        (num_layers, num_packs), dtype=torch.int64, device="cpu"
    )
    inf = torch.tensor(float("inf"), dtype=values.dtype, device="cpu")

    for position in range(num_groups):
        group = indices[:, position]
        available = pack_items < groups_per_pack
        candidate_weights = pack_weights.masked_fill(~available, inf)
        pack = candidate_weights.argmin(dim=-1)
        rank = pack_items.gather(1, pack[:, None]).squeeze(1)

        pack_index.scatter_(1, group[:, None], pack[:, None])
        rank_in_pack.scatter_(1, group[:, None], rank[:, None])

        item_weight = values.gather(1, group[:, None]).squeeze(1)
        pack_weights.scatter_add_(1, pack[:, None], item_weight[:, None])
        pack_items.scatter_(1, pack[:, None], (rank + 1)[:, None])

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate logical experts such that the largest per-replica load is
    greedily minimized.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0

    device = weight.device
    phy2log = torch.arange(
        num_phy, dtype=torch.int64, device=device
    ).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """Form replica-aware synthetic groups, then pack them hierarchically."""
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
            torch.arange(
                perm.size(1), dtype=torch.int64, device=perm.device
            ).expand(perm.shape),
        )
        return inv

    # Estimate global replica counts first.  Experts are then assigned in
    # descending residual per-copy-load order to the least-loaded group with
    # available capacity.  This keeps the group cardinality invariant while
    # making synthetic groups aware of the later replication step.
    _, _, projected_count = replicate_experts(
        weight, num_physical_experts
    )
    projected_share = weight / projected_count.to(weight.dtype)
    priority = torch.argsort(projected_share, dim=-1, descending=True)

    synthetic_members = torch.empty(
        (num_layers, num_groups, group_size),
        dtype=torch.int64,
        device=weight.device,
    )
    tokens_per_group = torch.zeros(
        (num_layers, num_groups),
        dtype=weight.dtype,
        device=weight.device,
    )
    projected_group_load = torch.zeros_like(tokens_per_group)
    group_items = torch.zeros(
        (num_layers, num_groups),
        dtype=torch.int64,
        device=weight.device,
    )
    layer_index = torch.arange(
        num_layers, dtype=torch.int64, device=weight.device
    )

    for position in range(num_logical_experts):
        expert = priority[:, position]
        available = group_items < group_size
        candidate_load = projected_group_load.masked_fill(
            ~available, float("inf")
        )
        group = candidate_load.argmin(dim=-1)
        rank = group_items.gather(1, group[:, None]).squeeze(1)

        # Advanced indexing updates only the selected (layer, group, rank)
        # slot, avoiding flattened indices and cross-group scatter overflow.
        synthetic_members[layer_index, group, rank] = expert

        expert_load = weight.gather(1, expert[:, None]).squeeze(1)
        share_load = projected_share.gather(
            1, expert[:, None]
        ).squeeze(1)
        tokens_per_group.scatter_add_(
            1, group[:, None], expert_load[:, None]
        )
        projected_group_load.scatter_add_(
            1, group[:, None], share_load[:, None]
        )
        group_items.scatter_(
            1, group[:, None], (rank + 1)[:, None]
        )

    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes
    )

    packed_groups = group_pack_index * groups_per_node + group_rank_in_pack
    packed_members = synthetic_members.gather(
        1,
        packed_groups.unsqueeze(-1).expand(-1, -1, group_size),
    )
    log2mlog = packed_members.flatten(-2)
    mlog2log = inverse(log2mlog)

    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy, num_gpus // num_nodes
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (
        pphy2mlog.view(num_layers, num_nodes, -1)
        + torch.arange(
            0,
            num_logical_experts,
            num_logical_experts // num_nodes,
            device=group_pack_index.device,
        ).view(1, -1, 1)
    ).flatten(-2)

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
    Entry point for expert-parallelism load balancing.
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

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(
            num_replicas, dtype=torch.int64, device=log2phy.device
        ).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts"]