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
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Population-of-constructions: build several deterministic greedy
    # candidates per layer (incumbent LPT, snake-draft, rotated-LPT, and
    # ascending-order LPT), then select, per layer, the candidate with the
    # smallest max pack load (mean-load tiebreak). Since the incumbent LPT
    # is one of the candidates, the chosen packing is pointwise
    # non-regressing on every layer. All candidates are fully vectorized
    # (one round per pack-fill level, no per-item Python loops).
    wf = weight.float()
    desc = wf.sort(-1, descending=True).indices
    asc = wf.sort(-1).indices

    def run_candidate(indices: torch.Tensor,
                      mode: str) -> tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor]:
        pack_index = torch.full_like(weight, -1, dtype=torch.int64)
        rank_in_pack = torch.full_like(pack_index, -1)
        pack_weights = torch.zeros(num_layers, num_packs,
                                   dtype=torch.float32)
        for r in range(groups_per_pack):
            item_idx = indices[:, r * num_packs:(r + 1) * num_packs]
            pack_order = pack_weights.argsort(-1)
            if mode == "snake" and (r & 1):
                pack_order = pack_order.flip(-1)
            elif mode == "rotated":
                pack_order = torch.roll(pack_order, r % num_packs, -1)
            pack_index.scatter_(1, item_idx, pack_order)
            rank_in_pack.scatter_(1, item_idx,
                                  torch.full_like(item_idx, r))
            pack_weights.scatter_add_(
                1, pack_order, wf.gather(1, item_idx))
        return pack_index, rank_in_pack, pack_weights

    candidates = [
        run_candidate(desc, "lpt"),
        run_candidate(desc, "snake"),
        run_candidate(desc, "rotated"),
        run_candidate(asc, "lpt"),
    ]
    # Per-layer lexicographic key: (max pack load, mean pack load).
    # Combine into one scalar so argmin is exact and tie-safe.
    keys = torch.stack([
        pw.max(-1).values * 1e6 + pw.mean(-1) for _, _, pw in candidates
    ])  # [num_candidates, num_layers]
    best = keys.argmin(0)  # [num_layers]

    pack_index = torch.stack([c[0] for c in candidates])[best,
                                                         torch.arange(
                                                             num_layers)]
    rank_in_pack = torch.stack([c[1] for c in candidates])[best,
                                                           torch.arange(
                                                               num_layers)]
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def _global_balance(weight: torch.Tensor, num_replicas: int,
                    num_gpus: int) -> tuple[torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
    """
    Global replicate-then-place pipeline: (a) replicate experts against the
    GLOBAL load distribution via greedy replication, (b) compute each
    physical expert's load as weight/logcnt, (c) run one global
    balanced_packing over [layers, num_replicas] with num_packs=num_gpus to
    assign physical experts to GPUs. This strictly enlarges the feasible
    placement set versus the hierarchical node-confined cascade.
    """
    num_layers, num_logical_experts = weight.shape
    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    phy2pphy = pack_index * (num_replicas // num_gpus) + rank_in_pack
    inv = torch.empty_like(phy2pphy)
    inv.scatter_(
        1, phy2pphy,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=phy2pphy.device).expand(num_layers, -1))
    pphy2log = phy2log.gather(-1, inv)
    pphyrank = phyrank.gather(-1, inv)
    return pphy2log, pphyrank, logcnt


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
    try:
        # Primary path: global replication + global GPU packing
        phy2log, phyrank, logcnt = _global_balance(weight, num_replicas,
                                                   num_gpus)
    except Exception:
        # Fallback: hierarchical group->node->GPU pipeline
        if num_groups % num_nodes == 0:
            phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_gpus)
        else:
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

