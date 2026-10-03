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
                     num_packs: int,
                     num_trials: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Approach: multi-start randomized serpentine packing with vectorized
    best-of-K selection. Items are sorted by weight (descending) and dealt
    into blocks of num_packs. For each of K independent trials, the pack
    order within every block is a fresh random permutation (still one item
    per pack per block, so cardinality n/m per pack is preserved by
    construction). All K trials for all rows run as one batched
    [K * X, n] sort/scatter, pack loads are summed with a single
    scatter_add_, and per row the trial minimizing the maximum pack load
    is selected via argmin over the trial axis. This is an embarrassingly
    parallel metaheuristic that closes most of the gap to true LPT without
    any Python-level per-item loop.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        num_trials: number of randomized restarts (best-of-K)

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=weight.device).expand(
                                      num_layers, num_groups)
        rank_in_pack = torch.zeros(num_layers, num_groups, dtype=torch.int64,
                                   device=weight.device)
        return pack_index, rank_in_pack

    if num_trials < 1:
        num_trials = 1
    K = num_trials

    # Sort items by weight, descending, once for all trials.
    indices = weight.sort(-1, descending=True).indices  # [X, n]
    indices = indices.unsqueeze(0).expand(K, num_layers, num_groups)
    indices = indices.reshape(K * num_layers, num_groups)

    # Random pack order within each block, per trial.
    # [K, groups_per_pack, num_packs]: independent permutation per block.
    rand = torch.rand(K, groups_per_pack, num_packs, device=weight.device)
    block_perm = rand.argsort(-1).to(torch.int64)
    pack_sorted = block_perm.reshape(K, 1, num_groups).expand(
        K, num_layers, num_groups).reshape(K * num_layers, num_groups)
    # Rank within pack = block index (each pack receives one item per block).
    rank_sorted = torch.arange(groups_per_pack, dtype=torch.int64,
                               device=weight.device).view(-1, 1).expand(
                                   groups_per_pack,
                                   num_packs).reshape(1, num_groups)
    rank_sorted = rank_sorted.unsqueeze(0).expand(
        K, num_layers, num_groups).reshape(K * num_layers, num_groups)

    # Scatter sorted deal back to original item positions, batched over
    # all K * X rows at once.
    pack_index = torch.empty(K * num_layers, num_groups, dtype=torch.int64,
                             device=weight.device)
    rank_in_pack = pack_index.new_empty(K * num_layers, num_groups)
    pack_index.scatter_(1, indices, pack_sorted)
    rank_in_pack.scatter_(1, indices, rank_sorted)

    # Vectorized scoring: per-pack load sums for every (trial, row).
    loads = torch.zeros(K * num_layers, num_packs,
                        dtype=weight.dtype, device=weight.device)
    loads.scatter_add_(1, pack_index,
                       weight.unsqueeze(0).expand(K, num_layers,
                                                  num_groups).reshape(
                                                      K * num_layers,
                                                      num_groups))
    max_load = loads.max(-1).values.view(K, num_layers)  # [K, X]

    # Best-of-K selection per row.
    best = max_load.argmin(0)  # [X]
    rows = torch.arange(num_layers, device=weight.device)
    pack_index = pack_index.view(K, num_layers, num_groups)[best, rows]
    rank_in_pack = rank_in_pack.view(K, num_layers, num_groups)[best, rows]
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

    def _max_gpu_load(phy2log_: torch.Tensor,
                      logcnt_: torch.Tensor) -> torch.Tensor:
        """Estimated max per-GPU token load for each layer.

        Slot i hosts logical expert phy2log_[l, i] carrying
        weight / replicas of its load; slots are summed per GPU
        (gpu = i // experts_per_gpu) and the worst GPU load returned.
        """
        per_phy = weight.gather(1, phy2log_) / logcnt_.gather(1, phy2log_)
        gpu_of_slot = torch.arange(
            num_replicas, device=phy2log_.device) // (
                num_replicas // num_gpus)
        gpu_load = torch.zeros(num_layers, num_gpus,
                               dtype=per_phy.dtype, device=per_phy.device)
        gpu_load.scatter_add_(
            1, gpu_of_slot.unsqueeze(0).expand(num_layers, -1), per_phy)
        return gpu_load.max(-1).values

    # Always compute the flat (single-level) policy: replicate experts
    # globally and pack all physical experts directly onto all GPUs,
    # removing the node/group packing bottleneck that can cap achievable
    # per-GPU balance.
    flat_phy2log, flat_phyrank, flat_logcnt = rebalance_experts_hierarchical(
        weight, num_replicas, 1, 1, num_gpus)
    if num_groups % num_nodes == 0:
        # Also compute the hierarchical policy and keep, per layer
        # (layers are independent), whichever yields the lower max
        # per-GPU load. Both pipelines are vectorized, so this is cheap.
        hier_phy2log, hier_phyrank, hier_logcnt = (
            rebalance_experts_hierarchical(
                weight, num_replicas, num_groups, num_nodes, num_gpus))
        pick_flat = (_max_gpu_load(flat_phy2log, flat_logcnt) <=
                     _max_gpu_load(hier_phy2log, hier_logcnt))
        rows = torch.arange(num_layers, device=flat_phy2log.device)
        hier_rows = (~pick_flat).nonzero(as_tuple=True)[0]
        flat_rows = pick_flat.nonzero(as_tuple=True)[0]
        phy2log = flat_phy2log.clone()
        phyrank = flat_phyrank.clone()
        logcnt = flat_logcnt.clone()
        phy2log[hier_rows] = hier_phy2log[hier_rows]
        phyrank[hier_rows] = hier_phyrank[hier_rows]
        logcnt[hier_rows] = hier_logcnt[hier_rows]
        _ = rows
    else:
        phy2log, phyrank, logcnt = (flat_phy2log, flat_phyrank, flat_logcnt)
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

