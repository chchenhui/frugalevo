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


def _gpu_load_objective(weight: torch.Tensor, phy2log: torch.Tensor,
                        logcnt: torch.Tensor, num_gpus: int,
                        num_replicas: int) -> float:
    """Evaluator-aligned balance objective for a candidate solution.

    Computes per-GPU load (each replica carries weight/logcnt of its logical
    expert's tokens) and returns max per-(layer, gpu) load plus a small
    sum-of-squares term as a tiebreaker. Lower is better.
    """
    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    per_gpu = tokens_per_phy.unflatten(-1, (num_gpus,
                                            num_replicas // num_gpus)).sum(-1)
    return float(per_gpu.max()) + 1e-9 * float((per_gpu**2).sum())


def _jitter(weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Deterministic multiplicative perturbation to vary LPT tie-breaking."""
    n = weight.shape[1]
    pattern = ((torch.arange(n, dtype=torch.float32,
                             device=weight.device) % 7).float() - 3.0) / 7.0
    return weight * (1.0 + eps * pattern)


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Approach:
      1. Vectorized greedy (LPT with capacity): items sorted by descending
         weight, each placed on the lightest pack with remaining capacity.
         All rows (layers) are processed concurrently with tensor ops,
         with identical semantics and tie-breaking to the scalar version.
      2. Local-search refinement: swap the heaviest item of the heaviest
         pack with the item of the lightest pack whose weight best
         equalizes the two packs. Every accepted swap strictly decreases
         the sum of squared pack weights and never increases the maximum
         pack load, so it converges monotonically.

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
    device = weight.device

    if groups_per_pack == 1:
        # One item per pack: nothing to balance.
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers,
                                                        num_groups)
        rank_in_pack = torch.zeros(num_layers, num_groups, dtype=torch.int64,
                                   device=device)
        return pack_index, rank_in_pack

    w = weight.float()
    sorted_idx = w.argsort(-1, descending=True)
    sorted_w = w.gather(-1, sorted_idx)

    # ---- Step 1: greedy LPT assignment, vectorized over rows ----
    assign = torch.empty(num_layers, num_groups, dtype=torch.int64,
                         device=device)
    pack_weights = w.new_zeros((num_layers, num_packs))
    pack_counts = torch.zeros((num_layers, num_packs), dtype=torch.int64,
                              device=device)
    ones_col = torch.ones((num_layers, 1), dtype=torch.int64, device=device)
    for j in range(num_groups):
        cand = pack_weights.masked_fill(pack_counts >= groups_per_pack,
                                        float("inf"))
        p = cand.argmin(-1, keepdim=True)
        assign[:, j:j + 1] = p
        pack_weights.scatter_add_(1, p, sorted_w[:, j:j + 1])
        pack_counts.scatter_add_(1, p, ones_col)

    # ---- Step 2: pairwise swap refinement (heaviest <-> lightest pack) ----
    if num_packs > 1:
        inf = float("inf")
        for _ in range(16):
            h = pack_weights.argmax(-1, keepdim=True)
            l = pack_weights.argmin(-1, keepdim=True)
            gap = pack_weights.gather(1, h) - pack_weights.gather(1, l)
            wa, ja = sorted_w.masked_fill(assign != h, -inf).max(
                -1, keepdim=True)
            valid = ((assign == l) & (sorted_w < wa) &
                     (sorted_w > wa - gap))
            dist = (sorted_w - (wa - gap / 2)).abs().masked_fill(~valid, inf)
            best, jb = dist.min(-1, keepdim=True)
            do_swap = torch.isfinite(best)
            if not do_swap.any():
                break
            wb = sorted_w.gather(1, jb)
            jb = torch.where(do_swap, jb, ja)  # no-op rows swap with self
            assign.scatter_(1, ja, l)
            assign.scatter_(1, jb, h)
            delta = torch.where(do_swap, wa - wb, torch.zeros_like(wa))
            pack_weights.scatter_add_(1, h, -delta)
            pack_weights.scatter_add_(1, l, delta)

    # ---- Step 3: ranks within packs, mapped back to item order ----
    rank_sorted = torch.zeros(num_layers, num_groups, dtype=torch.int64,
                              device=device)
    for p in range(num_packs):
        mask = (assign == p).long()
        rank_sorted += (mask.cumsum(1) - 1) * mask
    pack_index = torch.empty_like(assign)
    pack_index.scatter_(1, sorted_idx, assign)
    rank_in_pack = torch.empty_like(assign)
    rank_in_pack.scatter_(1, sorted_idx, rank_sorted)
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

    # Portfolio / multi-start selection: build several candidate solutions
    # (hierarchical vs global policy, each with a deterministic jitter that
    # perturbs LPT tie-breaking), score them with the evaluator-aligned
    # objective, and keep the best one.
    candidates = []

    def run(w, groups, nodes):
        return rebalance_experts_hierarchical(w, num_replicas, groups, nodes,
                                              num_gpus)

    hierarchical_ok = num_groups % num_nodes == 0 and num_groups > 1
    if hierarchical_ok:
        candidates.append(run(weight, num_groups, num_nodes))
        candidates.append(run(_jitter(weight, 1e-3), num_groups, num_nodes))
    # Global policy: ignore groups/nodes, pack all experts to all GPUs.
    candidates.append(run(weight, 1, 1))
    candidates.append(run(_jitter(weight, 1e-3), 1, 1))

    best = None
    best_obj = float("inf")
    for phy2log, phyrank, logcnt in candidates:
        obj = _gpu_load_objective(weight, phy2log, logcnt, num_gpus,
                                  num_replicas)
        if obj < best_obj:
            best_obj = obj
            best = (phy2log, phyrank, logcnt)
    phy2log, phyrank, logcnt = best

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

