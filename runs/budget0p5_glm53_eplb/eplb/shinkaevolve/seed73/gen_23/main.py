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


def _inverse_perm(perm: torch.Tensor) -> torch.Tensor:
    """Invert a batched permutation along the last dim."""
    inv = torch.empty_like(perm)
    inv.scatter_(
        1,
        perm,
        torch.arange(perm.size(1), dtype=torch.int64,
                     device=perm.device).expand(perm.shape),
    )
    return inv


class _LptPacker:
    """Longest-Processing-Time-first packer with fixed pack capacity.

    Packs n weighted items into m packs, each holding exactly n/m items,
    minimizing the maximum pack weight.
    """

    @staticmethod
    def pack(weight: torch.Tensor,
             num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
        num_items = weight.size(-1)
        assert num_items % num_packs == 0
        cap = num_items // num_packs

        if cap == 1:
            pack_index = torch.arange(num_items, dtype=torch.int64,
                                      device=weight.device).expand(
                                          weight.shape).contiguous()
            rank_in_pack = torch.zeros_like(pack_index)
            return pack_index, rank_in_pack

        indices = weight.argsort(dim=-1, descending=True).tolist()
        weights = weight.tolist()
        num_rows = weight.size(0)
        pack_index = torch.empty(num_rows, num_items, dtype=torch.int64)
        rank_in_pack = torch.empty(num_rows, num_items, dtype=torch.int64)

        for r in range(num_rows):
            row_idx = indices[r]
            row_w = weights[r]
            pack_load = [0.0] * num_packs
            pack_count = [0] * num_packs
            heap = [(0.0, p) for p in range(num_packs)]
            heapq.heapify(heap)
            pi_row = pack_index[r]
            ri_row = rank_in_pack[r]
            for item in row_idx:
                load, pack = heapq.heappop(heap)
                pi_row[item] = pack
                ri_row[item] = pack_count[pack]
                pack_count[pack] += 1
                load += row_w[item]
                pack_load[pack] = load
                if pack_count[pack] < cap:
                    heapq.heappush(heap, (load, pack))
        return pack_index, rank_in_pack


class _ExpertReplicator:
    """Greedy replication: repeatedly clone the expert with the highest
    per-replica load so the maximum replica load is minimized."""

    @staticmethod
    def replicate(weight: torch.Tensor,
                  num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            redundant_indices = (weight / logcnt).argmax(dim=-1)
            phy2log[:, i] = redundant_indices
            rank[:, i] = logcnt[arangen, redundant_indices]
            logcnt[arangen, redundant_indices] += 1
        return phy2log, rank, logcnt


class RebalancePipeline:
    """Orchestrates the three pipeline stages and assembles final maps."""

    def __init__(self, num_physical_experts: int, num_groups: int,
                 num_nodes: int, num_gpus: int):
        self.num_physical_experts = num_physical_experts
        self.num_groups = num_groups
        self.num_nodes = num_nodes
        self.num_gpus = num_gpus

    def run(self, weight: torch.Tensor):
        num_layers, num_logical_experts = weight.shape
        n_groups = self.num_groups
        n_nodes = self.num_nodes
        n_gpus = self.num_gpus
        n_phy = self.num_physical_experts

        assert num_logical_experts % n_groups == 0
        group_size = num_logical_experts // n_groups
        assert n_groups % n_nodes == 0
        groups_per_node = n_groups // n_nodes
        assert n_gpus % n_nodes == 0
        assert n_phy % n_gpus == 0
        phy_per_gpu = n_phy // n_gpus

        # ---- Stage 1: pack groups to nodes ----
        tokens_per_group = weight.unflatten(-1,
                                            (n_groups, group_size)).sum(-1)
        pack_index, rank_in_pack = _LptPacker.pack(tokens_per_group, n_nodes)
        log2mlog = (((pack_index * groups_per_node + rank_in_pack) *
                    group_size).unsqueeze(-1) +
                   torch.arange(group_size, dtype=torch.int64,
                                device=weight.device)).flatten(-2)
        mlog2log = _inverse_perm(log2mlog)

        # ---- Stage 2: replicate experts within nodes ----
        tokens_per_mlog = weight.gather(-1, mlog2log).view(
            -1, num_logical_experts // n_nodes)
        phy2mlog, phyrank, mlogcnt = _ExpertReplicator.replicate(
            tokens_per_mlog, n_phy // n_nodes)

        # ---- Stage 3: pack physical experts onto GPUs ----
        gpus_per_node = n_gpus // n_nodes
        num_mlog = num_logical_experts // n_nodes
        phy_per_node = n_phy // n_nodes

        tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
        gpack_index, grank_in_pack = _LptPacker.pack(tokens_per_phy,
                                                      gpus_per_node)
        phy2pphy = gpack_index * phy_per_gpu + grank_in_pack

        # ---- Joint refinement: adjust replication based on GPU loads ----
        # Compute actual per-GPU loads from the current packing.
        gpu_load = torch.zeros(tokens_per_phy.shape[0], gpus_per_node,
                                dtype=tokens_per_phy.dtype,
                                device=tokens_per_phy.device)
        gpu_load.scatter_add_(1, gpack_index, tokens_per_phy)
        mean_load = gpu_load.mean(-1, keepdim=True)
        hot_rows = (gpu_load.max(-1, keepdim=True).values >
                   mean_load * (1.0 + 1e-3)).squeeze(-1)
        if hot_rows.any():
            arange_rows = torch.arange(tokens_per_phy.shape[0],
                                       device=weight.device)
            hot_gpu = gpu_load.argmax(-1)
            cold_gpu = gpu_load.argmin(-1)
            # contribution of each physical expert to its GPU load
            # pick hottest physical expert in hot GPU, coldest in cold GPU
            hot_mask = gpack_index == hot_gpu.unsqueeze(-1)
            cold_mask = gpack_index == cold_gpu.unsqueeze(-1)
            tpp = torch.where(hot_mask, tokens_per_phy,
                              torch.full_like(tokens_per_phy, -1.0))
            hot_phy = tpp.argmax(-1)
            tpc = torch.where(cold_mask, tokens_per_phy,
                              torch.full_like(tokens_per_phy, float("inf")))
            cold_phy = tpc.argmin(-1)
            hot_mlog = phy2mlog.gather(-1, hot_phy.unsqueeze(-1)).squeeze(-1)
            cold_mlog = phy2mlog.gather(-1,
                                        cold_phy.unsqueeze(-1)).squeeze(-1)
            hot_cnt = mlogcnt[arange_rows, hot_mlog]
            cold_cnt = mlogcnt[arange_rows, cold_mlog]
            # valid move: hot expert has >1 replica, cold expert gains one,
            # and the move reduces imbalance
            hot_contrib = tokens_per_phy[arange_rows, hot_phy]
            new_hot = hot_contrib * (hot_cnt - 1) / hot_cnt
            new_cold = tokens_per_phy[arange_rows, cold_phy] * \
                cold_cnt / (cold_cnt + 1)
            delta_hot = (gpu_load[arange_rows, hot_gpu] - hot_contrib +
                         new_hot) - mean_load.squeeze(-1)
            delta_cold = (gpu_load[arange_rows, cold_gpu] + new_cold) - \
                mean_load.squeeze(-1)
            do_move = (hot_rows & (hot_cnt > 1) &
                       ((gpu_load[arange_rows, hot_gpu] -
                         gpu_load[arange_rows, cold_gpu]) >
                        (hot_contrib - new_hot) + new_cold))
            # Mutate replication counts before packing (never ranks after)
            mlogcnt[arange_rows[do_move], hot_mlog[do_move]] -= 1
            mlogcnt[arange_rows[do_move], cold_mlog[do_move]] += 1
            # Rebuild phy2mlog and rank from the new counts (vectorized)
            n_rows = tokens_per_mlog.shape[0]
            sorted_mlog = (-tokens_per_mlog / mlogcnt).argsort(-1)
            cum_slots = torch.arange(phy_per_node,
                                     device=weight.device).unsqueeze(0)
            # number of slots allocated to experts in sorted order
            cnt_sorted = mlogcnt.gather(-1, sorted_mlog)
            boundaries = cnt_sorted.cumsum(-1)
            # slot j belongs to the first expert whose cumulative count > j
            slot_owner = (cum_slots.expand(n_rows, -1) < boundaries).float(
            ).argmax(-1)
            phy2mlog = sorted_mlog.gather(-1, slot_owner)
            # rank within the expert
            prev_bnd = boundaries.gather(
                -1, (slot_owner - 1).clamp(min=0)) - cnt_sorted.gather(
                    -1, slot_owner)
            start = torch.where(slot_owner > 0, prev_bnd,
                                torch.zeros_like(prev_bnd))
            phyrank = cum_slots.squeeze(0).expand(n_rows, -1).gather(
                -1, slot_owner) - start
            # Re-pack with updated per-replica loads
            tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
            gpack_index, grank_in_pack = _LptPacker.pack(tokens_per_phy,
                                                          gpus_per_node)
            phy2pphy = gpack_index * phy_per_gpu + grank_in_pack
        pphy2phy = _inverse_perm(phy2pphy)

        # ---- Assembly ----
        pphy2mlog = phy2mlog.gather(-1, pphy2phy)
        pphy2mlog = (pphy2mlog.view(num_layers, n_nodes, -1) +
                     torch.arange(0, num_logical_experts,
                                  num_logical_experts // n_nodes,
                                  device=weight.device).view(
                                      1, -1, 1)).flatten(-2)
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
    if num_groups % num_nodes == 0:
        pipeline = RebalancePipeline(num_replicas, num_groups, num_nodes,
                                     num_gpus)
    else:
        # global policy: single node, single group
        pipeline = RebalancePipeline(num_replicas, 1, 1, num_gpus)
    phy2log, phyrank, logcnt = pipeline.run(weight)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    scatter_idx = (phy2log * maxlogcnt + phyrank).view(num_layers, -1)
    values = torch.arange(num_replicas, dtype=torch.int64,
                          device=log2phy.device).unsqueeze(0).expand(
                              num_layers, -1)
    log2phy.view(num_layers, -1).scatter_(-1, scatter_idx, values)
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END

__all__ = ["rebalance_experts"]