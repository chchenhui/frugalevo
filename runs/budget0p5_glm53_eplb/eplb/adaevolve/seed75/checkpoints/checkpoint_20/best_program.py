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


def _waterfill_counts(w: torch.Tensor,
                       num_phy: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bisection water-filling over the FULL layer (global, not per-node):
    find the minimal peak per-replica load T with sum(ceil(w/T)) <= num_phy,
    then pad leftover slots greedily by max w/cnt. Returns (logcnt, T).
    """
    lo = w.sum(-1) / num_phy
    hi = w.max(-1).values.clone()
    for _ in range(30):
        mid = (lo + hi) / 2
        cnt = torch.ceil(w / mid.unsqueeze(-1)).clamp(min=1)
        if (cnt.sum(-1) <= num_phy).all():
            hi = mid
        else:
            lo = mid
    logcnt = torch.ceil(w / hi.unsqueeze(-1)).clamp(min=1).long()
    need = num_phy - logcnt.sum(-1)
    rows = torch.arange(w.size(0), dtype=torch.int64)
    while (need > 0).any():
        score = w / logcnt
        idx = score.argmax(-1)
        sel = need > 0
        logcnt[rows[sel], idx[sel]] += 1
        need[sel] -= 1
    return logcnt, hi


def _pack_layer(wl, cnt, num_gpus, epg):
    """
    One global LPT packing of a single layer's physical experts onto GPUs:
    each logical expert contributes `cnt[e]` items of load wl[e]/cnt[e];
    every GPU holds exactly `epg` items. Followed by a swap-refinement pass
    exchanging one item between the heaviest and lightest GPU while it
    reduces the heavier of the two. Returns (gpu_items, gpu_loads).
    """
    items = []
    for e in range(len(wl)):
        c = int(cnt[e])
        ld = wl[e] / c
        for _ in range(c):
            items.append((ld, e))
    items.sort(reverse=True)
    heap = [(0.0, g) for g in range(num_gpus)]
    heapq.heapify(heap)
    pc = [0] * num_gpus
    pw = [0.0] * num_gpus
    gi = [[] for _ in range(num_gpus)]
    for ld, e in items:
        while heap and pc[heap[0][1]] >= epg:
            heapq.heappop(heap)
        _, g = heapq.heappop(heap)
        gi[g].append((ld, e))
        pw[g] += ld
        pc[g] += 1
        if pc[g] < epg:
            heapq.heappush(heap, (pw[g], g))
    # Swap refinement between heaviest and lightest GPU.
    for _ in range(4 * num_gpus):
        h = max(range(num_gpus), key=pw.__getitem__)
        l = min(range(num_gpus), key=pw.__getitem__)
        if pw[h] - pw[l] <= 1e-12:
            break
        best = None
        best_val = pw[h]
        for wa, ea in gi[h]:
            for wb, eb in gi[l]:
                nh = pw[h] - wa + wb
                nl = pw[l] + wa - wb
                m = nh if nh > nl else nl
                if m < best_val - 1e-12:
                    best_val = m
                    best = (ea, wa, eb, wb, nh, nl)
        if best is None:
            break
        ea, wa, eb, wb, nh, nl = best
        gi[h].remove((wa, ea))
        gi[h].append((wb, eb))
        gi[l].remove((wb, eb))
        gi[l].append((wa, ea))
        pw[h], pw[l] = nh, nl
    return gi, pw


def _layout(w, logcnt, num_gpus, epg):
    """
    Pack every layer via _pack_layer and materialize the physical layout:
    physical slot p = gpu * epg + slot-in-gpu, with replica ranks assigned
    densely per logical expert. Returns (phy2log, phyrank, per-layer info).
    """
    L, E = w.shape
    P = num_gpus * epg
    phy2log = torch.zeros(L, P, dtype=torch.int64)
    phyrank = torch.zeros(L, P, dtype=torch.int64)
    infos = []
    wl = w.tolist()
    cl = logcnt.tolist()
    for r in range(L):
        gi, pw = _pack_layer(wl[r], cl[r], num_gpus, epg)
        placed = [0] * E
        for g in range(num_gpus):
            for s, (ld, e) in enumerate(gi[g]):
                p = g * epg + s
                phy2log[r, p] = e
                phyrank[r, p] = placed[e]
                placed[e] += 1
        infos.append((gi, pw))
    return phy2log, phyrank, infos


def rebalance_experts_global(
    weight: torch.Tensor,
    num_phy: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Flattened (non-hierarchical) balancer:
    (1) global water-filling replication counts per layer;
    (2) one global LPT packing of all physical experts onto GPUs with the
        exact per-GPU capacity num_phy/num_gpus, plus swap refinement;
    (3) iterative refinement: for layers whose peak GPU load exceeds the
        water-filling target T, shift one replica count from an expert on
        the lightest GPU (smallest load-increase penalty w/(c*(c-1))) to an
        expert on the peak GPU (largest load-reduction gain w/(c*(c+1))),
        re-pack, and keep the result only if the summed peak GPU load
        strictly improves. This directly minimizes max GPU load instead of
        the hierarchical proxy objective.
    """
    L, E = weight.shape
    assert num_phy >= E
    assert num_phy % num_gpus == 0
    epg = num_phy // num_gpus
    w = weight
    logcnt, target = _waterfill_counts(w, num_phy)
    wl = w.tolist()
    tl = target.tolist()
    best = None
    best_peak = None
    for _ in range(5):
        phy2log, phyrank, infos = _layout(w, logcnt, num_gpus, epg)
        peaks = [max(pw) for _, pw in infos]
        total = sum(peaks)
        if best_peak is None or total < best_peak - 1e-12:
            best_peak = total
            best = (phy2log, phyrank, logcnt.clone())
        else:
            break  # refinement no longer improving; keep best
        if all(peaks[r] <= tl[r] * 1.0001 for r in range(L)):
            break  # hit theoretical water-filling target
        # Propose one replica-count shift per unbalanced layer.
        cl = logcnt.tolist()
        changed = False
        for r in range(L):
            if peaks[r] <= tl[r] * 1.0001:
                continue
            gi, pw = infos[r]
            h = max(range(num_gpus), key=pw.__getitem__)
            l = min(range(num_gpus), key=pw.__getitem__)
            best_e, best_gain = None, 0.0
            for _, e in gi[h]:
                c = cl[r][e]
                gain = wl[r][e] / (c * (c + 1))
                if gain > best_gain:
                    best_gain, best_e = gain, e
            best_f, best_inc = None, float("inf")
            for _, f in gi[l]:
                c = cl[r][f]
                if c > 1:
                    inc = wl[r][f] / (c * (c - 1))
                    if inc < best_inc:
                        best_inc, best_f = inc, f
            if best_e is not None and best_f is not None and best_e != best_f:
                cl[r][best_e] += 1
                cl[r][best_f] -= 1
                changed = True
        if not changed:
            break
        logcnt = torch.tensor(cl, dtype=torch.int64)
    return best


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
    # Flattened global policy: joint replication + GPU packing with
    # iterative peak-load refinement (hierarchy removed to avoid
    # locally-optimal per-node decisions propagating imbalance).
    phy2log, phyrank, logcnt = rebalance_experts_global(
        weight, num_replicas, num_gpus)
    num_redundant_experts = num_replicas - num_logical_experts
    # Guard against out-of-bounds scatter: size the buffer by the actual
    # maximum replica count produced by the replication step.
    maxlogcnt = max(num_redundant_experts + 1, int(logcnt.max().item()))
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

