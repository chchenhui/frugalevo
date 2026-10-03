GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

import numpy as np


def _kvpr(w, mem):
    """KVPR = weighted req rate / free memory."""
    return w / (GPU_MEM_SIZE - mem) if mem < GPU_MEM_SIZE else float('inf')


def _greedy(gpu_num, models, key_fn):
    """One greedy construction: place models in key_fn order, each on the GPU
    minimizing the resulting KVPR (tie-break: more free memory)."""
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [0.0] * gpu_num
    weighted_req_rate = [0.0] * gpu_num
    for model in sorted(models, key=key_fn):
        best_idx = None
        best_key = (float('inf'), float('-inf'))
        for gpu_id in range(gpu_num):
            free = shared_kv[gpu_id] + model.model_size
            if free < GPU_MEM_SIZE:
                ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / (GPU_MEM_SIZE - free)
                k = (ratio, -free)
                if k < best_key:
                    best_key = k
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(f"Cannot place model of size {model.model_size} GB")
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] += model.model_size
    return placement, shared_kv, weighted_req_rate


_SEED_KEYS = [
    lambda m: (-(m.req_rate / m.slo),),                      # load desc (incumbent)
    lambda m: (-m.model_size,),                              # size desc
    lambda m: (-(m.req_rate / m.slo) / m.model_size,),       # load per GB desc
    lambda m: (-(m.req_rate / m.slo) * m.model_size,),       # load*size desc
    lambda m: (m.model_size, -(m.req_rate / m.slo)),         # size asc then load desc
    lambda m: (m.req_rate / m.slo,) + (m.model_size,),       # load asc (anti-greedy)
    lambda m: (-m.req_rate / m.slo, -m.model_size),          # load desc then size desc
    lambda m: (m.slo,),                                      # slo asc (tight SLO first)
    lambda m: (-m.slo, -m.model_size),                       # slo desc then size desc
    lambda m: (-m.model_size, m.req_rate / m.slo),           # size desc then load asc
]


def _max_kvpr(weighted_req_rate, shared_kv, gpu_num):
    """Global max KVPR helper."""
    return max(_kvpr(weighted_req_rate[g], shared_kv[g]) for g in range(gpu_num))


def _snapshot(placement):
    """Copy of the placement so kicks can never alias the recorded best."""
    return {g: list(ms) for g, ms in placement.items()}


def _feasible_at_T(gpu_num, models, T):
    """Check if a placement with max KVPR <= T exists, via budget greedy.

    KVPR_g <= T  <=>  sum(w_i) + T*sum(m_i) <= T*GPU_MEM_SIZE, so each model
    consumes (w_i + T*m_i) of a per-GPU budget T*GPU_MEM_SIZE; the hard
    memory cap sum(m_i) < GPU_MEM_SIZE is enforced separately. Models are
    assigned LPT-style (largest cost first) to any GPU that fits.
    Returns (placement, shared_kv, weighted_req_rate) or None.
    """
    placement = {g: [] for g in range(gpu_num)}
    budget = [T * GPU_MEM_SIZE] * gpu_num
    mem = [0.0] * gpu_num
    for model in sorted(models, key=lambda m: (-(m.req_rate / m.slo + T * m.model_size),)):
        cost = model.req_rate / model.slo + T * model.model_size
        for g in range(gpu_num):
            if cost <= budget[g] and mem[g] + model.model_size < GPU_MEM_SIZE:
                placement[g].append(model)
                budget[g] -= cost
                mem[g] += model.model_size
                break
        else:
            return None
    shared_kv = mem
    weighted_req_rate = [(T * GPU_MEM_SIZE - b) - T * shared_kv[g]
                         for g, b in enumerate(budget)]
    return placement, shared_kv, weighted_req_rate


def compute_model_placement(gpu_num, models):
    """
    Threshold-bisection: build an incumbent from the best greedy seed +
    local search, then binary-search the min-max target KVPR T. For each
    candidate T a budget-greedy feasibility check attempts a placement with
    all KVPR <= T; feasible placements are polished by local search and
    kept only if strictly better than the incumbent (never regress).
    """
    # Incumbent: best of the deterministic seeds, refined by local search.
    # Every refined seed basin is kept as an annealing start point.
    best_placement = None
    best_max = float('inf')
    seed_pool = []
    for key_fn in _SEED_KEYS:
        placement, shared_kv, weighted_req_rate = _greedy(gpu_num, models, key_fn)
        _local_search(gpu_num, placement, shared_kv, weighted_req_rate)
        cur = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
        seed_pool.append((cur, _snapshot(placement)))
        if cur < best_max:
            best_max = cur
            best_placement = _snapshot(placement)

    # Lower bound on any achievable max KVPR (aggregate load / total free mem).
    total_w = sum(m.req_rate / m.slo for m in models)
    total_m = sum(m.model_size for m in models)
    lo = total_w / (gpu_num * GPU_MEM_SIZE - total_m) if total_m < gpu_num * GPU_MEM_SIZE else best_max
    hi = best_max

    for _ in range(40):
        if hi - lo <= 1e-9 * max(1.0, hi):
            break
        mid = 0.5 * (lo + hi)
        if mid >= hi:
            break
        res = _feasible_at_T(gpu_num, models, mid)
        if res is None:
            lo = mid
            continue
        placement, shared_kv, weighted_req_rate = res
        _local_search(gpu_num, placement, shared_kv, weighted_req_rate)
        new = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
        if new < best_max - 1e-12:
            best_max = new
            best_placement = _snapshot(placement)
            hi = min(mid, new)
        else:
            hi = mid
    # Multi-start stochastic-escape annealing: each refined seed basin is a
    # start point; accept-worse moves plus reheating on stagnation let the
    # search cross KVPR barriers strict descent cannot. Bounded (200 iters
    # per start), deterministic, best snapshot restored; the incumbent is
    # the guaranteed fallback.
    for start_max, start_snap in seed_pool:
        cand_max, cand_snap = _anneal(gpu_num, start_snap, start_max)
        if cand_max < best_max - 1e-12:
            best_max = cand_max
            best_placement = cand_snap
    if best_max < seed_pool[0][0] or best_placement is None:
        pass  # best_placement always holds the global best snapshot
    # Final polish of the global best.
    shared_kv = [sum(m.model_size for m in best_placement[g]) for g in range(gpu_num)]
    weighted_req_rate = [sum(m.req_rate / m.slo for m in best_placement[g])
                         for g in range(gpu_num)]
    _local_search(gpu_num, best_placement, shared_kv, weighted_req_rate)
    best_max = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
    return best_placement


def _anneal(gpu_num, start_snap, start_max, iters=200, seed=0):
    """Reheated simulated annealing from one start basin.

    Each iteration picks a random feasible move or swap, evaluates the new
    max KVPR in O(g), and accepts if better or with probability
    exp(-delta/T). After 60 stagnant iterations the state resets to the
    best-seen and the temperature is doubled (reheat), enabling escape.
    Returns (max_kvpr, snapshot); never worse than the start basin.
    """
    rng = np.random.RandomState(seed)
    placement = _snapshot(start_snap)
    shared_kv = [0.0] * gpu_num
    weighted_req_rate = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in placement[g]:
            shared_kv[g] += m.model_size
            weighted_req_rate[g] += m.req_rate / m.slo
    cur = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
    best_max, best_snap = cur, _snapshot(placement)
    T_ann = 0.05 * max(start_max, 1e-9)
    stagnant = 0
    for _ in range(iters):
        moves = [('move', g1, g2, model)
                 for g1 in range(gpu_num) for model in placement[g1]
                 for g2 in range(gpu_num)
                 if g2 != g1 and shared_kv[g2] + model.model_size < GPU_MEM_SIZE]
        swaps = [('swap', g1, g2, a, b)
                 for g1 in range(gpu_num) for g2 in range(g1 + 1, gpu_num)
                 for a in placement[g1] for b in placement[g2]
                 if (shared_kv[g1] - a.model_size + b.model_size < GPU_MEM_SIZE and
                     shared_kv[g2] - b.model_size + a.model_size < GPU_MEM_SIZE)]
        if swaps and (not moves or rng.random() < 0.5):
            _, g1, g2, a, b = swaps[rng.randint(len(swaps))]
            ra, rb = a.req_rate / a.slo, b.req_rate / b.slo
            w1 = weighted_req_rate[g1] - ra + rb
            w2 = weighted_req_rate[g2] - rb + ra
            m1 = shared_kv[g1] - a.model_size + b.model_size
            m2 = shared_kv[g2] - b.model_size + a.model_size
            new = max(_kvpr(w1, m1), _kvpr(w2, m2),
                      max((_kvpr(weighted_req_rate[g], shared_kv[g])
                           for g in range(gpu_num) if g not in (g1, g2)),
                          default=0.0))
            apply = lambda: (placement[g1].remove(a), placement[g2].remove(b),
                             placement[g1].append(b), placement[g2].append(a))
        elif moves:
            _, g1, g2, model = moves[rng.randint(len(moves))]
            w1 = weighted_req_rate[g1] - model.req_rate / model.slo
            w2 = weighted_req_rate[g2] + model.req_rate / model.slo
            m1 = shared_kv[g1] - model.model_size
            m2 = shared_kv[g2] + model.model_size
            new = max(_kvpr(w1, m1), _kvpr(w2, m2),
                      max((_kvpr(weighted_req_rate[g], shared_kv[g])
                           for g in range(gpu_num) if g not in (g1, g2)),
                          default=0.0))
            apply = lambda: (placement[g1].remove(model), placement[g2].append(model))
        else:
            break
        delta = new - cur
        if delta < -1e-12 or rng.random() < np.exp(-delta / T_ann):
            apply()
            weighted_req_rate[g1], weighted_req_rate[g2] = w1, w2
            shared_kv[g1], shared_kv[g2] = m1, m2
            cur = new
            if cur < best_max - 1e-12:
                best_max, best_snap = cur, _snapshot(placement)
                stagnant = 0
            else:
                stagnant += 1
        else:
            stagnant += 1
        if stagnant >= 60:
            # Reheat: reset to best-seen state, raise temperature.
            placement = _snapshot(best_snap)
            for g in range(gpu_num):
                shared_kv[g] = sum(m.model_size for m in placement[g])
                weighted_req_rate[g] = sum(m.req_rate / m.slo for m in placement[g])
            cur = best_max
            T_ann *= 2.0
            stagnant = 0
    return best_max, best_snap


def _local_search(gpu_num, placement, shared_kv, weighted_req_rate):
    """Hill climber: move/swap models while strictly reducing max KVPR."""

    improved = True
    while improved:
        improved = False
        cur = max(_kvpr(weighted_req_rate[g], shared_kv[g]) for g in range(gpu_num))
        for g1 in range(gpu_num):
            for model in list(placement[g1]):
                # try moving model to another GPU
                for g2 in range(gpu_num):
                    if g2 == g1:
                        continue
                    if shared_kv[g2] + model.model_size < GPU_MEM_SIZE:
                        w1 = weighted_req_rate[g1] - model.req_rate / model.slo
                        w2 = weighted_req_rate[g2] + model.req_rate / model.slo
                        m1 = shared_kv[g1] - model.model_size
                        m2 = shared_kv[g2] + model.model_size
                        new_max = max(_kvpr(w1, m1), _kvpr(w2, m2),
                                      max(_kvpr(weighted_req_rate[g], shared_kv[g])
                                          for g in range(gpu_num) if g not in (g1, g2)))
                        if new_max < cur - 1e-12:
                            placement[g1].remove(model)
                            placement[g2].append(model)
                            weighted_req_rate[g1], weighted_req_rate[g2] = w1, w2
                            shared_kv[g1], shared_kv[g2] = m1, m2
                            improved = True
                            cur = new_max
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # try swaps between GPU pairs
            for g1 in range(gpu_num):
                for g2 in range(g1 + 1, gpu_num):
                    for a in placement[g1]:
                        for b in placement[g2]:
                            if (shared_kv[g1] - a.model_size + b.model_size < GPU_MEM_SIZE and
                                    shared_kv[g2] - b.model_size + a.model_size < GPU_MEM_SIZE):
                                ra, rb = a.req_rate / a.slo, b.req_rate / b.slo
                                w1 = weighted_req_rate[g1] - ra + rb
                                w2 = weighted_req_rate[g2] - rb + ra
                                m1 = shared_kv[g1] - a.model_size + b.model_size
                                m2 = shared_kv[g2] - b.model_size + a.model_size
                                new_max = max(_kvpr(w1, m1), _kvpr(w2, m2),
                                              max(_kvpr(weighted_req_rate[g], shared_kv[g])
                                                  for g in range(gpu_num) if g not in (g1, g2)))
                                if new_max < cur - 1e-12:
                                    placement[g1].remove(a)
                                    placement[g2].remove(b)
                                    placement[g1].append(b)
                                    placement[g2].append(a)
                                    weighted_req_rate[g1], weighted_req_rate[g2] = w1, w2
                                    shared_kv[g1], shared_kv[g2] = m1, m2
                                    improved = True
                                    cur = new_max
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return placement  # final state of hill climber (unused by caller)

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm

    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):

        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr


    print(f"Max KVPR: {avg_kvpr:.3f}")
