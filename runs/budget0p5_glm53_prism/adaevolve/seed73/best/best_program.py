GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Bisection on the KVPR threshold T with FFD bin-packing feasibility.

    A GPU holding subset S has KVPR <= T iff sum(w_i) <= T*(80 - sum(s_i)),
    i.e. sum(w_i + T*s_i) <= 80*T. So for a candidate T each model gets a
    virtual size v_i = w_i + T*s_i and feasibility is packing all items into
    gpu_num bins of capacity 80*T (first-fit-decreasing, with best-fit retry).
    Feasibility is monotone in T, so bisection finds the minimal feasible T.
    The packing at the best feasible T is then polished by local search
    (moves/swaps) to lower the true max KVPR further.
    """

    def kvpr(load, free):
        return load / free if free > 0 else float('inf')

    def max_kvpr(loads, frees):
        best = 0.0
        for g in range(gpu_num):
            if frees[g] < GPU_MEM_SIZE:
                best = max(best, kvpr(loads[g], frees[g]))
        return best

    items = [(m.req_rate / m.slo, m.model_size, m) for m in models]

    def try_pack(T):
        """FFD (then best-fit) packing of virtual sizes w+T*s into bins of
        capacity 80*T. Returns placement dict or None if infeasible."""
        cap = GPU_MEM_SIZE * T
        if cap <= 0:
            return None
        for order_key in (lambda it: -(it[0] + T * it[1]),   # FFD
                          lambda it: (it[1], -it[0])):        # best-fit retry
            bins = [[] for _ in range(gpu_num)]
            rem = [cap] * gpu_num
            ok = True
            for w, s, m in sorted(items, key=order_key):
                v = w + T * s
                if v > cap + 1e-12:
                    ok = False
                    break
                # first fit
                placed = False
                for b in range(gpu_num):
                    if rem[b] >= v - 1e-9:
                        rem[b] -= v
                        bins[b].append(m)
                        placed = True
                        break
                if not placed:
                    # best fit (tightest bin)
                    tight = min(rem, default=None)
                    if tight is not None and tight >= v - 1e-9:
                        b = rem.index(tight)
                        rem[b] -= v
                        bins[b].append(m)
                        placed = True
                    else:
                        ok = False
                        break
            if ok:
                return {g: bins[g] for g in range(gpu_num)}
        return None

    def kvpr_of(placement):
        loads = [0.0] * gpu_num
        frees = [GPU_MEM_SIZE] * gpu_num
        for g in range(gpu_num):
            for m in placement[g]:
                loads[g] += m.req_rate / m.slo
                frees[g] -= m.model_size
        return max_kvpr(loads, frees)

    # --- Bisection on T ---
    lo, hi = 0.0, 0.0
    best_pack = None
    # find an upper bound by geometric growth
    T = 1.0
    for _ in range(60):
        p = try_pack(T)
        if p is not None:
            hi = T
            best_pack = p
            break
        T *= 2.0
    else:
        hi = None

    if hi is not None:
        lo = 0.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            p = try_pack(mid)
            if p is not None:
                hi = mid
                best_pack = p
            else:
                lo = mid
        placement = best_pack
    else:
        # packing failed entirely; fall back to simple greedy
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        for w, s, m in sorted(items, key=lambda it: -it[0]):
            for g in range(gpu_num):
                if free_mem[g] - s > 0:
                    placement[g].append(m)
                    free_mem[g] -= s
                    break
            else:
                for g in range(gpu_num):
                    if free_mem[g] >= s:
                        placement[g].append(m)
                        free_mem[g] -= s
                        break
                else:
                    raise ValueError("Cannot place model")

    # --- Local-search polish (moves, then swaps) ---
    loads = [0.0] * gpu_num
    frees = [GPU_MEM_SIZE] * gpu_num
    for g in range(gpu_num):
        for m in placement[g]:
            loads[g] += m.req_rate / m.slo
            frees[g] -= m.model_size

    improved = True
    while improved:
        improved = False
        cur_max = max_kvpr(loads, frees)
        for src in range(gpu_num):
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or frees[dst] - m.model_size <= 0:
                        continue
                    loads[src] -= w; frees[src] += m.model_size
                    loads[dst] += w; frees[dst] -= m.model_size
                    if max_kvpr(loads, frees) < cur_max - 1e-12:
                        placement[src].remove(m)
                        placement[dst].append(m)
                        improved = True
                        break
                    loads[src] += w; frees[src] -= m.model_size
                    loads[dst] -= w; frees[dst] += m.model_size
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # swaps
        hot = max((g for g in range(gpu_num) if placement[g]),
                  key=lambda g: kvpr(loads[g], frees[g]), default=None)
        if hot is None:
            break
        for m in list(placement[hot]):
            w1 = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == hot:
                    continue
                for m2 in list(placement[dst]):
                    w2 = m2.req_rate / m2.slo
                    if (frees[hot] + m.model_size - m2.model_size <= 0 or
                            frees[dst] + m2.model_size - m.model_size <= 0):
                        continue
                    loads[hot] += w2 - w1
                    frees[hot] += m.model_size - m2.model_size
                    loads[dst] += w1 - w2
                    frees[dst] += m2.model_size - m.model_size
                    if max_kvpr(loads, frees) < cur_max - 1e-12:
                        placement[hot].remove(m); placement[hot].append(m2)
                        placement[dst].remove(m2); placement[dst].append(m)
                        improved = True
                        break
                    loads[hot] -= w2 - w1
                    frees[hot] -= m.model_size - m2.model_size
                    loads[dst] -= w1 - w2
                    frees[dst] -= m2.model_size - m.model_size
                if improved:
                    break
            if improved:
                break

    return placement

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
