GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy placement trying several model orderings; each model is assigned to
    the GPU minimizing the KVPR after placement, subject to memory capacity.
    The ordering yielding the lowest max KVPR is returned. Guards against
    zero remaining memory (division by zero) by treating it as infinite KVPR.
    """

    def try_order(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            r = model.req_rate / model.slo
            best, best_kv = None, float('inf')
            for g in range(gpu_num):
                rem = free[g] - model.model_size
                if rem > 0:
                    kvpr = (load[g] + r) / rem
                elif rem == 0:
                    # Model exactly fills remaining memory; feasible only if no load
                    if load[g] + r <= 0:
                        kvpr = 0.0
                    else:
                        continue
                else:
                    continue
                if kvpr < best_kv:
                    best_kv, best = kvpr, g
            if best is None:
                return None
            placement[best].append(model)
            load[best] += r
            free[best] -= model.model_size
        worst = max((load[g] / free[g]) if free[g] > 0 else
                    (0.0 if load[g] == 0 else float('inf'))
                    for g in range(gpu_num))
        return placement, worst

    # Complementary-pairing constructive heuristic with reserve headroom:
    # models sorted by rate density (r/s) descending; a GPU is forbidden if
    # its post-placement free memory drops below the largest remaining model
    # size while another feasible GPU exists. Near-ties (within 5%) prefer
    # the GPU with the most free memory, preserving headroom for big models.
    def try_reserve(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [float(GPU_MEM_SIZE)] * gpu_num
        load = [0.0] * gpu_num
        for i, model in enumerate(order):
            r = model.req_rate / model.slo
            remaining_max = max((m.model_size for m in order[i + 1:]), default=0.0)
            cands = []
            for g in range(gpu_num):
                rem = free[g] - model.model_size
                if rem <= 0:
                    continue
                if rem < remaining_max and any(
                        free[h] - model.model_size >= remaining_max
                        for h in range(gpu_num) if h != g):
                    continue  # reserve violated, feasible alternative exists
                cands.append(((load[g] + r) / rem, -free[g], g))
            if not cands:
                return None
            cands.sort()
            best_kv = cands[0][0]
            best_g = cands[0][2]
            for kv, negfree, g in cands:
                if kv <= best_kv * 1.05 + 1e-12:
                    best_g = g  # cands sorted by -free: first near-tie has max free
                    break
            placement[best_g].append(model)
            load[best_g] += r
            free[best_g] -= model.model_size
        worst = max((load[g] / free[g]) if free[g] > 0 else
                    (0.0 if load[g] == 0 else float('inf'))
                    for g in range(gpu_num))
        return placement, worst

    orderings = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size / (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo * m.model_size, reverse=True),
    ]
    best = try_reserve(orderings[0])
    for order in orderings:
        for res in (try_order(order), try_reserve(order)):
            if res is not None and (best is None or res[1] < best[1]):
                best = res
    if best is None:
        raise ValueError("Unable to place all models within GPU memory.")
    placement = best[0]

    def kvpr_of(g):
        mem = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
        if mem == 0:
            return 0.0 if not placement[g] else float('inf')
        return sum(m.req_rate / m.slo for m in placement[g]) / mem

    # Exact minimax KVPR via bisection on threshold T with MILP feasibility
    # (scipy.optimize.milp). The greedy placement above serves as the
    # incumbent upper bound T_hi and as a valid fallback. Binary variables
    # x[m,g] assign each model to one GPU subject to:
    #   - each model assigned exactly once,
    #   - per-GPU memory: sum_m s_m x[m,g] <= GPU_MEM_SIZE,
    #   - linearized KVPR: sum_m (r_m + T*s_m) x[m,g] <= T*GPU_MEM_SIZE.
    # Bisection tightens T_hi; any solver failure returns the greedy incumbent.
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds

    n = len(models)
    if n == 0 or gpu_num <= 0:
        return placement

    r = np.array([m.req_rate / m.slo for m in models], dtype=float)
    s = np.array([m.model_size for m in models], dtype=float)
    gpu_of = np.zeros(n, dtype=int)
    for g in range(gpu_num):
        for m in placement[g]:
            gpu_of[[i for i, mm in enumerate(models) if mm is m][0]] = g

    def max_kvpr_assign(assign):
        load = np.zeros(gpu_num)
        mem = np.full(gpu_num, float(GPU_MEM_SIZE))
        np.add.at(load, assign, r)
        np.subtract.at(mem, assign, s)
        return float(np.max(np.where(mem > 0, load / np.maximum(mem, 1e-12),
                                     np.where(load == 0, 0.0, np.inf))))

    T_hi = max_kvpr_assign(gpu_of)
    T_lo = max(0.0, float(r.sum()) / max(gpu_num * GPU_MEM_SIZE - float(s.sum()), 1e-12))
    best_assign = gpu_of.copy()
    if T_hi > T_lo + 1e-9:
        M = n * gpu_num
        A_eq = np.zeros((n, M))
        for i in range(n):
            A_eq[i, i * gpu_num:(i + 1) * gpu_num] = 1.0
        A_mem = np.zeros((gpu_num, M))
        for i in range(n):
            for g in range(gpu_num):
                A_mem[g, i * gpu_num + g] = s[i]
        cons_eq = LinearConstraint(A_eq, np.ones(n), np.ones(n))
        cons_mem = LinearConstraint(A_mem, -np.inf, np.full(gpu_num, float(GPU_MEM_SIZE)))
        bounds = Bounds(np.zeros(M), np.ones(M))
        integrality = np.ones(M)
        for _ in range(15):
            T = 0.5 * (T_lo + T_hi)
            w = r + T * s
            A_kv = np.zeros((gpu_num, M))
            for i in range(n):
                for g in range(gpu_num):
                    A_kv[g, i * gpu_num + g] = w[i]
            cons_kv = LinearConstraint(A_kv, -np.inf, np.full(gpu_num, T * GPU_MEM_SIZE))
            try:
                res = milp(c=np.zeros(M), constraints=[cons_eq, cons_mem, cons_kv],
                           integrality=integrality, bounds=bounds,
                           options={"time_limit": 0.5, "mip_rel_gap": 0.0})
            except Exception:
                break
            if res.status == 0 and res.x is not None:
                assign = np.array([g for i in range(n) for g in range(gpu_num)
                                   if res.x[i * gpu_num + g] > 0.5], dtype=int)
                if len(assign) == n:
                    T_hi = T
                    best_assign = assign
                else:
                    T_lo = T
            else:
                T_lo = T
            if T_hi - T_lo < 1e-6:
                break
        if max_kvpr_assign(best_assign) < max_kvpr_assign(gpu_of) - 1e-12:
            placement = {g: [] for g in range(gpu_num)}
            for i, g in enumerate(best_assign):
                placement[int(g)].append(models[i])
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
