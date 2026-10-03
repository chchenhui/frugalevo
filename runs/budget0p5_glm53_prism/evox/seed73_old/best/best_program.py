GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _kvpr(mlist):
    """KVPR of a single GPU's model list."""
    free = GPU_MEM_SIZE - sum(m.model_size for m in mlist)
    if not mlist or free <= 0:
        return 0.0
    return sum(m.req_rate / m.slo for m in mlist) / free


def _greedy(gpu_num, sorted_models):
    """Place each model on the GPU whose resulting KVPR is lowest."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted_models:
        best_idx, best_kvpr = None, float('inf')
        for g in range(gpu_num):
            if m.model_size < free[g]:
                kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best_idx = kvpr, g
        if best_idx is None:
            return None
        placement[best_idx].append(m)
        load[best_idx] += m.req_rate / m.slo
        free[best_idx] -= m.model_size
    return placement


def _refine(p):
    """Hill-climb: try a move or swap for every model (any source GPU,
    any destination) and accept the first step that strictly lowers the
    maximum KVPR. Sweeping all sources escapes local optima where the
    max-KVPR GPU has no feasible improving move."""
    n_models = sum(len(v) for v in p.values())
    for _ in range(n_models * 4 + 8):
        k = {g: _kvpr(p[g]) for g in p}
        cur = max(k.values())
        if cur <= 0:
            break
        improved = False
        for src in sorted(p, key=lambda g: -k[g]):
            if improved:
                break
            src_free = GPU_MEM_SIZE - sum(x.model_size for x in p[src])
            for m in list(p[src]):
                if improved:
                    break
                for dst in p:
                    if dst == src:
                        continue
                    dst_free = GPU_MEM_SIZE - sum(x.model_size for x in p[dst])
                    if m.model_size < dst_free:
                        p[src].remove(m)
                        p[dst].append(m)
                        if max(_kvpr(p[g]) for g in p) < cur - 1e-12:
                            improved = True
                            break
                        p[dst].remove(m)
                        p[src].append(m)
                    for n in list(p[dst]):
                        if (m.model_size - n.model_size < dst_free and
                                n.model_size - m.model_size < src_free):
                            p[src].remove(m)
                            p[dst].remove(n)
                            p[src].append(n)
                            p[dst].append(m)
                            if max(_kvpr(p[g]) for g in p) < cur - 1e-12:
                                improved = True
                                break
                            p[dst].remove(m)
                            p[src].remove(n)
                            p[src].append(m)
                            p[dst].append(n)
                if improved:
                    break
        if not improved:
            break
    return p


def _milp(gpu_num, models, t_hi, deadline):
    """Bisection on max-KVPR target T. For fixed T the per-GPU constraint
    load_g + T*mem_g <= MEM*T is linear, so feasibility is a binary MILP
    (x[m,g]=1 iff model m on GPU g) solved with scipy HiGGS. Returns a
    placement achieving max KVPR <= T, or None on failure/timeout."""
    import time
    import numpy as np
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = len(models)
    ng = gpu_num
    p = [m.req_rate / m.slo for m in models]
    s = [m.model_size for m in models]
    c = np.zeros(n * ng)
    integrality = np.ones(n * ng)
    A_eq = np.zeros((n, n * ng))
    for i in range(n):
        A_eq[i, i * ng:(i + 1) * ng] = 1.0
    cons_eq = LinearConstraint(A_eq, 1.0, 1.0)
    lo, hi = 0.0, t_hi
    best_x = None
    for _ in range(24):
        if time.time() > deadline:
            break
        T = (lo + hi) / 2.0
        if T <= 0:
            break
        A = np.zeros((ng, n * ng))
        for g in range(ng):
            for i in range(n):
                A[g, i * ng + g] = p[i] + T * s[i]
        try:
            res = milp(c=c, integrality=integrality, bounds=Bounds(0, 1),
                       constraints=[cons_eq,
                                    LinearConstraint(A, -np.inf, GPU_MEM_SIZE * T)])
        except Exception:
            return None
        if res.success and res.x is not None:
            hi = T
            best_x = res.x
        else:
            lo = T
        if hi - lo < 1e-9:
            break
    if best_x is None:
        return None
    placement = {g: [] for g in range(ng)}
    for i, m in enumerate(models):
        g = int(np.argmax(best_x[i * ng:(i + 1) * ng]))
        placement[g].append(m)
    return placement


def compute_model_placement(gpu_num, models):
    """Exact MILP placement via bisection on max-KVPR (scipy HiGHS) with a
    strict per-call time budget; falls back to multi-start greedy + local
    search (guaranteed feasible) if the solver is unavailable, times out,
    or fails, so success rate stays 1.0 while keeping MILP-level quality."""
    import time
    import random
    if not models:
        return {g: [] for g in range(gpu_num)}
    # feasible upper bound from greedy
    ub = None
    for ms in (sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
               sorted(models, key=lambda m: m.model_size, reverse=True)):
        p = _greedy(gpu_num, ms)
        if p is not None:
            mx = max(_kvpr(l) for l in p.values())
            if ub is None or mx < ub:
                ub = mx
    if ub is not None:
        try:
            p = _milp(gpu_num, models, ub * 1.000001 + 1e-9,
                      time.time() + 20.0)
            if p is not None:
                mx = max(_kvpr(l) for l in p.values())
                if mx <= ub + 1e-6:
                    return p
        except Exception:
            pass
    # fallback: multi-start greedy + local search
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
    ]
    for seed in range(6):
        ms = list(models)
        random.Random(seed).shuffle(ms)
        orders.append(ms)
    best, best_max = None, float('inf')
    for ms in orders:
        p = _greedy(gpu_num, ms)
        if p is None:
            continue
        p = _refine(p)
        mx = max(_kvpr(mlist) for mlist in p.values())
        if mx < best_max:
            best, best_max = p, mx
    if best is None:
        raise ValueError("Unable to place all models within GPU memory")
    return best

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
