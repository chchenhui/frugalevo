GPU_MEM_SIZE = 80  # GB


# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use a greedy incumbent, then binary-search a globally coupled MILP KVPR bound."""
    eps = 1e-8
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")
    n = len(models)
    loads = [float(m.req_rate / m.slo) for m in models]
    sizes = [float(m.model_size) for m in models]
    if any(s >= GPU_MEM_SIZE - eps for s in sizes):
        raise ValueError("A model leaves no positive KV-cache memory on a GPU")

    order = sorted(range(n), key=lambda i: (-loads[i], -sizes[i], i))
    bins = [[] for _ in range(gpu_num)]
    rem = [float(GPU_MEM_SIZE)] * gpu_num
    load_sum = [0.0] * gpu_num
    for i in order:
        choices = [(load_sum[g] / rem[g], -rem[g], g)
                   for g in range(gpu_num) if rem[g] - sizes[i] > eps]
        if not choices:
            raise ValueError("Unable to find a feasible positive-residual-memory placement")
        g = min(choices)[2]
        bins[g].append(i)
        rem[g] -= sizes[i]
        load_sum[g] += loads[i]

    def pressure(bs):
        return max(load_sum[g] / rem[g] for g in range(gpu_num))

    incumbent = [x[:] for x in bins]
    upper = pressure(incumbent)
    lower = max(
        [loads[i] / (GPU_MEM_SIZE - sizes[i]) for i in range(n)] +
        [sum(loads) / (gpu_num * GPU_MEM_SIZE - sum(sizes)), 0.0]
    )

    try:
        import numpy as np
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import lil_matrix

        v = n * gpu_num
        A = lil_matrix((n + 2 * gpu_num, v), dtype=float)
        for i in range(n):
            for g in range(gpu_num):
                A[i, i * gpu_num + g] = 1.0
        for g in range(gpu_num):
            for i in range(n):
                k = i * gpu_num + g
                A[n + g, k] = sizes[i]
                A[n + gpu_num + g, k] = loads[i]
        best = incumbent
        lo, hi = lower, upper
        for _ in range(24):
            mid = (lo + hi) / 2.0
            for g in range(gpu_num):
                for i in range(n):
                    A[n + gpu_num + g, i * gpu_num + g] = loads[i] + mid * sizes[i]
            lb = np.r_[np.ones(n), np.full(gpu_num, -np.inf),
                       np.full(gpu_num, -np.inf)]
            ub = np.r_[np.ones(n), np.full(gpu_num, GPU_MEM_SIZE - eps),
                       np.full(gpu_num, GPU_MEM_SIZE * mid)]
            result = milp(np.zeros(v), integrality=np.ones(v),
                          bounds=Bounds(0, 1),
                          constraints=LinearConstraint(A.tocsr(), lb, ub),
                          options={"time_limit": 0.15})
            if not result.success or result.x is None:
                lo = mid
                continue
            trial = [[] for _ in range(gpu_num)]
            for i in range(n):
                g = int(np.argmax(result.x[i * gpu_num:(i + 1) * gpu_num]))
                trial[g].append(i)
            tr = [GPU_MEM_SIZE - sum(sizes[i] for i in trial[g])
                  for g in range(gpu_num)]
            tl = [sum(loads[i] for i in trial[g]) for g in range(gpu_num)]
            if min(tr) > eps and max(tl[g] / tr[g] for g in range(gpu_num)) <= mid + 1e-7:
                best, hi = trial, mid
            else:
                lo = mid
        bins = best
    except Exception:
        pass

    # Repartition the current worst GPU with each partner.  Small pairs are
    # enumerated exactly; larger pairs use a bounded depth-first enumeration.
    def gpu_pressure(items):
        free = GPU_MEM_SIZE - sum(sizes[i] for i in items)
        return (sum(loads[i] for i in items) / free
                if free > eps else float("inf"))

    def placement_pressure(state):
        return max(gpu_pressure(group) for group in state)

    worst = max(range(gpu_num), key=lambda g: gpu_pressure(bins[g]))
    incumbent_value = placement_pressure(bins)
    best_state = None
    best_value = incumbent_value

    for partner in range(gpu_num):
        if partner == worst:
            continue
        union = bins[worst] + bins[partner]
        k = len(union)
        if not union:
            continue

        pair_value = float("inf")
        pair_split = None

        def consider(mask):
            nonlocal pair_value, pair_split
            left = [union[j] for j in range(k) if mask & (1 << j)]
            right = [union[j] for j in range(k) if not (mask & (1 << j))]
            if (GPU_MEM_SIZE - sum(sizes[i] for i in left) <= eps or
                    GPU_MEM_SIZE - sum(sizes[i] for i in right) <= eps):
                return
            value = max(gpu_pressure(left), gpu_pressure(right))
            if value < pair_value:
                pair_value = value
                pair_split = (left, right)

        if k <= 18:
            for mask in range(1 << k):
                consider(mask)
        else:
            nodes = [0]
            chosen = []

            def visit(pos, used):
                if nodes[0] >= 50000:
                    return
                if pos == k:
                    nodes[0] += 1
                    consider(sum(1 << j for j in chosen))
                    return
                visit(pos + 1, used)
                new_used = used + sizes[union[pos]]
                if new_used < GPU_MEM_SIZE - eps:
                    chosen.append(pos)
                    visit(pos + 1, new_used)
                    chosen.pop()

            visit(0, 0.0)

        if pair_split is not None:
            trial = [group[:] for group in bins]
            trial[worst], trial[partner] = pair_split
            value = placement_pressure(trial)
            if value < best_value - 1e-12:
                best_value = value
                best_state = trial

    if best_state is not None:
        bins = best_state

    return {g: [models[i] for i in bins[g]] for g in range(gpu_num)}


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for gpu_num, gpu_models in test_cases:
        results = compute_model_placement(gpu_num, gpu_models)
        all_kvpr.append(safe_float(calculate_kvcache_pressure(results)))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")