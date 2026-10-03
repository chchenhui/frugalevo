GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use greedy incumbents and bounded MILP bisection to minimize maximum KVPR."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    n = len(models)
    size = [m.model_size for m in models]
    pressure = [m.req_rate / m.slo for m in models]

    if any(s > GPU_MEM_SIZE for s in size):
        raise ValueError("A model is larger than a GPU")

    def ratio(w, u):
        remain = GPU_MEM_SIZE - u
        return w / remain if remain > 0 else float("inf")

    def objective(used, weight):
        # The full descending vector makes plateau moves useful and deterministic.
        return tuple(sorted(
            (ratio(weight[g], used[g]) for g in range(gpu_num)),
            reverse=True,
        ))

    def construct(order):
        bins = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        weight = [0.0] * gpu_num
        for k in order:
            best_g = -1
            best_key = None
            for g in range(gpu_num):
                if used[g] + size[k] > GPU_MEM_SIZE:
                    continue
                nu = used[:]
                nw = weight[:]
                nu[g] += size[k]
                nw[g] += pressure[k]
                rs = [ratio(nw[j], nu[j]) for j in range(gpu_num)]
                # Compare the complete resulting KVPR profile.  This matches
                # the local-search objective and lets construction reduce the
                # next bottleneck when the maximum is tied.
                key = (tuple(sorted(rs, reverse=True)), -used[g])
                if best_key is None or key < best_key:
                    best_key, best_g = key, g
            if best_g < 0:
                return None
            bins[best_g].append(k)
            used[best_g] += size[k]
            weight[best_g] += pressure[k]
        return bins, used, weight

    # Different constructions are useful because memory packing can block an
    # otherwise pressure-balanced greedy placement.
    orders = [
        sorted(range(n), key=lambda k: (size[k], pressure[k]), reverse=True),
        sorted(range(n), key=lambda k: (pressure[k], size[k]), reverse=True),
        sorted(range(n), key=lambda k: (size[k] * pressure[k], size[k]), reverse=True),
        sorted(range(n), key=lambda k: (
            pressure[k] / size[k] if size[k] else float("inf"), pressure[k]
        ), reverse=True),
    ]

    seeds = [x for x in (construct(order) for order in orders) if x is not None]
    if not seeds:
        raise ValueError("Unable to place all models on the available GPUs")

    best = min(seeds, key=lambda state: objective(state[1], state[2]))
    best_obj = objective(best[1], best[2])

    # A fixed threshold T is feasible exactly when each GPU satisfies
    # sum(pressure + T*size) <= GPU_MEM_SIZE*T.
    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds

        incumbent = best_obj[0]
        lo, hi = 0.0, max(float(incumbent), 1e-12)
        total = n * gpu_num
        assignment = None

        for _ in range(18):
            threshold = (lo + hi) / 2.0
            rows = []
            lower = []
            upper = []

            for k in range(n):
                row = np.zeros(total)
                for g in range(gpu_num):
                    row[k * gpu_num + g] = 1.0
                rows.append(row)
                lower.append(1.0)
                upper.append(1.0)

            for g in range(gpu_num):
                row = np.zeros(total)
                for k in range(n):
                    row[k * gpu_num + g] = size[k]
                rows.append(row)
                lower.append(-np.inf)
                upper.append(GPU_MEM_SIZE)

                row = np.zeros(total)
                for k in range(n):
                    row[k * gpu_num + g] = pressure[k] + threshold * size[k]
                rows.append(row)
                lower.append(-np.inf)
                upper.append(GPU_MEM_SIZE * threshold)

            result = milp(
                c=np.zeros(total),
                integrality=np.ones(total),
                bounds=Bounds(np.zeros(total), np.ones(total)),
                constraints=LinearConstraint(
                    np.asarray(rows), np.asarray(lower), np.asarray(upper)
                ),
                options={"time_limit": 0.15},
            )

            if result.x is None:
                lo = threshold
                continue

            decoded = [[] for _ in range(gpu_num)]
            for k in range(n):
                chosen = [
                    g for g in range(gpu_num)
                    if result.x[k * gpu_num + g] > 0.5
                ]
                if len(chosen) != 1:
                    decoded = None
                    break
                decoded[chosen[0]].append(k)

            if decoded is None:
                lo = threshold
                continue

            used = [sum(size[k] for k in group) for group in decoded]
            weight = [sum(pressure[k] for k in group) for group in decoded]
            if max(used, default=0.0) <= GPU_MEM_SIZE:
                candidate = (decoded, used, weight)
                value = objective(used, weight)
                if value < best_obj:
                    best, best_obj = candidate, value
                assignment = candidate
                hi = threshold
            else:
                lo = threshold

    except (ImportError, Exception):
        pass

    return {g: [models[k] for k in best[0][g]] for g in range(gpu_num)}

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):
        results = compute_model_placement(gpu_num, gpu_models)
        all_kvpr.append(safe_float(calculate_kvcache_pressure(results)))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr
    print(f"Max KVPR: {avg_kvpr:.3f}")