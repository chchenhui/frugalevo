GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use DIRECT over deterministic priority/GPU assignments, then improve by moves and swaps."""

    n = len(models)
    size = [m.model_size for m in models]
    weight = [m.req_rate / m.slo for m in models]
    eps = 1e-10

    if any(s >= GPU_MEM_SIZE - eps for s in size):
        raise ValueError("A model leaves no KV-cache memory")

    def pressure(free, load):
        return max(load[g] / free[g] for g in range(gpu_num))

    def decode(x=None):
        # x contains one ordering coordinate and one preferred-GPU coordinate
        # per model.  Canonicalizing labels removes equivalent GPU permutations.
        if x is None:
            order = sorted(range(n), key=lambda i: (-size[i], -weight[i]))
            preferred = [0] * n
        else:
            order = sorted(range(n), key=lambda i: (x[i], -size[i]))
            raw = [min(gpu_num - 1, int(x[n + i] * gpu_num)) for i in range(n)]
            rename, preferred = {}, []
            for g in raw:
                if g not in rename:
                    rename[g] = len(rename)
                preferred.append(rename[g])

        groups = {g: [] for g in range(gpu_num)}
        free, load = [GPU_MEM_SIZE] * gpu_num, [0.0] * gpu_num
        for i in order:
            choices = [g for g in range(gpu_num) if free[g] - size[i] > eps]
            if not choices:
                return None
            p = preferred[i]
            # A preferred label affects tie breaking, while the actual decision
            # minimizes the exact post-placement maximum KVPR.
            g = min(choices, key=lambda g: (
                max((load[j] + (weight[i] if j == g else 0.0)) /
                    (free[j] - (size[i] if j == g else 0.0))
                    for j in range(gpu_num)),
                g != p, g))
            groups[g].append(i)
            free[g] -= size[i]
            load[g] += weight[i]
        return groups, free, load

    def improve(state):
        groups, free, load = state
        for _ in range(min(5, n)):
            best, bestv = None, pressure(free, load)
            for a in range(gpu_num):
                for i in groups[a]:
                    for b in range(gpu_num):
                        if a == b or free[b] - size[i] <= eps:
                            continue
                        free[a] += size[i]; load[a] -= weight[i]
                        free[b] -= size[i]; load[b] += weight[i]
                        v = pressure(free, load)
                        free[a] -= size[i]; load[a] += weight[i]
                        free[b] += size[i]; load[b] -= weight[i]
                        if v < bestv - 1e-12:
                            best, bestv = ("move", a, b, i), v
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for i in groups[a]:
                        for j in groups[b]:
                            if free[a] + size[i] - size[j] <= eps or free[b] + size[j] - size[i] <= eps:
                                continue
                            free[a] += size[i] - size[j]; load[a] += weight[j] - weight[i]
                            free[b] += size[j] - size[i]; load[b] += weight[i] - weight[j]
                            v = pressure(free, load)
                            free[a] -= size[i] - size[j]; load[a] -= weight[j] - weight[i]
                            free[b] -= size[j] - size[i]; load[b] -= weight[i] - weight[j]
                            if v < bestv - 1e-12:
                                best, bestv = ("swap", a, b, i, j), v
            if best is None:
                break
            if best[0] == "move":
                _, a, b, i = best
                groups[a].remove(i); groups[b].append(i)
                free[a] += size[i]; load[a] -= weight[i]
                free[b] -= size[i]; load[b] += weight[i]
            else:
                _, a, b, i, j = best
                groups[a].remove(i); groups[b].remove(j)
                groups[a].append(j); groups[b].append(i)
                free[a] += size[i] - size[j]; load[a] += weight[j] - weight[i]
                free[b] += size[j] - size[i]; load[b] += weight[i] - weight[j]
        return groups, free, load

    best = decode()
    if best is None:
        raise ValueError("Models do not fit in the available GPU memory")
    best = improve(best)
    bestv = pressure(best[1], best[2])

    try:
        from scipy.optimize import direct

        def objective(x):
            state = decode(x)
            if state is None:
                return 1e12
            # This is exactly max load/(80-used), equivalently testing all
            # load + T*used <= 80*T feasibility thresholds.
            return pressure(state[1], state[2])

        result = direct(objective, [(0.0, 1.0)] * (2 * n),
                        maxfun=min(1200, 120 + 60 * n),
                        locally_biased=False)
        candidate = decode(result.x)
        if candidate is not None:
            candidate = improve(candidate)
            value = pressure(candidate[1], candidate[2])
            if value < bestv:
                best, bestv = candidate, value
    except Exception:
        pass

    return {g: [models[i] for i in best[0][g]] for g in range(gpu_num)}

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
