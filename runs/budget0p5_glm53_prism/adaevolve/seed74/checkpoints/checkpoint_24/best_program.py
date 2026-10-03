GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy + local search to build an initial solution, then
    simulated annealing over the model->GPU assignment array. SA moves are
    single-model relocations or pairwise swaps; the objective is max KVPR
    plus a large penalty for memory overflow, so the search may pass
    transiently through infeasible states while always remembering the best
    feasible state ever visited. Geometric cooling, fixed seed, incremental
    load/free updates. Always returns a dict gpu_id -> list of models.
    """

    def kvpr(load, free):
        return load / free if free > 0 else float('inf')

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            r = model.req_rate / model.slo
            best_idx, best_ratio = None, float('inf')
            for g in range(gpu_num):
                if free[g] - model.model_size > 0:
                    ratio = kvpr(load[g] + r, free[g] - model.model_size)
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None, None, None
            placement[best_idx].append(model)
            load[best_idx] += r
            free[best_idx] -= model.model_size
        return placement, free, load

    def refine(placement, free, load):
        # Local search accepting improving moves plus a budgeted number of
        # sideways (equal max KVPR) moves to escape plateaus.
        sideways_budget = 20
        improved = True
        while improved:
            improved = False
            cur_max = max(kvpr(load[g], free[g]) for g in range(gpu_num))
            # Pass 1: single-model moves
            for src in range(gpu_num):
                for model in list(placement[src]):
                    r = model.req_rate / model.slo
                    for dst in range(gpu_num):
                        if dst == src or free[dst] - model.model_size <= 0:
                            continue
                        new_max = max(
                            kvpr(load[src] - r, free[src] + model.model_size),
                            kvpr(load[dst] + r, free[dst] - model.model_size),
                            max(kvpr(load[g], free[g]) for g in range(gpu_num)
                                if g not in (src, dst)),
                        )
                        if new_max < cur_max - 1e-12 or (
                                new_max <= cur_max + 1e-12
                                and sideways_budget > 0):
                            if new_max >= cur_max - 1e-12:
                                sideways_budget -= 1
                            placement[src].remove(model)
                            placement[dst].append(model)
                            load[src] -= r; free[src] += model.model_size
                            load[dst] += r; free[dst] -= model.model_size
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Pass 2: pairwise swaps
            for src in range(gpu_num):
                for dst in range(src + 1, gpu_num):
                    for m1 in list(placement[src]):
                        for m2 in list(placement[dst]):
                            r1, r2 = m1.req_rate / m1.slo, m2.req_rate / m2.slo
                            fs = free[src] + m1.model_size - m2.model_size
                            fd = free[dst] + m2.model_size - m1.model_size
                            if fs <= 0 or fd <= 0:
                                continue
                            others = max(
                                (kvpr(load[g], free[g]) for g in range(gpu_num)
                                 if g not in (src, dst)), default=0.0)
                            new_max = max(
                                kvpr(load[src] - r1 + r2, fs),
                                kvpr(load[dst] - r2 + r1, fd),
                                others,
                            )
                            if new_max < cur_max - 1e-12:
                                placement[src].remove(m1); placement[src].append(m2)
                                placement[dst].remove(m2); placement[dst].append(m1)
                                load[src] += r2 - r1
                                free[src] = fs
                                load[dst] += r1 - r2
                                free[dst] = fd
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement

    import random
    rng = random.Random(42)
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        list(models),
    ]
    # Seeded random restarts for extra diversity
    for _ in range(6):
        order = list(models)
        rng.shuffle(order)
        orders.append(order)

    best_placement, best_max = None, float('inf')
    best_assign = None
    for order in orders:
        placement, free, load = greedy(order)
        if placement is None:
            continue
        placement = refine(placement, free, load)
        cur_max = max(kvpr(load[g], free[g]) for g in range(gpu_num))
        if cur_max < best_max:
            best_max, best_placement = cur_max, placement
            best_assign = [None] * len(models)
            for g, ms in placement.items():
                for m in ms:
                    best_assign[models.index(m)] = g

    # ---- Simulated annealing over model->GPU assignments ----
    if best_assign is not None and len(models) > 0:
        import random
        rng = random.Random(1234)
        n = len(models)
        r = [m.req_rate / m.slo for m in models]
        s = [m.model_size for m in models]
        assign = list(best_assign)
        load = [0.0] * gpu_num
        free = [GPU_MEM_SIZE] * gpu_num
        for i in range(n):
            load[assign[i]] += r[i]
            free[assign[i]] -= s[i]

        def energy():
            pen = 0.0
            mx = 0.0
            for g in range(gpu_num):
                if free[g] <= 0:
                    pen += (GPU_MEM_SIZE - free[g])
                    mx = float('inf')
                else:
                    v = load[g] / free[g]
                    if v > mx:
                        mx = v
            return mx + 1e6 * pen

        cur_e = energy()
        iters = min(20000, 400 * n * gpu_num)
        T0, T1 = 1.0, 1e-4
        alpha = (T1 / T0) ** (1.0 / max(1, iters))
        T = T0
        for _ in range(iters):
            i = rng.randrange(n)
            if rng.random() < 0.5:
                # move model i to a random other gpu
                g2 = rng.randrange(gpu_num)
                if g2 == assign[i]:
                    continue
                g1 = assign[i]
                load[g1] -= r[i]; free[g1] += s[i]
                load[g2] += r[i]; free[g2] -= s[i]
                new_e = energy()
                if new_e <= cur_e or rng.random() < pow(2.718281828,
                                                        -(new_e - cur_e) / T):
                    assign[i] = g2
                    cur_e = new_e
                else:
                    load[g1] += r[i]; free[g1] -= s[i]
                    load[g2] -= r[i]; free[g2] += s[i]
            else:
                # swap model i with a random model on another gpu
                j = rng.randrange(n)
                if assign[i] == assign[j]:
                    continue
                g1, g2 = assign[i], assign[j]
                load[g1] += r[j] - r[i]; free[g1] += s[i] - s[j]
                load[g2] += r[i] - r[j]; free[g2] += s[j] - s[i]
                assign[i], assign[j] = g2, g1
                new_e = energy()
                if new_e <= cur_e or rng.random() < pow(2.718281828,
                                                        -(new_e - cur_e) / T):
                    cur_e = new_e
                else:
                    assign[i], assign[j] = g1, g2
                    load[g1] -= r[j] - r[i]; free[g1] -= s[i] - s[j]
                    load[g2] -= r[i] - r[j]; free[g2] -= s[j] - s[i]
            T *= alpha
            # track best feasible
            if all(free[g] > 0 for g in range(gpu_num)):
                mx = max(load[g] / free[g] for g in range(gpu_num))
                if mx < best_max - 1e-12:
                    best_max = mx
                    best_assign = list(assign)

        if best_assign is not None:
            best_placement = {g: [] for g in range(gpu_num)}
            for i in range(n):
                best_placement[best_assign[i]].append(models[i])

    if best_placement is None:
        # No ordering fit; fall back to a best-effort placement that always
        # returns a dict (overcommit only if truly impossible).
        best_placement, _, _ = greedy(sorted(models, key=lambda m: m.model_size))
        if best_placement is None:
            placement = {g: [] for g in range(gpu_num)}
            free = [GPU_MEM_SIZE] * gpu_num
            for model in sorted(models, key=lambda m: m.model_size):
                fits = [g for g in range(gpu_num) if free[g] >= model.model_size]
                g = fits[0] if fits else min(range(gpu_num),
                                             key=lambda x: free[x])
                placement[g].append(model)
                free[g] -= model.model_size
            best_placement = placement

    return best_placement

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
