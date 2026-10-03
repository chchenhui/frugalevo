GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-restart greedy + local search. For each of several model orderings
    (by req_rate/slo, by size, by load density), greedily assign each model to
    the GPU minimizing resulting KVPR, then refine with single-model moves and
    pairwise swaps. Return the placement with the lowest max KVPR.
    """

    def kvpr(load, free):
        return load / free if free > 0 else float('inf')

    def solve(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        # Greedy assignment minimizing KVPR after placement
        for model in order:
            r = model.req_rate / model.slo
            best_idx, best_ratio = None, float('inf')
            for gpu_id in range(gpu_num):
                if free[gpu_id] - model.model_size > 0:
                    ratio = kvpr(load[gpu_id] + r, free[gpu_id] - model.model_size)
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, gpu_id
            if best_idx is None:
                return None, None, None
            placement[best_idx].append(model)
            load[best_idx] += r
            free[best_idx] -= model.model_size
        return placement, free, load

    def refine(placement, free, load):
        improved = True
        while improved:
            improved = False
            cur_max = max(kvpr(load[g], free[g]) for g in range(gpu_num))

            # Pass 1: move a single model to another GPU
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
                        if new_max < cur_max - 1e-12:
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

            # Pass 2: swap a pair of models between two GPUs
            for src in range(gpu_num):
                for dst in range(src + 1, gpu_num):
                    for m1 in list(placement[src]):
                        for m2 in list(placement[dst]):
                            r1, r2 = m1.req_rate / m1.slo, m2.req_rate / m2.slo
                            if free[src] + m1.model_size - m2.model_size <= 0:
                                continue
                            if free[dst] + m2.model_size - m1.model_size <= 0:
                                continue
                            others = max(
                                (kvpr(load[g], free[g]) for g in range(gpu_num)
                                 if g not in (src, dst)), default=0.0)
                            new_max = max(
                                kvpr(load[src] - r1 + r2,
                                     free[src] + m1.model_size - m2.model_size),
                                kvpr(load[dst] - r2 + r1,
                                     free[dst] + m2.model_size - m1.model_size),
                                others,
                            )
                            if new_max < cur_max - 1e-12:
                                placement[src].remove(m1); placement[src].append(m2)
                                placement[dst].remove(m2); placement[dst].append(m1)
                                load[src] += r2 - r1
                                free[src] += m1.model_size - m2.model_size
                                load[dst] += r1 - r2
                                free[dst] += m2.model_size - m1.model_size
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
        list(models),
    ]

    best_placement, best_max = None, float('inf')
    for order in orders:
        placement, free, load = solve(order)
        if placement is None:
            continue
        placement = refine(placement, free, load)
        cur_max = max(kvpr(load[g], free[g]) for g in range(gpu_num))
        if cur_max < best_max:
            best_max, best_placement = cur_max, placement

    if best_placement is None:
        raise ValueError("Cannot fit models in GPU memory")

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
