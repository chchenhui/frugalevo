GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement (several sort orders) each followed by a
    local search with move and swap phases; return the best placement found.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    def kvpr(w, free_mem):
        return w / free_mem if free_mem > 0 else float('inf')

    def solve(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        weight = [0.0] * gpu_num
        for model in order:
            best_idx, best_val = None, float('inf')
            for g in range(gpu_num):
                if model.model_size <= free[g]:
                    val = kvpr(weight[g] + model.req_rate / model.slo,
                               free[g] - model.model_size)
                    if val < best_val:
                        best_val, best_idx = val, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weight[best_idx] += model.req_rate / model.slo
            free[best_idx] -= model.model_size

        # Local search: move or swap models to lower the max KVPR
        improved = True
        while improved:
            improved = False
            worst = max(range(gpu_num),
                        key=lambda g: kvpr(weight[g], free[g]))
            worst_val = kvpr(weight[worst], free[worst])
            # Move phase
            for m in list(placement[worst]):
                src_w = weight[worst] - m.req_rate / m.slo
                src_f = free[worst] + m.model_size
                for g in range(gpu_num):
                    if g == worst or m.model_size > free[g]:
                        continue
                    dst = kvpr(weight[g] + m.req_rate / m.slo,
                               free[g] - m.model_size)
                    if max(dst, kvpr(src_w, src_f)) < worst_val:
                        placement[worst].remove(m)
                        placement[g].append(m)
                        weight[worst], free[worst] = src_w, src_f
                        weight[g] += m.req_rate / m.slo
                        free[g] -= m.model_size
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Swap phase
            for m in list(placement[worst]):
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    for m2 in list(placement[g]):
                        if (free[worst] + m.model_size - m2.model_size < 0 or
                                free[g] + m2.model_size - m.model_size < 0):
                            continue
                        sw = weight[worst] - m.req_rate / m.slo + m2.req_rate / m2.slo
                        sf = free[worst] + m.model_size - m2.model_size
                        dw = weight[g] - m2.req_rate / m2.slo + m.req_rate / m.slo
                        df = free[g] + m2.model_size - m.model_size
                        if max(kvpr(sw, sf), kvpr(dw, df)) < worst_val:
                            placement[worst].remove(m)
                            placement[g].remove(m2)
                            placement[worst].append(m2)
                            placement[g].append(m)
                            weight[worst], free[worst] = sw, sf
                            weight[g], free[g] = dw, df
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement, max(kvpr(weight[g], free[g]) for g in range(gpu_num))

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size,
               reverse=True),
        list(models),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size),
    ]

    best, best_val = None, float('inf')
    for order in orders:
        res = solve(order)
        if res is None:
            raise ValueError("Unable to place all models on the GPUs.")
        placement, val = res
        if val < best_val:
            best, best_val = placement, val
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
