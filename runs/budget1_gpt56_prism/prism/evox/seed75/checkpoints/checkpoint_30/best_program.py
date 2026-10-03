GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily balance current KVPR, then apply strictly improving moves or swaps."""
    placement = {g: [] for g in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        demand = model.req_rate / model.slo
        gpu = min(
            (g for g in range(gpu_num) if model.model_size <= remaining[g]),
            key=lambda g: load[g] / max(remaining[g], 1e-12),
            default=None,
        )
        if gpu is None:
            raise ValueError(f"Unable to place model of size {model.model_size} GB.")
        placement[gpu].append(model)
        load[gpu] += demand
        remaining[gpu] -= model.model_size

    def score():
        return max(load[g] / max(remaining[g], 1e-12) for g in range(gpu_num))

    while True:
        best_score, best = score(), None
        for source in range(gpu_num):
            for model in placement[source]:
                demand, size = model.req_rate / model.slo, model.model_size
                for target in range(gpu_num):
                    if source == target or size > remaining[target]:
                        continue
                    old = load[source], remaining[source], load[target], remaining[target]
                    load[source] -= demand
                    remaining[source] += size
                    load[target] += demand
                    remaining[target] -= size
                    candidate = score()
                    load[source], remaining[source], load[target], remaining[target] = old
                    if candidate < best_score:
                        best_score, best = candidate, (source, target, model)

        for source in range(gpu_num):
            for target in range(source + 1, gpu_num):
                for left in placement[source]:
                    dl, sl = left.req_rate / left.slo, left.model_size
                    for right in placement[target]:
                        dr, sr = right.req_rate / right.slo, right.model_size
                        if sl > remaining[target] + sr or sr > remaining[source] + sl:
                            continue
                        old = load[source], remaining[source], load[target], remaining[target]
                        load[source] += dr - dl
                        remaining[source] += sl - sr
                        load[target] += dl - dr
                        remaining[target] += sr - sl
                        candidate = score()
                        load[source], remaining[source], load[target], remaining[target] = old
                        if candidate < best_score:
                            best_score, best = candidate, (source, target, left, right)

        if best is None:
            return placement
        if len(best) == 3:
            source, target, model = best
            placement[source].remove(model)
            placement[target].append(model)
            demand, size = model.req_rate / model.slo, model.model_size
            load[source] -= demand
            remaining[source] += size
            load[target] += demand
            remaining[target] -= size
        else:
            source, target, left, right = best
            placement[source].remove(left)
            placement[target].remove(right)
            placement[source].append(right)
            placement[target].append(left)
            dl, sl = left.req_rate / left.slo, left.model_size
            dr, sr = right.req_rate / right.slo, right.model_size
            load[source] += dr - dl
            remaining[source] += sl - sr
            load[target] += dl - dr
            remaining[target] += sr - sl

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
