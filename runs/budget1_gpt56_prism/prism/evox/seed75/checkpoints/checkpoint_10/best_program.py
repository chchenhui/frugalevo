GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily balance KVPR, then apply the best feasible KVPR-reducing model swap."""
    placement = {g: [] for g in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        demand = model.req_rate / model.slo
        gpu = min(
            (g for g in range(gpu_num) if model.model_size <= remaining[g]),
            key=lambda g: load[g] / remaining[g],
            default=None,
        )
        if gpu is None:
            raise ValueError(f"Unable to place model of size {model.model_size} GB.")
        placement[gpu].append(model)
        load[gpu] += demand
        remaining[gpu] -= model.model_size

    def pressure(demand, memory):
        return demand / memory if memory > 0 else float("inf")

    best = max(pressure(load[g], remaining[g]) for g in range(gpu_num))
    swap = None
    for source in range(gpu_num):
        for target in range(source + 1, gpu_num):
            other = max(
                (pressure(load[g], remaining[g]) for g in range(gpu_num)
                 if g != source and g != target),
                default=0.0,
            )
            for i, left in enumerate(placement[source]):
                left_load = left.req_rate / left.slo
                for j, right in enumerate(placement[target]):
                    right_load = right.req_rate / right.slo
                    source_mem = remaining[source] + left.model_size - right.model_size
                    target_mem = remaining[target] + right.model_size - left.model_size
                    if source_mem < 0 or target_mem < 0:
                        continue
                    score = max(
                        other,
                        pressure(load[source] - left_load + right_load, source_mem),
                        pressure(load[target] - right_load + left_load, target_mem),
                    )
                    if score < best:
                        best = score
                        swap = source, target, i, j, left_load, right_load, source_mem, target_mem

    if swap is not None:
        source, target, i, j, left_load, right_load, source_mem, target_mem = swap
        placement[source][i], placement[target][j] = placement[target][j], placement[source][i]
        load[source] += right_load - left_load
        load[target] += left_load - right_load
        remaining[source], remaining[target] = source_mem, target_mem

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
