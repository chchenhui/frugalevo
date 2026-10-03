GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily place hard models, then relocate models while reducing max KVPR."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def pressure(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    ordered = sorted(
        models,
        key=lambda m: pressure(m.req_rate / m.slo, GPU_MEM_SIZE - m.model_size),
        reverse=True,
    )

    for model in ordered:
        weight = model.req_rate / model.slo
        best = min(
            (
                gpu for gpu in range(gpu_num)
                if remaining[gpu] >= model.model_size
            ),
            key=lambda gpu: pressure(
                load[gpu] + weight, remaining[gpu] - model.model_size
            ),
            default=None,
        )
        if best is None:
            raise ValueError("Models cannot fit in the available GPU memory")
        placement[best].append(model)
        load[best] += weight
        remaining[best] -= model.model_size

    # A few best-improving relocations repair greedy decisions without expensive search.
    for _ in range(min(8, len(models))):
        current = max(pressure(load[g], remaining[g]) for g in range(gpu_num))
        best_move = None
        best_value = current

        for source in range(gpu_num):
            for model in placement[source]:
                size = model.model_size
                weight = model.req_rate / model.slo
                for target in range(gpu_num):
                    if target == source or remaining[target] < size:
                        continue
                    value = max(
                        pressure(
                            load[g] + (weight if g == target else -weight if g == source else 0),
                            remaining[g] - (size if g == target else -size if g == source else 0),
                        )
                        for g in range(gpu_num)
                    )
                    if value < best_value:
                        best_value = value
                        best_move = source, target, model, size, weight

        if best_move is None:
            break
        source, target, model, size, weight = best_move
        placement[source].remove(model)
        placement[target].append(model)
        load[source] -= weight
        remaining[source] += size
        load[target] += weight
        remaining[target] -= size

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
