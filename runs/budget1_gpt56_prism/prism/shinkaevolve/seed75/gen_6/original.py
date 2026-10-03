GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place models while minimizing the maximum post-placement KVPR."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # Place the models that are hardest to accommodate first.
    sorted_models = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo, m.model_size),
        reverse=True,
    )

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
    pressure = [0.0 for _ in range(gpu_num)]

    # Select by the resulting, not current, KVPR.
    for model in sorted_models:
        demand = model.req_rate / model.slo
        candidates = [
            gpu_id for gpu_id in range(gpu_num)
            if model.model_size < remaining[gpu_id]
        ]
        if not candidates:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        best_idx = min(
            candidates,
            key=lambda gpu_id: (
                (pressure[gpu_id] + demand) /
                (remaining[gpu_id] - model.model_size),
                -remaining[gpu_id],
            ),
        )
        placement[best_idx].append(model)
        pressure[best_idx] += demand
        remaining[best_idx] -= model.model_size

    # Relocate models from the bottleneck GPU when this lowers global KVPR.
    for _ in range(len(models)):
        ratios = [
            pressure[gpu_id] / remaining[gpu_id]
            for gpu_id in range(gpu_num)
        ]
        source = max(range(gpu_num), key=lambda gpu_id: ratios[gpu_id])
        old_max = ratios[source]
        best_move = None
        best_max = old_max

        for model in placement[source]:
            demand = model.req_rate / model.slo
            source_ratio = (pressure[source] - demand) / (
                remaining[source] + model.model_size
            )
            for target in range(gpu_num):
                if target == source or model.model_size >= remaining[target]:
                    continue
                target_ratio = (pressure[target] + demand) / (
                    remaining[target] - model.model_size
                )
                candidate_max = max(
                    source_ratio,
                    target_ratio,
                    *(
                        ratios[gpu_id] for gpu_id in range(gpu_num)
                        if gpu_id != source and gpu_id != target
                    ),
                )
                if candidate_max < best_max:
                    best_max = candidate_max
                    best_move = (model, target, demand)

        if best_move is None:
            break

        model, target, demand = best_move
        placement[source].remove(model)
        placement[target].append(model)
        remaining[source] += model.model_size
        remaining[target] -= model.model_size
        pressure[source] -= demand
        pressure[target] += demand

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