GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement minimizing the maximum KV cache pressure.
    """

    def kvpr(load, remaining):
        if remaining <= 0:
            return float("inf") if load > 0 else 0.0
        return load / remaining

    # Try complementary packing orders.  Large models need roomy GPUs, whereas
    # high-load models dominate the objective; retaining the best feasible seed
    # gives the local search a substantially better starting point.
    orders = [
        sorted(models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True),
        sorted(models, key=lambda m: (m.model_size, m.req_rate / m.slo), reverse=True),
        sorted(
            models,
            key=lambda m: (
                (m.req_rate / m.slo) / max(1e-9, GPU_MEM_SIZE - m.model_size),
                m.model_size,
            ),
            reverse=True,
        ),
    ]

    best_seed = None
    for ordered_models in orders:
        candidate = {gpu_id: [] for gpu_id in range(gpu_num)}
        candidate_remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
        candidate_load = [0.0 for _ in range(gpu_num)]
        feasible = True

        for model in ordered_models:
            model_load = model.req_rate / model.slo
            best_gpu = None
            best_max_kvpr = float("inf")

            for gpu_id in range(gpu_num):
                if model.model_size > candidate_remaining[gpu_id]:
                    continue
                projected_max = max(
                    kvpr(
                        candidate_load[i] + (model_load if i == gpu_id else 0.0),
                        candidate_remaining[i] - (model.model_size if i == gpu_id else 0.0),
                    )
                    for i in range(gpu_num)
                )
                if projected_max < best_max_kvpr:
                    best_max_kvpr = projected_max
                    best_gpu = gpu_id

            if best_gpu is None:
                feasible = False
                break
            candidate[best_gpu].append(model)
            candidate_load[best_gpu] += model_load
            candidate_remaining[best_gpu] -= model.model_size

        if feasible:
            score = max(kvpr(candidate_load[i], candidate_remaining[i]) for i in range(gpu_num))
            if best_seed is None or score < best_seed[0]:
                best_seed = (score, candidate, candidate_remaining, candidate_load)

    if best_seed is None:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB"
        )

    _, placement, remaining, load = best_seed

    # Apply the best improving relocation or pairwise swap.  Swaps escape
    # memory-constrained local minima where neither model can move alone.
    while True:
        current_max = max(kvpr(load[i], remaining[i]) for i in range(gpu_num))
        best_action = None
        best_max_kvpr = current_max

        for source in range(gpu_num):
            for model in placement[source]:
                model_load = model.req_rate / model.slo
                for target in range(gpu_num):
                    if source == target or model.model_size > remaining[target]:
                        continue

                    candidate_max = 0.0
                    for gpu_id in range(gpu_num):
                        if gpu_id == source:
                            value = kvpr(
                                load[gpu_id] - model_load,
                                remaining[gpu_id] + model.model_size,
                            )
                        elif gpu_id == target:
                            value = kvpr(
                                load[gpu_id] + model_load,
                                remaining[gpu_id] - model.model_size,
                            )
                        else:
                            value = kvpr(load[gpu_id], remaining[gpu_id])
                        candidate_max = max(candidate_max, value)

                    if candidate_max < best_max_kvpr - 1e-12:
                        best_max_kvpr = candidate_max
                        best_action = ("move", source, target, model, model_load)

        for left in range(gpu_num):
            for right in range(left + 1, gpu_num):
                for left_model in placement[left]:
                    left_size = left_model.model_size
                    left_load = left_model.req_rate / left_model.slo
                    for right_model in placement[right]:
                        right_size = right_model.model_size
                        right_load = right_model.req_rate / right_model.slo
                        left_remaining = remaining[left] + left_size - right_size
                        right_remaining = remaining[right] + right_size - left_size

                        if left_remaining < 0 or right_remaining < 0:
                            continue

                        candidate_max = 0.0
                        for gpu_id in range(gpu_num):
                            if gpu_id == left:
                                value = kvpr(
                                    load[left] - left_load + right_load,
                                    left_remaining,
                                )
                            elif gpu_id == right:
                                value = kvpr(
                                    load[right] - right_load + left_load,
                                    right_remaining,
                                )
                            else:
                                value = kvpr(load[gpu_id], remaining[gpu_id])
                            candidate_max = max(candidate_max, value)

                        if candidate_max < best_max_kvpr - 1e-12:
                            best_max_kvpr = candidate_max
                            best_action = (
                                "swap",
                                left,
                                right,
                                left_model,
                                right_model,
                                left_load,
                                right_load,
                            )

        if best_action is None:
            break

        if best_action[0] == "move":
            _, source, target, model, model_load = best_action
            placement[source].remove(model)
            placement[target].append(model)
            load[source] -= model_load
            load[target] += model_load
            remaining[source] += model.model_size
            remaining[target] -= model.model_size
        else:
            _, left, right, left_model, right_model, left_load, right_load = best_action
            placement[left].remove(left_model)
            placement[right].remove(right_model)
            placement[left].append(right_model)
            placement[right].append(left_model)
            load[left] += right_load - left_load
            load[right] += left_load - right_load
            remaining[left] += left_model.model_size - right_model.model_size
            remaining[right] += right_model.model_size - left_model.model_size

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