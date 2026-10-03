GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement that minimizes maximum KV cache pressure.
    """

    def demand(model):
        return model.req_rate / model.slo

    def pressure(load, remaining):
        if remaining > 0:
            return load / remaining
        return float("inf") if load > 0 else 0.0

    def greedy(order, use_resulting_pressure=True):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        loads = [0.0] * gpu_num
        scores = [0.0] * gpu_num

        for model in order:
            model_demand = demand(model)
            best_gpu = None
            best_key = None

            for gpu_id in range(gpu_num):
                if model.model_size > remaining[gpu_id]:
                    continue

                new_remaining = remaining[gpu_id] - model.model_size
                new_score = pressure(loads[gpu_id] + model_demand, new_remaining)

                if use_resulting_pressure:
                    objective = max(
                        new_score,
                        max(scores[:gpu_id] + scores[gpu_id + 1:], default=0.0),
                    )
                    key = (objective, new_score, new_remaining)
                else:
                    key = (scores[gpu_id], new_score, new_remaining)

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu_id

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            loads[best_gpu] += model_demand
            remaining[best_gpu] -= model.model_size
            scores[best_gpu] = pressure(loads[best_gpu], remaining[best_gpu])

        return placement, loads, remaining, scores

    def best_fit_size_order(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        loads = [0.0] * gpu_num

        for model in order:
            candidates = [
                gpu_id for gpu_id in range(gpu_num)
                if model.model_size <= remaining[gpu_id]
            ]
            if not candidates:
                return None

            gpu_id = min(candidates, key=lambda i: remaining[i] - model.model_size)
            placement[gpu_id].append(model)
            remaining[gpu_id] -= model.model_size
            loads[gpu_id] += demand(model)

        scores = [pressure(loads[i], remaining[i]) for i in range(gpu_num)]
        return placement, loads, remaining, scores

    by_pressure = sorted(models, key=demand, reverse=True)
    by_size = sorted(models, key=lambda m: m.model_size, reverse=True)
    by_singleton_pressure = sorted(
        models,
        key=lambda m: demand(m) / max(1e-12, GPU_MEM_SIZE - m.model_size),
        reverse=True,
    )
    by_density = sorted(
        models,
        key=lambda m: demand(m) / max(1e-12, m.model_size),
        reverse=True,
    )

    candidates = []
    for order in (by_pressure, by_size, by_singleton_pressure, by_density):
        result = greedy(order, True)
        if result is not None:
            candidates.append(result)

    result = greedy(by_pressure, False)
    if result is not None:
        candidates.append(result)

    result = best_fit_size_order(by_size)
    if result is not None:
        candidates.append(result)

    if not candidates:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs with {GPU_MEM_SIZE} GB each."
        )

    placement, loads, remaining, scores = min(
        candidates, key=lambda result: max(result[3], default=0.0)
    )

    while True:
        current_max = max(scores, default=0.0)
        best_action = None
        best_max = current_max

        for source in range(gpu_num):
            for model in placement[source]:
                model_demand = demand(model)

                for target in range(gpu_num):
                    if source == target or model.model_size > remaining[target]:
                        continue

                    source_load = loads[source] - model_demand
                    source_remaining = remaining[source] + model.model_size
                    target_load = loads[target] + model_demand
                    target_remaining = remaining[target] - model.model_size

                    new_scores = scores[:]
                    new_scores[source] = pressure(source_load, source_remaining)
                    new_scores[target] = pressure(target_load, target_remaining)
                    new_max = max(new_scores, default=0.0)

                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_action = (
                            "move", source, target, model,
                            source_load, source_remaining,
                            target_load, target_remaining,
                        )

        for first in range(gpu_num):
            for second in range(first + 1, gpu_num):
                for model_a in placement[first]:
                    demand_a = demand(model_a)

                    for model_b in placement[second]:
                        if (model_b.model_size > remaining[first] + model_a.model_size or
                                model_a.model_size > remaining[second] + model_b.model_size):
                            continue

                        demand_b = demand(model_b)
                        first_load = loads[first] - demand_a + demand_b
                        second_load = loads[second] - demand_b + demand_a
                        first_remaining = (
                            remaining[first] + model_a.model_size - model_b.model_size
                        )
                        second_remaining = (
                            remaining[second] + model_b.model_size - model_a.model_size
                        )

                        new_scores = scores[:]
                        new_scores[first] = pressure(first_load, first_remaining)
                        new_scores[second] = pressure(second_load, second_remaining)
                        new_max = max(new_scores, default=0.0)

                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_action = (
                                "swap", first, second, model_a, model_b,
                                first_load, first_remaining,
                                second_load, second_remaining,
                            )

        if best_action is None:
            break

        if best_action[0] == "move":
            (_, source, target, model,
             source_load, source_remaining,
             target_load, target_remaining) = best_action

            placement[source].remove(model)
            placement[target].append(model)
            loads[source] = source_load
            remaining[source] = source_remaining
            loads[target] = target_load
            remaining[target] = target_remaining
        else:
            (_, first, second, model_a, model_b,
             first_load, first_remaining,
             second_load, second_remaining) = best_action

            placement[first].remove(model_a)
            placement[second].remove(model_b)
            placement[first].append(model_b)
            placement[second].append(model_a)
            loads[first] = first_load
            remaining[first] = first_remaining
            loads[second] = second_load
            remaining[second] = second_remaining

        scores = [pressure(loads[i], remaining[i]) for i in range(gpu_num)]

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