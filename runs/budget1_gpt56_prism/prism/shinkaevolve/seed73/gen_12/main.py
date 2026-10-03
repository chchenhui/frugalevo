GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement that minimizes maximum KV cache pressure.
    """

    def pressure(load, remaining):
        if remaining > 0:
            return load / remaining
        return float("inf") if load > 0 else 0.0

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        loads = [0.0] * gpu_num
        scores = [0.0] * gpu_num

        for model in order:
            demand = model.req_rate / model.slo
            best_gpu = None
            best_key = None

            for gpu_id in range(gpu_num):
                if model.model_size > remaining[gpu_id]:
                    continue

                new_remaining = remaining[gpu_id] - model.model_size
                new_score = pressure(loads[gpu_id] + demand, new_remaining)
                objective = max(
                    new_score,
                    max(scores[:gpu_id] + scores[gpu_id + 1:], default=0.0),
                )
                # On equal objective values, pack more tightly to preserve capacity.
                key = (objective, new_score, new_remaining)

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu_id

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            loads[best_gpu] += demand
            remaining[best_gpu] -= model.model_size
            scores[best_gpu] = pressure(loads[best_gpu], remaining[best_gpu])

        return placement, loads, remaining, scores

    model_pressure = lambda m: m.req_rate / m.slo
    orders = [
        # High-demand models should normally be balanced first.
        sorted(models, key=model_pressure, reverse=True),
        # Large models first improves the chance of preserving a feasible packing.
        sorted(models, key=lambda m: m.model_size, reverse=True),
        # A model which leaves little capacity is especially costly on a crowded GPU.
        sorted(
            models,
            key=lambda m: model_pressure(m) / max(1e-12, GPU_MEM_SIZE - m.model_size),
            reverse=True,
        ),
        # Resolve equal (or similar) demand in favor of the more restrictive model.
        sorted(
            models,
            key=lambda m: (model_pressure(m), m.model_size),
            reverse=True,
        ),
        # Conversely, prioritize memory-heavy models when their demand is comparable.
        sorted(
            models,
            key=lambda m: (m.model_size, model_pressure(m)),
            reverse=True,
        ),
        # This hybrid emphasizes models that have both high pressure and large size.
        sorted(
            models,
            key=lambda m: (
                model_pressure(m) / max(1e-12, GPU_MEM_SIZE - m.model_size),
                model_pressure(m),
                m.model_size,
            ),
            reverse=True,
        ),
    ]

    candidates = [result for result in (greedy(order) for order in orders) if result]
    if not candidates:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs with {GPU_MEM_SIZE} GB each."
        )

    placement, loads, remaining, scores = min(
        candidates, key=lambda result: max(result[3], default=0.0)
    )

    # Use both relocations and exchanges: a beneficial rebalance may require
    # freeing memory on the destination before a model can be transferred.
    while True:
        current_max = max(scores, default=0.0)
        best_action = None
        best_max = current_max

        for source in range(gpu_num):
            for model in placement[source]:
                demand = model_pressure(model)
                for target in range(gpu_num):
                    if source == target or model.model_size > remaining[target]:
                        continue

                    new_loads = loads[:]
                    new_remaining = remaining[:]
                    new_loads[source] -= demand
                    new_remaining[source] += model.model_size
                    new_loads[target] += demand
                    new_remaining[target] -= model.model_size
                    new_max = max(
                        pressure(new_loads[gpu], new_remaining[gpu])
                        for gpu in range(gpu_num)
                    )
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_action = (
                            "move", source, target, model, new_loads, new_remaining
                        )

        for first in range(gpu_num):
            for second in range(first + 1, gpu_num):
                for model_a in placement[first]:
                    demand_a = model_pressure(model_a)
                    for model_b in placement[second]:
                        demand_b = model_pressure(model_b)
                        if (model_b.model_size > remaining[first] + model_a.model_size
                                or model_a.model_size > remaining[second] + model_b.model_size):
                            continue

                        new_loads = loads[:]
                        new_remaining = remaining[:]
                        new_loads[first] += demand_b - demand_a
                        new_loads[second] += demand_a - demand_b
                        new_remaining[first] += model_a.model_size - model_b.model_size
                        new_remaining[second] += model_b.model_size - model_a.model_size
                        new_max = max(
                            pressure(new_loads[gpu], new_remaining[gpu])
                            for gpu in range(gpu_num)
                        )
                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_action = (
                                "swap", first, second, model_a, model_b,
                                new_loads, new_remaining,
                            )

        if best_action is None:
            break

        if best_action[0] == "move":
            _, source, target, model, loads, remaining = best_action
            placement[source].remove(model)
            placement[target].append(model)
        else:
            _, first, second, model_a, model_b, loads, remaining = best_action
            placement[first].remove(model_a)
            placement[second].remove(model_b)
            placement[first].append(model_b)
            placement[second].append(model_a)

        scores = [pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)]

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