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

        # A relocation local optimum can still be improved by exchanging models
    # between two GPUs.  Swaps are especially useful when memory prevents a
    # beneficial one-way move.
    while True:
        current_max = max(scores, default=0.0)
        best_swap = None
        best_max = current_max

        for first_gpu in range(gpu_num):
            for second_gpu in range(first_gpu + 1, gpu_num):
                for first_model in placement[first_gpu]:
                    first_demand = model_pressure(first_model)
                    for second_model in placement[second_gpu]:
                        second_demand = model_pressure(second_model)

                        first_remaining = (
                            remaining[first_gpu]
                            + first_model.model_size
                            - second_model.model_size
                        )
                        second_remaining = (
                            remaining[second_gpu]
                            + second_model.model_size
                            - first_model.model_size
                        )
                        if first_remaining < 0 or second_remaining < 0:
                            continue

                        new_loads = loads[:]
                        new_remaining = remaining[:]
                        new_loads[first_gpu] += second_demand - first_demand
                        new_loads[second_gpu] += first_demand - second_demand
                        new_remaining[first_gpu] = first_remaining
                        new_remaining[second_gpu] = second_remaining
                        new_max = max(
                            pressure(new_loads[gpu], new_remaining[gpu])
                            for gpu in range(gpu_num)
                        )

                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_swap = (
                                first_gpu,
                                second_gpu,
                                first_model,
                                second_model,
                                new_loads,
                                new_remaining,
                            )

        if best_swap is None:
            break

        (
            first_gpu,
            second_gpu,
            first_model,
            second_model,
            loads,
            remaining,
        ) = best_swap
        placement[first_gpu].remove(first_model)
        placement[second_gpu].remove(second_model)
        placement[first_gpu].append(second_model)
        placement[second_gpu].append(first_model)
        scores = [pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)]

    return placement, loads, remaining, scores

    model_pressure = lambda m: m.req_rate / m.slo
    orders = [
        sorted(models, key=model_pressure, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(
            models,
            key=lambda m: model_pressure(m) / max(1e-12, GPU_MEM_SIZE - m.model_size),
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

    # Move a model whenever doing so strictly reduces the global maximum KVPR.
    while True:
        current_max = max(scores, default=0.0)
        best_move = None
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
                        best_move = (source, target, model, new_loads, new_remaining)

        if best_move is None:
            break

        source, target, model, loads, remaining = best_move
        placement[source].remove(model)
        placement[target].append(model)
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