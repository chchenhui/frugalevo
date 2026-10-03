GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models feasibly while minimizing the maximum GPU KV cache pressure.

    A target pressure T is feasible for a GPU when:
        load / remaining_memory <= T
    which is equivalent to packing each model with effective size:
        model_size + model_load / T
    into capacity GPU_MEM_SIZE.
    """
    EPS = 1e-12
    BINARY_STEPS = 30
    MAX_LOCAL_PASSES = 40

    if gpu_num <= 0:
        if models:
            raise ValueError("Unable to place models because gpu_num must be positive.")
        return {}

    model_count = len(models)
    if model_count == 0:
        return {gpu: [] for gpu in range(gpu_num)}

    sizes = [float(model.model_size) for model in models]
    weights = [float(model.req_rate) / float(model.slo) for model in models]

    if any(size > GPU_MEM_SIZE + EPS for size in sizes):
        raise ValueError(
            f"Unable to place a model exceeding GPU memory of {GPU_MEM_SIZE} GB."
        )

    if sum(sizes) > gpu_num * GPU_MEM_SIZE + EPS:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    def pressure(load, remaining):
        if remaining <= EPS:
            return float("inf") if load > EPS else 0.0
        return load / remaining

    def score(loads, remaining):
        return max(pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num))

    def state_from_assignments(assignments):
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        for model_id, gpu in enumerate(assignments):
            loads[gpu] += weights[model_id]
            remaining[gpu] -= sizes[model_id]
        return loads, remaining

    def greedy_assign(order, pack_ties=False):
        assignments = [-1] * model_count
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num

        for model_id in order:
            size = sizes[model_id]
            weight = weights[model_id]
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + EPS:
                    continue

                next_remaining = remaining[gpu] - size
                next_pressure = pressure(loads[gpu] + weight, next_remaining)

                other_peak = 0.0
                for other in range(gpu_num):
                    if other != gpu:
                        other_peak = max(
                            other_peak,
                            pressure(loads[other], remaining[other]),
                        )

                peak = max(other_peak, next_pressure)
                tie_memory = next_remaining if pack_ties else -next_remaining
                key = (peak, next_pressure, tie_memory, gpu)

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            assignments[model_id] = best_gpu
            loads[best_gpu] += weight
            remaining[best_gpu] -= size

        return assignments, loads, remaining

    def pack_threshold(target):
        if target <= EPS:
            return None

        effective = [sizes[i] + weights[i] / target for i in range(model_count)]
        if max(effective) > GPU_MEM_SIZE + EPS:
            return None

        ids = list(range(model_count))
        orders = [
            sorted(ids, key=lambda i: (-effective[i], -sizes[i], -weights[i])),
            sorted(ids, key=lambda i: (-effective[i], -weights[i], -sizes[i])),
            sorted(
                ids,
                key=lambda i: (
                    -effective[i],
                    -(weights[i] / max(sizes[i], EPS)),
                    -sizes[i],
                ),
            ),
        ]

        best = None
        best_value = float("inf")

        for order_index, order in enumerate(orders):
            assignments = [-1] * model_count
            free = [float(GPU_MEM_SIZE)] * gpu_num

            for model_id in order:
                item = effective[model_id]
                feasible = [
                    gpu for gpu in range(gpu_num)
                    if free[gpu] + EPS >= item
                ]
                if not feasible:
                    assignments = None
                    break

                if order_index == 1:
                    chosen = feasible[0]
                else:
                    chosen = min(
                        feasible,
                        key=lambda gpu: (free[gpu] - item, gpu),
                    )

                assignments[model_id] = chosen
                free[chosen] -= item

            if assignments is not None:
                loads, remaining = state_from_assignments(assignments)
                value = score(loads, remaining)
                if value < best_value - EPS:
                    best = (assignments, loads, remaining)
                    best_value = value

        return best

    def improve(assignments, loads, remaining):
        """Steepest descent with feasible relocations and pairwise swaps."""
        for _ in range(MAX_LOCAL_PASSES):
            current = score(loads, remaining)
            best_value = current
            best_action = None

            for model_id in range(model_count):
                source = assignments[model_id]
                size = sizes[model_id]
                weight = weights[model_id]

                for target in range(gpu_num):
                    if target == source or size > remaining[target] + EPS:
                        continue

                    source_value = pressure(
                        loads[source] - weight,
                        remaining[source] + size,
                    )
                    target_value = pressure(
                        loads[target] + weight,
                        remaining[target] - size,
                    )

                    candidate = max(source_value, target_value)
                    for gpu in range(gpu_num):
                        if gpu != source and gpu != target:
                            candidate = max(
                                candidate,
                                pressure(loads[gpu], remaining[gpu]),
                            )

                    if candidate < best_value - EPS:
                        best_value = candidate
                        best_action = ("move", model_id, target)

            for first in range(model_count):
                first_gpu = assignments[first]
                first_size = sizes[first]
                first_weight = weights[first]

                for second in range(first + 1, model_count):
                    second_gpu = assignments[second]
                    if first_gpu == second_gpu:
                        continue

                    second_size = sizes[second]
                    second_weight = weights[second]

                    first_remaining = (
                        remaining[first_gpu] + first_size - second_size
                    )
                    second_remaining = (
                        remaining[second_gpu] + second_size - first_size
                    )

                    if first_remaining < -EPS or second_remaining < -EPS:
                        continue

                    first_value = pressure(
                        loads[first_gpu] - first_weight + second_weight,
                        first_remaining,
                    )
                    second_value = pressure(
                        loads[second_gpu] - second_weight + first_weight,
                        second_remaining,
                    )

                    candidate = max(first_value, second_value)
                    for gpu in range(gpu_num):
                        if gpu != first_gpu and gpu != second_gpu:
                            candidate = max(
                                candidate,
                                pressure(loads[gpu], remaining[gpu]),
                            )

                    if candidate < best_value - EPS:
                        best_value = candidate
                        best_action = ("swap", first, second)

            if best_action is None:
                break

            if best_action[0] == "move":
                _, model_id, target = best_action
                source = assignments[model_id]
                assignments[model_id] = target
                loads[source] -= weights[model_id]
                remaining[source] += sizes[model_id]
                loads[target] += weights[model_id]
                remaining[target] -= sizes[model_id]
            else:
                _, first, second = best_action
                first_gpu = assignments[first]
                second_gpu = assignments[second]

                assignments[first], assignments[second] = second_gpu, first_gpu
                loads[first_gpu] += weights[second] - weights[first]
                loads[second_gpu] += weights[first] - weights[second]
                remaining[first_gpu] += sizes[first] - sizes[second]
                remaining[second_gpu] += sizes[second] - sizes[first]

        return assignments, loads, remaining

    total_size = sum(sizes)
    total_weight = sum(weights)
    free_memory = gpu_num * GPU_MEM_SIZE - total_size

    low = total_weight / free_memory if free_memory > EPS else 0.0
    for size, weight in zip(sizes, weights):
        if weight > EPS and GPU_MEM_SIZE - size > EPS:
            low = max(low, weight / (GPU_MEM_SIZE - size))

    high = max(1.0, low * 2.0)
    initial = pack_threshold(high)
    while initial is None and high < 1e15:
        high *= 10.0
        initial = pack_threshold(high)

    if initial is None:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    candidates = [initial]

    for _ in range(BINARY_STEPS):
        middle = (low + high) / 2.0
        candidate = pack_threshold(middle)
        if candidate is None:
            low = middle
        else:
            high = middle
            candidates.append(candidate)

    ids = list(range(model_count))
    greedy_orders = [
        sorted(ids, key=lambda i: (-weights[i], -sizes[i])),
        sorted(ids, key=lambda i: (-sizes[i], -weights[i])),
        sorted(
            ids,
            key=lambda i: (
                -(weights[i] / max(GPU_MEM_SIZE - sizes[i], EPS)),
                -sizes[i],
            ),
        ),
        sorted(
            ids,
            key=lambda i: (
                -(weights[i] / max(sizes[i], EPS)),
                -sizes[i],
            ),
        ),
    ]

    for index, order in enumerate(greedy_orders):
        candidate = greedy_assign(order, pack_ties=(index == 1))
        if candidate is not None:
            candidates.append(candidate)

    best = None
    best_value = float("inf")

    for assignments, loads, remaining in candidates:
        assignments = assignments[:]
        loads = loads[:]
        remaining = remaining[:]
        assignments, loads, remaining = improve(assignments, loads, remaining)
        value = score(loads, remaining)

        if value < best_value - EPS:
            best = assignments
            best_value = value

    placement = {gpu: [] for gpu in range(gpu_num)}
    for model_id, gpu in enumerate(best):
        placement[gpu].append(models[model_id])

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