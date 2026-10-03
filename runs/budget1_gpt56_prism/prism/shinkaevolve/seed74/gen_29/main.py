GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models while minimizing maximum KV cache pressure.

    For a target pressure T, a GPU containing models i is valid when:
        sum(weight_i) / (80 - sum(size_i)) <= T

    This is equivalent to:
        sum(size_i + weight_i / T) <= 80

    Thus, binary search on T with effective-size bin packing gives a strong
    initial placement, followed by strictly improving move/swap local search.
    """
    EPSILON = 1e-12
    BINARY_SEARCH_STEPS = 32
    MAX_LOCAL_PASSES = 36

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    model_count = len(models)
    if model_count == 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    sizes = [float(model.model_size) for model in models]
    weights = [float(model.req_rate) / float(model.slo) for model in models]

    for size in sizes:
        if size > GPU_MEM_SIZE + EPSILON:
            raise ValueError(
                f"Unable to place model of size {size} GB: "
                f"it exceeds GPU memory of {GPU_MEM_SIZE} GB."
            )

    if sum(sizes) > gpu_num * GPU_MEM_SIZE + EPSILON:
        raise ValueError(
            "Unable to place all models on available GPUs. "
            "Total model memory exceeds total GPU memory."
        )

    def pressure(load, remaining):
        if remaining <= EPSILON:
            return float("inf") if load > EPSILON else 0.0
        return load / remaining

    def score_key(loads, remaining):
        values = sorted(
            (pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)),
            reverse=True,
        )
        return tuple(values)

    def better(candidate, current):
        """Strict objective improvement, then lexicographic pressure balance."""
        if candidate[0] < current[0] - EPSILON:
            return True
        if abs(candidate[0] - current[0]) <= EPSILON:
            return candidate[1:] < current[1:]
        return False

    def assignment_state(assignments):
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        for model_id, gpu_id in enumerate(assignments):
            loads[gpu_id] += weights[model_id]
            remaining[gpu_id] -= sizes[model_id]
        return loads, remaining

    def pack_for_threshold(target):
        """
        Try several decreasing bin-packing orders at a fixed pressure target.
        Returns the feasible assignment having the lowest actual KVPR.
        """
        if target <= EPSILON:
            return None

        effective = [
            sizes[i] + weights[i] / target
            for i in range(model_count)
        ]

        if max(effective) > GPU_MEM_SIZE + EPSILON:
            return None

        model_ids = list(range(model_count))
        orders = [
            sorted(model_ids, key=lambda i: (-effective[i], -sizes[i], -weights[i])),
            sorted(model_ids, key=lambda i: (-effective[i], -weights[i], -sizes[i])),
            sorted(
                model_ids,
                key=lambda i: (
                    -effective[i],
                    -(weights[i] / max(sizes[i], EPSILON)),
                    -sizes[i],
                ),
            ),
        ]

        best = None
        best_key = None

        for order_index, order in enumerate(orders):
            assignments = [-1] * model_count
            effective_remaining = [float(GPU_MEM_SIZE)] * gpu_num

            for model_id in order:
                item_size = effective[model_id]
                feasible_gpus = [
                    gpu_id
                    for gpu_id in range(gpu_num)
                    if effective_remaining[gpu_id] + EPSILON >= item_size
                ]

                if not feasible_gpus:
                    assignments = None
                    break

                if order_index == 1:
                    # First-fit decreasing preserves larger empty regions.
                    chosen_gpu = feasible_gpus[0]
                else:
                    # Best-fit decreasing reduces effective-memory fragmentation.
                    chosen_gpu = min(
                        feasible_gpus,
                        key=lambda gpu_id: (
                            effective_remaining[gpu_id] - item_size,
                            gpu_id,
                        ),
                    )

                assignments[model_id] = chosen_gpu
                effective_remaining[chosen_gpu] -= item_size

            if assignments is None:
                continue

            loads, remaining = assignment_state(assignments)
            key = score_key(loads, remaining)

            if best is None or better(key, best_key):
                best = (assignments, loads, remaining)
                best_key = key

        return best

    # A finite upper pressure bound. At this large threshold, effective sizes
    # are essentially regular model sizes, so this also validates feasibility.
    high = 1.0
    initial = pack_for_threshold(high)

    while initial is None and high < 1e15:
        high *= 10.0
        initial = pack_for_threshold(high)

    if initial is None:
        raise ValueError(
            "Unable to place all models on available GPUs under memory constraints."
        )

    # A lower bound from aggregate memory/load and each individual model.
    total_size = sum(sizes)
    total_weight = sum(weights)
    free_memory = gpu_num * GPU_MEM_SIZE - total_size

    if free_memory > EPSILON and total_weight > EPSILON:
        low = total_weight / free_memory
    else:
        low = 0.0

    for size, weight in zip(sizes, weights):
        if weight > EPSILON and GPU_MEM_SIZE - size > EPSILON:
            low = max(low, weight / (GPU_MEM_SIZE - size))

    best_solution = initial

    # Binary-search the lowest threshold for which effective-size bin packing
    # finds a feasible placement.
    for _ in range(BINARY_SEARCH_STEPS):
        middle = (low + high) / 2.0
        candidate = pack_for_threshold(middle)

        if candidate is not None:
            high = middle
            best_solution = candidate
        else:
            low = middle

    assignments, loads, remaining = best_solution

    # Strictly improving local search. Pairwise swaps overcome cases where
    # useful relocations are blocked by memory fragmentation.
    for _ in range(MAX_LOCAL_PASSES):
        current_key = score_key(loads, remaining)
        best_key = current_key
        best_operation = None

        # Single-model relocations.
        for model_id in range(model_count):
            source = assignments[model_id]
            size = sizes[model_id]
            weight = weights[model_id]

            for target in range(gpu_num):
                if target == source or size > remaining[target] + EPSILON:
                    continue

                loads[source] -= weight
                remaining[source] += size
                loads[target] += weight
                remaining[target] -= size

                candidate_key = score_key(loads, remaining)

                loads[source] += weight
                remaining[source] -= size
                loads[target] -= weight
                remaining[target] += size

                if better(candidate_key, best_key):
                    best_key = candidate_key
                    best_operation = ("move", model_id, target)

        # Pairwise exchanges.
        for first_id in range(model_count):
            first_gpu = assignments[first_id]
            first_size = sizes[first_id]
            first_weight = weights[first_id]

            for second_id in range(first_id + 1, model_count):
                second_gpu = assignments[second_id]
                if first_gpu == second_gpu:
                    continue

                second_size = sizes[second_id]
                second_weight = weights[second_id]

                first_remaining = remaining[first_gpu] + first_size - second_size
                second_remaining = remaining[second_gpu] + second_size - first_size

                if first_remaining < -EPSILON or second_remaining < -EPSILON:
                    continue

                old_first_remaining = remaining[first_gpu]
                old_second_remaining = remaining[second_gpu]

                loads[first_gpu] += second_weight - first_weight
                loads[second_gpu] += first_weight - second_weight
                remaining[first_gpu] = first_remaining
                remaining[second_gpu] = second_remaining

                candidate_key = score_key(loads, remaining)

                loads[first_gpu] += first_weight - second_weight
                loads[second_gpu] += second_weight - first_weight
                remaining[first_gpu] = old_first_remaining
                remaining[second_gpu] = old_second_remaining

                if better(candidate_key, best_key):
                    best_key = candidate_key
                    best_operation = ("swap", first_id, second_id)

        if best_operation is None:
            break

        if best_operation[0] == "move":
            _, model_id, target = best_operation
            source = assignments[model_id]

            assignments[model_id] = target
            loads[source] -= weights[model_id]
            remaining[source] += sizes[model_id]
            loads[target] += weights[model_id]
            remaining[target] -= sizes[model_id]
        else:
            _, first_id, second_id = best_operation
            first_gpu = assignments[first_id]
            second_gpu = assignments[second_id]

            assignments[first_id], assignments[second_id] = second_gpu, first_gpu

            loads[first_gpu] += weights[second_id] - weights[first_id]
            loads[second_gpu] += weights[first_id] - weights[second_id]
            remaining[first_gpu] += sizes[first_id] - sizes[second_id]
            remaining[second_gpu] += sizes[second_id] - sizes[first_id]

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    for model_id, gpu_id in enumerate(assignments):
        placement[gpu_id].append(models[model_id])

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