GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a feasible placement minimizing the maximum KV cache pressure.

    KVPR(gpu) =
        sum(model.req_rate / model.slo) /
        (GPU_MEM_SIZE - sum(model.model_size))

    The algorithm uses several greedy initializations followed by bounded
    local search with single-model moves and pairwise model swaps.
    """
    EPSILON = 1e-12
    MAX_LOCAL_PASSES = 24

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

    def gpu_kvpr(load, remaining):
        if remaining <= EPSILON:
            return float("inf") if load > EPSILON else 0.0
        return load / remaining

    def objective(loads, remaining):
        return max(gpu_kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num))

    def objective_key(loads, remaining):
        # Lexicographic pressure balance: maximum KVPR remains the primary
        # objective, with lower-ranked GPU pressures breaking ties.
        return tuple(
            sorted(
                (gpu_kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)),
                reverse=True,
            )
        )

    def greedy_place(order, pack_ties):
        assignments = [-1] * model_count
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num

        for model_id in order:
            size = sizes[model_id]
            weight = weights[model_id]
            best_gpu = None
            best_key = None

            for gpu_id in range(gpu_num):
                if size > remaining[gpu_id] + EPSILON:
                    continue

                new_remaining = remaining[gpu_id] - size
                new_load = loads[gpu_id] + weight

                candidate_max = 0.0
                for other_gpu in range(gpu_num):
                    if other_gpu == gpu_id:
                        value = gpu_kvpr(new_load, new_remaining)
                    else:
                        value = gpu_kvpr(loads[other_gpu], remaining[other_gpu])
                    candidate_max = max(candidate_max, value)

                # Both configurations minimize post-placement maximum KVPR.
                # Packing ties reduces fragmentation; preserving ties leaves
                # room for later large models.
                tie_memory = new_remaining if pack_ties else -new_remaining
                candidate_key = (
                    candidate_max,
                    gpu_kvpr(new_load, new_remaining),
                    tie_memory,
                    gpu_id,
                )

                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_gpu = gpu_id

            if best_gpu is None:
                return None

            assignments[model_id] = best_gpu
            loads[best_gpu] += weight
            remaining[best_gpu] -= size

        return assignments, loads, remaining

    # Different orderings reduce sensitivity to greedy placement order.
    model_ids = list(range(model_count))
    orderings = [
        sorted(model_ids, key=lambda i: (-weights[i], -sizes[i])),
        sorted(model_ids, key=lambda i: (-sizes[i], -weights[i])),
        sorted(
            model_ids,
            key=lambda i: (
                -(weights[i] / max(sizes[i], EPSILON)),
                -sizes[i],
            ),
        ),
        # A model's isolated pressure reflects both its request load and the
        # KV-cache memory it leaves after its weights are loaded.
        sorted(
            model_ids,
            key=lambda i: (
                -(weights[i] / max(GPU_MEM_SIZE - sizes[i], EPSILON)),
                -sizes[i],
            ),
        ),
        # Prioritize large models whose standalone pressure is also high,
        # reducing fragmentation before smaller models are considered.
        sorted(
            model_ids,
            key=lambda i: (
                -(
                    weights[i]
                    * sizes[i]
                    / max(GPU_MEM_SIZE - sizes[i], EPSILON)
                ),
                -sizes[i],
            ),
        ),
    ]

    best_solution = None
    best_key = None

    for order in orderings:
        for pack_ties in (False, True):
            candidate = greedy_place(order, pack_ties)
            if candidate is None:
                continue

            assignments, loads, remaining = candidate
            candidate_key = objective_key(loads, remaining)

            if best_key is None or candidate_key < best_key:
                best_solution = (assignments, loads, remaining)
                best_key = candidate_key

    if best_solution is None:
        raise ValueError(
            "Unable to place all models on available GPUs. "
            "The models do not fit under the memory constraints."
        )

    assignments, loads, remaining = best_solution

    # Steepest-descent local search. Each pass chooses the best improving
    # single-model move or pairwise swap.
    for _ in range(MAX_LOCAL_PASSES):
        candidate_score = objective_key(loads, remaining)
        best_operation = None

        # Evaluate moving one model to another GPU.
        for model_id in range(model_count):
            source_gpu = assignments[model_id]
            size = sizes[model_id]
            weight = weights[model_id]

            for target_gpu in range(gpu_num):
                if target_gpu == source_gpu:
                    continue
                if size > remaining[target_gpu] + EPSILON:
                    continue

                loads[source_gpu] -= weight
                remaining[source_gpu] += size
                loads[target_gpu] += weight
                remaining[target_gpu] -= size

                score = objective_key(loads, remaining)

                loads[source_gpu] += weight
                remaining[source_gpu] -= size
                loads[target_gpu] -= weight
                remaining[target_gpu] += size

                if score < candidate_score:
                    candidate_score = score
                    best_operation = ("move", model_id, target_gpu)

        # Evaluate swapping models currently on different GPUs.
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

                new_first_remaining = remaining[first_gpu] + first_size - second_size
                new_second_remaining = remaining[second_gpu] + second_size - first_size

                if new_first_remaining < -EPSILON or new_second_remaining < -EPSILON:
                    continue

                loads[first_gpu] += second_weight - first_weight
                remaining[first_gpu] = new_first_remaining
                loads[second_gpu] += first_weight - second_weight
                remaining[second_gpu] = new_second_remaining

                score = objective_key(loads, remaining)

                loads[first_gpu] += first_weight - second_weight
                remaining[first_gpu] += second_size - first_size
                loads[second_gpu] += second_weight - first_weight
                remaining[second_gpu] += first_size - second_size

                if score < candidate_score:
                    candidate_score = score
                    best_operation = ("swap", first_id, second_id)

        if best_operation is None:
            break

        if best_operation[0] == "move":
            _, model_id, target_gpu = best_operation
            source_gpu = assignments[model_id]

            assignments[model_id] = target_gpu
            loads[source_gpu] -= weights[model_id]
            remaining[source_gpu] += sizes[model_id]
            loads[target_gpu] += weights[model_id]
            remaining[target_gpu] -= sizes[model_id]
        else:
            _, first_id, second_id = best_operation
            first_gpu = assignments[first_id]
            second_gpu = assignments[second_id]

            assignments[first_id], assignments[second_id] = second_gpu, first_gpu

            loads[first_gpu] += weights[second_id] - weights[first_id]
            remaining[first_gpu] += sizes[first_id] - sizes[second_id]

            loads[second_gpu] += weights[first_id] - weights[second_id]
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