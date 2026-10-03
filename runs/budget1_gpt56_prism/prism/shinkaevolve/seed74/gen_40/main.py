GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Minimize maximum KV cache pressure subject to GPU memory capacity.

    For a target pressure T:
        load / (80 - used_memory) <= T
    is equivalent to:
        sum(model_size + model_load / T) <= 80.

    A beam-search bin packer is used to explore multiple partial assignments
    at each target instead of relying on a single greedy packing order.
    """
    EPS = 1e-12
    BINARY_STEPS = 26
    BEAM_WIDTH = 18
    LOCAL_PASSES = 28

    if gpu_num <= 0:
        if models:
            raise ValueError("Unable to place models because gpu_num must be positive.")
        return {}

    n = len(models)
    if n == 0:
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

    def objective(loads, remaining):
        return max(
            pressure(loads[gpu], remaining[gpu])
            for gpu in range(gpu_num)
        )

    def state_from_assignment(assignment):
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num

        for model_id, gpu in enumerate(assignment):
            loads[gpu] += weights[model_id]
            remaining[gpu] -= sizes[model_id]

        return loads, remaining

    def fallback_pack(effective, order):
        """Best-fit decreasing fallback for threshold feasibility."""
        assignment = [-1] * n
        free = [float(GPU_MEM_SIZE)] * gpu_num

        for model_id in order:
            item = effective[model_id]
            choices = [
                gpu for gpu in range(gpu_num)
                if free[gpu] + EPS >= item
            ]

            if not choices:
                return None

            gpu = min(choices, key=lambda g: (free[g] - item, g))
            assignment[model_id] = gpu
            free[gpu] -= item

        loads, remaining = state_from_assignment(assignment)
        return assignment, loads, remaining

    def pack_at_target(target):
        """
        Search feasible effective-size packings with a compact beam of
        alternative partial assignments.
        """
        if target <= EPS:
            return None

        effective = [
            sizes[i] + weights[i] / target
            for i in range(n)
        ]

        if max(effective) > GPU_MEM_SIZE + EPS:
            return None

        if sum(effective) > gpu_num * GPU_MEM_SIZE + EPS:
            return None

        order = sorted(
            range(n),
            key=lambda i: (
                -effective[i],
                -sizes[i],
                -weights[i],
            ),
        )

        # Each state is:
        # (assignment, loads, remaining, effective_remaining)
        states = [
            (
                [-1] * n,
                [0.0] * gpu_num,
                [float(GPU_MEM_SIZE)] * gpu_num,
                [float(GPU_MEM_SIZE)] * gpu_num,
            )
        ]

        for model_id in order:
            item = effective[model_id]
            size = sizes[model_id]
            weight = weights[model_id]
            next_states = []
            seen = set()

            for assignment, loads, remaining, free in states:
                equivalent_bins = set()

                for gpu in range(gpu_num):
                    if free[gpu] + EPS < item:
                        continue

                    # GPUs with identical state are interchangeable.
                    bin_signature = (
                        round(free[gpu], 10),
                        round(remaining[gpu], 10),
                        round(loads[gpu], 10),
                    )
                    if bin_signature in equivalent_bins:
                        continue
                    equivalent_bins.add(bin_signature)

                    new_assignment = assignment[:]
                    new_loads = loads[:]
                    new_remaining = remaining[:]
                    new_free = free[:]

                    new_assignment[model_id] = gpu
                    new_loads[gpu] += weight
                    new_remaining[gpu] -= size
                    new_free[gpu] -= item

                    # Canonical GPU-state signature removes label permutations.
                    signature = tuple(sorted(
                        (
                            round(new_free[g], 8),
                            round(new_remaining[g], 8),
                            round(new_loads[g], 8),
                        )
                        for g in range(gpu_num)
                    ))

                    if signature in seen:
                        continue
                    seen.add(signature)

                    next_states.append(
                        (
                            new_assignment,
                            new_loads,
                            new_remaining,
                            new_free,
                        )
                    )

            if not next_states:
                return fallback_pack(effective, order)

            def beam_key(state):
                _, loads, remaining, free = state
                peak = objective(loads, remaining)

                # Packing effective space tightly leaves fewer fragmented bins.
                compactness = -sum(value * value for value in free)

                # Prefer balanced real pressures when compactness is equal.
                pressures = sorted(
                    (
                        pressure(loads[g], remaining[g])
                        for g in range(gpu_num)
                    ),
                    reverse=True,
                )

                return (peak, compactness, tuple(pressures))

            next_states.sort(key=beam_key)
            states = next_states[:BEAM_WIDTH]

        best = min(
            states,
            key=lambda state: objective(state[1], state[2]),
        )
        return best[0], best[1], best[2]

    def improve(assignment, loads, remaining):
        """Steepest strictly-improving relocation and swap descent."""
        for _ in range(LOCAL_PASSES):
            current = objective(loads, remaining)
            best_value = current
            best_action = None

            for model_id in range(n):
                source = assignment[model_id]
                size = sizes[model_id]
                weight = weights[model_id]

                for target in range(gpu_num):
                    if target == source or remaining[target] + EPS < size:
                        continue

                    old_source_load = loads[source]
                    old_target_load = loads[target]
                    old_source_remaining = remaining[source]
                    old_target_remaining = remaining[target]

                    loads[source] -= weight
                    remaining[source] += size
                    loads[target] += weight
                    remaining[target] -= size

                    value = objective(loads, remaining)

                    loads[source] = old_source_load
                    loads[target] = old_target_load
                    remaining[source] = old_source_remaining
                    remaining[target] = old_target_remaining

                    if value < best_value - EPS:
                        best_value = value
                        best_action = ("move", model_id, target)

            for first in range(n):
                first_gpu = assignment[first]

                for second in range(first + 1, n):
                    second_gpu = assignment[second]
                    if first_gpu == second_gpu:
                        continue

                    first_remaining = (
                        remaining[first_gpu] + sizes[first] - sizes[second]
                    )
                    second_remaining = (
                        remaining[second_gpu] + sizes[second] - sizes[first]
                    )

                    if first_remaining < -EPS or second_remaining < -EPS:
                        continue

                    old_first_load = loads[first_gpu]
                    old_second_load = loads[second_gpu]
                    old_first_remaining = remaining[first_gpu]
                    old_second_remaining = remaining[second_gpu]

                    loads[first_gpu] += weights[second] - weights[first]
                    loads[second_gpu] += weights[first] - weights[second]
                    remaining[first_gpu] = first_remaining
                    remaining[second_gpu] = second_remaining

                    value = objective(loads, remaining)

                    loads[first_gpu] = old_first_load
                    loads[second_gpu] = old_second_load
                    remaining[first_gpu] = old_first_remaining
                    remaining[second_gpu] = old_second_remaining

                    if value < best_value - EPS:
                        best_value = value
                        best_action = ("swap", first, second)

            if best_action is None:
                break

            if best_action[0] == "move":
                _, model_id, target = best_action
                source = assignment[model_id]

                assignment[model_id] = target
                loads[source] -= weights[model_id]
                remaining[source] += sizes[model_id]
                loads[target] += weights[model_id]
                remaining[target] -= sizes[model_id]
            else:
                _, first, second = best_action
                first_gpu = assignment[first]
                second_gpu = assignment[second]

                assignment[first], assignment[second] = second_gpu, first_gpu
                loads[first_gpu] += weights[second] - weights[first]
                loads[second_gpu] += weights[first] - weights[second]
                remaining[first_gpu] += sizes[first] - sizes[second]
                remaining[second_gpu] += sizes[second] - sizes[first]

        return assignment, loads, remaining

    total_size = sum(sizes)
    total_weight = sum(weights)
    total_free = gpu_num * GPU_MEM_SIZE - total_size

    low = total_weight / total_free if total_free > EPS else 0.0

    for size, weight in zip(sizes, weights):
        if weight > EPS and GPU_MEM_SIZE - size > EPS:
            low = max(low, weight / (GPU_MEM_SIZE - size))

    high = max(1.0, low * 2.0)
    initial = pack_at_target(high)

    while initial is None and high < 1e15:
        high *= 10.0
        initial = pack_at_target(high)

    if initial is None:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    candidates = [initial]

    for _ in range(BINARY_STEPS):
        middle = (low + high) / 2.0
        candidate = pack_at_target(middle)

        if candidate is None:
            low = middle
        else:
            high = middle
            candidates.append(candidate)

    best_assignment = None
    best_value = float("inf")

    for assignment, loads, remaining in candidates:
        assignment = assignment[:]
        loads = loads[:]
        remaining = remaining[:]

        assignment, loads, remaining = improve(
            assignment,
            loads,
            remaining,
        )

        value = objective(loads, remaining)
        if value < best_value - EPS:
            best_value = value
            best_assignment = assignment

    placement = {gpu: [] for gpu in range(gpu_num)}
    for model_id, gpu in enumerate(best_assignment):
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