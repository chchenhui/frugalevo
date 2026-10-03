GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement minimizing maximum KV cache pressure.

    Returns:
        Dictionary mapping GPU ids to lists of assigned models.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    model_list = list(models)
    count = len(model_list)
    sizes = []
    weights = []

    for model in model_list:
        if model.slo == 0:
            raise ValueError("model.slo must be non-zero")
        if model.model_size > GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on a "
                f"{GPU_MEM_SIZE} GB GPU"
            )
        sizes.append(float(model.model_size))
        weights.append(float(model.req_rate / model.slo))

    def pressure(load, remaining):
        if remaining > 0:
            return load / remaining
        return float("inf") if load > 0 else 0.0

    def score_vector(loads, remaining):
        return tuple(
            sorted(
                (pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)),
                reverse=True,
            )
        )

    def construct(order, pack_ties):
        groups = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        scores = [0.0] * gpu_num

        for item in order:
            size = sizes[item]
            weight = weights[item]
            best = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + 1e-12:
                    continue

                new_remaining = remaining[gpu] - size
                new_score = pressure(loads[gpu] + weight, new_remaining)
                other_max = max(
                    (scores[other] for other in range(gpu_num) if other != gpu),
                    default=0.0,
                )
                objective = max(new_score, other_max)

                # Packing preserves empty capacity for future large models;
                # spreading is useful when pressure balancing dominates.
                tie_memory = new_remaining if pack_ties else -new_remaining
                key = (objective, new_score, tie_memory, gpu)

                if best is None or key < best[0]:
                    best = (key, gpu)

            if best is None:
                return None

            gpu = best[1]
            groups[gpu].append(item)
            loads[gpu] += weight
            remaining[gpu] -= size
            scores[gpu] = pressure(loads[gpu], remaining[gpu])

        return groups, loads, remaining

    def improve(state):
        groups, loads, remaining = state
        epsilon = 1e-12
        iteration_limit = max(12, count * 4)

        for _ in range(iteration_limit):
            current_values = [
                pressure(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)
            ]
            current_key = tuple(sorted(current_values, reverse=True))
            current_max = current_key[0] if current_key else 0.0
            bottlenecks = [
                gpu
                for gpu in range(gpu_num)
                if current_values[gpu] >= current_max - epsilon
            ]

            best_key = current_key
            best_action = None

            def candidate_key(first, second, first_load_delta, first_mem_delta,
                              second_load_delta, second_mem_delta):
                values = list(current_values)
                values[first] = pressure(
                    loads[first] + first_load_delta,
                    remaining[first] + first_mem_delta,
                )
                values[second] = pressure(
                    loads[second] + second_load_delta,
                    remaining[second] + second_mem_delta,
                )
                return tuple(sorted(values, reverse=True))

            # Moves from bottleneck GPUs.
            for source in bottlenecks:
                for item in groups[source]:
                    size = sizes[item]
                    weight = weights[item]
                    for target in range(gpu_num):
                        if source == target or size > remaining[target] + epsilon:
                            continue

                        key = candidate_key(
                            source, target,
                            -weight, size,
                            weight, -size,
                        )
                        if key < best_key:
                            best_key = key
                            best_action = ("move", source, target, item)

            # One-for-one swaps.
            for left in range(gpu_num):
                for right in range(left + 1, gpu_num):
                    if left not in bottlenecks and right not in bottlenecks:
                        continue

                    for item_left in groups[left]:
                        for item_right in groups[right]:
                            left_after = (
                                remaining[left]
                                + sizes[item_left]
                                - sizes[item_right]
                            )
                            right_after = (
                                remaining[right]
                                + sizes[item_right]
                                - sizes[item_left]
                            )
                            if left_after < -epsilon or right_after < -epsilon:
                                continue

                            key = candidate_key(
                                left, right,
                                weights[item_right] - weights[item_left],
                                sizes[item_left] - sizes[item_right],
                                weights[item_left] - weights[item_right],
                                sizes[item_right] - sizes[item_left],
                            )
                            if key < best_key:
                                best_key = key
                                best_action = ("swap", left, right, item_left, item_right)

            # A one-for-two exchange is only needed after ordinary moves and
            # swaps cannot improve the lexicographic pressure vector.
            if best_action is None:
                for first in range(gpu_num):
                    for second in range(first + 1, gpu_num):
                        if first not in bottlenecks and second not in bottlenecks:
                            continue

                        for left, right in ((first, second), (second, first)):
                            right_items = groups[right]
                            for item_left in groups[left]:
                                for a in range(len(right_items)):
                                    for b in range(a + 1, len(right_items)):
                                        item_a = right_items[a]
                                        item_b = right_items[b]

                                        left_mem_delta = (
                                            sizes[item_left]
                                            - sizes[item_a]
                                            - sizes[item_b]
                                        )
                                        right_mem_delta = -left_mem_delta
                                        if (
                                            remaining[left] + left_mem_delta < -epsilon
                                            or remaining[right] + right_mem_delta < -epsilon
                                        ):
                                            continue

                                        right_weight = (
                                            weights[item_a] + weights[item_b]
                                        )
                                        key = candidate_key(
                                            left, right,
                                            right_weight - weights[item_left],
                                            left_mem_delta,
                                            weights[item_left] - right_weight,
                                            right_mem_delta,
                                        )
                                        if key < best_key:
                                            best_key = key
                                            best_action = (
                                                "exchange",
                                                left,
                                                right,
                                                item_left,
                                                item_a,
                                                item_b,
                                            )

            if best_action is None:
                break

            action = best_action[0]

            if action == "move":
                _, source, target, item = best_action
                groups[source].remove(item)
                groups[target].append(item)
                loads[source] -= weights[item]
                remaining[source] += sizes[item]
                loads[target] += weights[item]
                remaining[target] -= sizes[item]

            elif action == "swap":
                _, left, right, item_left, item_right = best_action
                left_index = groups[left].index(item_left)
                right_index = groups[right].index(item_right)
                groups[left][left_index] = item_right
                groups[right][right_index] = item_left

                loads[left] += weights[item_right] - weights[item_left]
                loads[right] += weights[item_left] - weights[item_right]
                remaining[left] += sizes[item_left] - sizes[item_right]
                remaining[right] += sizes[item_right] - sizes[item_left]

            else:
                _, left, right, item_left, item_a, item_b = best_action
                groups[left].remove(item_left)
                groups[right].remove(item_a)
                groups[right].remove(item_b)
                groups[left].extend((item_a, item_b))
                groups[right].append(item_left)

                right_weight = weights[item_a] + weights[item_b]
                loads[left] += right_weight - weights[item_left]
                loads[right] += weights[item_left] - right_weight

                memory_delta = sizes[item_left] - sizes[item_a] - sizes[item_b]
                remaining[left] += memory_delta
                remaining[right] -= memory_delta

        return groups, loads, remaining

    def risk(item):
        return weights[item] / max(1e-9, GPU_MEM_SIZE - sizes[item])

    indices = list(range(count))

    # Eight complementary seeds: pressure-heavy, capacity-heavy, density-heavy,
    # and objective-aware capacity-risk orderings.
    orders = [
        sorted(indices, key=lambda i: (weights[i], sizes[i]), reverse=True),
        sorted(indices, key=lambda i: (sizes[i], weights[i]), reverse=True),
        sorted(indices, key=lambda i: risk(i), reverse=True),
        sorted(indices, key=lambda i: weights[i] * sizes[i], reverse=True),
        sorted(
            indices,
            key=lambda i: weights[i] / max(sizes[i], 1e-9),
            reverse=True,
        ),
        sorted(indices, key=lambda i: (weights[i], -sizes[i]), reverse=True),
        sorted(indices, key=lambda i: (sizes[i], -weights[i]), reverse=True),
        sorted(
            indices,
            key=lambda i: (
                weights[i] / max(1e-9, GPU_MEM_SIZE - sizes[i]),
                weights[i],
                sizes[i],
            ),
            reverse=True,
        ),
    ]

    best_state = None
    best_score = float("inf")

    for order in orders:
        for pack_ties in (False, True):
            state = construct(order, pack_ties)
            if state is None:
                continue

            state = improve(state)
            value = score_vector(state[1], state[2])[0] if gpu_num else 0.0

            if value < best_score:
                best_score = value
                best_state = state

    if best_state is None:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB each"
        )

    groups = best_state[0]
    return {
        gpu: [model_list[item] for item in groups[gpu]]
        for gpu in range(gpu_num)
    }

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