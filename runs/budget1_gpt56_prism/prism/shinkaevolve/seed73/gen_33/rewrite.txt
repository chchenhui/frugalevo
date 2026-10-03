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

        size = float(model.model_size)
        if size > GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on a "
                f"{GPU_MEM_SIZE} GB GPU"
            )

        sizes.append(size)
        weights.append(float(model.req_rate / model.slo))

    def pressure(load, remaining):
        if remaining > 0:
            return load / remaining
        return float("inf") if load > 0 else 0.0

    def pressure_key(loads, remaining):
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

        for item in order:
            size = sizes[item]
            weight = weights[item]
            current = [
                pressure(loads[gpu], remaining[gpu])
                for gpu in range(gpu_num)
            ]
            best = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + 1e-12:
                    continue

                values = list(current)
                after_remaining = remaining[gpu] - size
                values[gpu] = pressure(loads[gpu] + weight, after_remaining)

                # Compare complete pressure vectors so tied bottlenecks are
                # balanced before they become the next maximum-pressure GPU.
                objective = tuple(sorted(values, reverse=True))
                tie_memory = after_remaining if pack_ties else -after_remaining
                candidate = (objective, tie_memory, gpu)

                if best is None or candidate < best:
                    best = candidate

            if best is None:
                return None

            gpu = best[2]
            groups[gpu].append(item)
            loads[gpu] += weight
            remaining[gpu] -= size

        return groups, loads, remaining

    def improve(state):
        groups, loads, remaining = state
        epsilon = 1e-12
        limit = max(12, count * 4)

        for _ in range(limit):
            values = [
                pressure(loads[gpu], remaining[gpu])
                for gpu in range(gpu_num)
            ]
            current_key = tuple(sorted(values, reverse=True))
            current_max = current_key[0] if current_key else 0.0
            bottlenecks = {
                gpu
                for gpu in range(gpu_num)
                if values[gpu] >= current_max - epsilon
            }

            best_key = current_key
            best_action = None

            def key_after(left, left_load_delta, left_mem_delta,
                          right, right_load_delta, right_mem_delta):
                candidate = list(values)
                candidate[left] = pressure(
                    loads[left] + left_load_delta,
                    remaining[left] + left_mem_delta,
                )
                candidate[right] = pressure(
                    loads[right] + right_load_delta,
                    remaining[right] + right_mem_delta,
                )
                return tuple(sorted(candidate, reverse=True))

            # Move an item away from a current pressure bottleneck.
            for source in bottlenecks:
                for item in groups[source]:
                    size = sizes[item]
                    weight = weights[item]

                    for target in range(gpu_num):
                        if source == target or size > remaining[target] + epsilon:
                            continue

                        candidate = key_after(
                            source, -weight, size,
                            target, weight, -size,
                        )
                        if candidate < best_key:
                            best_key = candidate
                            best_action = ("move", source, target, item)

            # Exchange one item between GPU pairs when a direct move is blocked.
            for left in range(gpu_num):
                for right in range(left + 1, gpu_num):
                    if left not in bottlenecks and right not in bottlenecks:
                        continue

                    for item_left in groups[left]:
                        for item_right in groups[right]:
                            left_memory_delta = (
                                sizes[item_left] - sizes[item_right]
                            )
                            right_memory_delta = -left_memory_delta

                            if (
                                remaining[left] + left_memory_delta < -epsilon
                                or remaining[right] + right_memory_delta < -epsilon
                            ):
                                continue

                            candidate = key_after(
                                left,
                                weights[item_right] - weights[item_left],
                                left_memory_delta,
                                right,
                                weights[item_left] - weights[item_right],
                                right_memory_delta,
                            )
                            if candidate < best_key:
                                best_key = candidate
                                best_action = (
                                    "swap", left, right, item_left, item_right
                                )

            # A compound exchange can resolve memory fragmentation that prevents
            # ordinary moves and one-for-one swaps.
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

                                        left_memory_delta = (
                                            sizes[item_left]
                                            - sizes[item_a]
                                            - sizes[item_b]
                                        )
                                        right_memory_delta = -left_memory_delta

                                        if (
                                            remaining[left] + left_memory_delta < -epsilon
                                            or remaining[right] + right_memory_delta < -epsilon
                                        ):
                                            continue

                                        right_weight = (
                                            weights[item_a] + weights[item_b]
                                        )
                                        candidate = key_after(
                                            left,
                                            right_weight - weights[item_left],
                                            left_memory_delta,
                                            right,
                                            weights[item_left] - right_weight,
                                            right_memory_delta,
                                        )
                                        if candidate < best_key:
                                            best_key = candidate
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

                exchanged_weight = weights[item_a] + weights[item_b]
                loads[left] += exchanged_weight - weights[item_left]
                loads[right] += weights[item_left] - exchanged_weight

                memory_delta = sizes[item_left] - sizes[item_a] - sizes[item_b]
                remaining[left] += memory_delta
                remaining[right] -= memory_delta

        return groups, loads, remaining

    def risk(item):
        return weights[item] / max(1e-9, GPU_MEM_SIZE - sizes[item])

    indices = list(range(count))

    # A small diverse portfolio covers pressure-heavy, capacity-heavy, density,
    # and memory-risk cases without excessive construction overhead.
    orders = [
        sorted(indices, key=lambda i: (weights[i], sizes[i]), reverse=True),
        sorted(indices, key=lambda i: (sizes[i], weights[i]), reverse=True),
        sorted(indices, key=lambda i: weights[i] * sizes[i], reverse=True),
        sorted(
            indices,
            key=lambda i: weights[i] / max(sizes[i], 1e-9),
            reverse=True,
        ),
        sorted(indices, key=lambda i: (sizes[i], -weights[i]), reverse=True),
        sorted(indices, key=lambda i: (weights[i], -sizes[i]), reverse=True),
        sorted(indices, key=risk, reverse=True),
        sorted(
            indices,
            key=lambda i: (
                int(sizes[i] / 20.0),
                weights[i],
                sizes[i],
            ),
            reverse=True,
        ),
    ]

    best_state = None
    best_key = None

    for order in orders:
        for pack_ties in (False, True):
            state = construct(order, pack_ties)
            if state is None:
                continue

            state = improve(state)
            candidate_key = pressure_key(state[1], state[2])

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
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