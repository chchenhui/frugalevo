GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models while minimizing the maximum KV cache pressure.

    A GPU with model load L and used model memory M has:
        KVPR = L / (GPU_MEM_SIZE - M)

    For a candidate pressure T, KVPR <= T is equivalent to:
        L + T * M <= T * GPU_MEM_SIZE

    This permits a transformed two-dimensional packing search using both
    physical model memory and pressure-adjusted capacity.
    """
    EPS = 1e-10

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    count = len(models)
    if count == 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    sizes = [float(model.model_size) for model in models]
    loads_per_model = [
        float(model.req_rate) / float(model.slo) for model in models
    ]

    for size in sizes:
        if size > GPU_MEM_SIZE + EPS:
            raise ValueError(
                f"Unable to place model of size {size} GB: "
                f"it exceeds GPU memory of {GPU_MEM_SIZE} GB."
            )

    ids = list(range(count))

    def pressure(load, remaining):
        if remaining <= EPS:
            return float("inf") if load > EPS else 0.0
        return load / remaining

    def score(loads, remaining):
        values = sorted(
            (pressure(loads[g], remaining[g]) for g in range(gpu_num)),
            reverse=True,
        )
        return (
            values[0],
            values[1] if gpu_num > 1 else 0.0,
            values[2] if gpu_num > 2 else 0.0,
            -min(remaining),
        )

    def build_from_order(order, threshold=None, pack_memory=False):
        """
        If threshold is None, greedily minimize the real post-placement KVPR.
        Otherwise use transformed pressure packing:
            load + threshold * used_memory <= threshold * GPU_MEM_SIZE.
        """
        assignment = [-1] * count
        gpu_loads = [0.0] * gpu_num
        remaining = [float(GPU_MEM_SIZE)] * gpu_num

        transformed_capacity = threshold * GPU_MEM_SIZE if threshold else 0.0

        for item in order:
            size = sizes[item]
            item_load = loads_per_model[item]
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + EPS:
                    continue

                new_remaining = remaining[gpu] - size
                new_load = gpu_loads[gpu] + item_load

                if threshold is not None:
                    used_memory = GPU_MEM_SIZE - new_remaining
                    transformed = new_load + threshold * used_memory

                    if transformed > transformed_capacity + EPS:
                        continue

                    max_fill = 0.0
                    for other in range(gpu_num):
                        if other == gpu:
                            fill = transformed / max(transformed_capacity, EPS)
                        else:
                            old_used = GPU_MEM_SIZE - remaining[other]
                            old_transformed = (
                                gpu_loads[other] + threshold * old_used
                            )
                            fill = old_transformed / max(transformed_capacity, EPS)
                        if fill > max_fill:
                            max_fill = fill

                    # Use transformed balance first.  Alternate between
                    # preserving room and compacting memory to reduce
                    # sensitivity to greedy ordering.
                    memory_tie = new_remaining if pack_memory else -new_remaining
                    key = (
                        max_fill,
                        pressure(new_load, new_remaining),
                        memory_tie,
                        gpu,
                    )
                else:
                    maximum = 0.0
                    for other in range(gpu_num):
                        if other == gpu:
                            value = pressure(new_load, new_remaining)
                        else:
                            value = pressure(gpu_loads[other], remaining[other])
                        maximum = max(maximum, value)

                    memory_tie = new_remaining if pack_memory else -new_remaining
                    key = (
                        maximum,
                        pressure(new_load, new_remaining),
                        memory_tie,
                        gpu,
                    )

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            assignment[item] = best_gpu
            gpu_loads[best_gpu] += item_load
            remaining[best_gpu] -= size

        return assignment, gpu_loads, remaining

    orders = [
        sorted(ids, key=lambda i: (-sizes[i], -loads_per_model[i])),
        sorted(ids, key=lambda i: (-loads_per_model[i], -sizes[i])),
        sorted(
            ids,
            key=lambda i: (
                -(loads_per_model[i] / max(GPU_MEM_SIZE - sizes[i], EPS)),
                -sizes[i],
            ),
        ),
        sorted(
            ids,
            key=lambda i: (
                -(loads_per_model[i] / max(sizes[i], EPS)),
                -sizes[i],
            ),
        ),
        sorted(
            ids,
            key=lambda i: (
                -(loads_per_model[i] * sizes[i]),
                -sizes[i],
            ),
        ),
    ]

    # First obtain a robust conventional feasible solution.
    best = None
    best_value = None
    for order in orders:
        for pack_memory in (False, True):
            candidate = build_from_order(order, None, pack_memory)
            if candidate is None:
                continue
            value = score(candidate[1], candidate[2])
            if best_value is None or value < best_value:
                best = candidate
                best_value = value

    if best is None:
        raise ValueError(
            "Unable to place all models on available GPUs. "
            "The models do not fit under the memory constraints."
        )

    # Search for lower feasible transformed-pressure packings.  Every
    # successful ordering is evaluated by its actual, not transformed, KVPR.
    upper = best_value[0]
    lower = 0.0

    if upper != float("inf") and upper > EPS:
        for _ in range(18):
            threshold = (lower + upper) / 2.0
            threshold_best = None
            threshold_value = None

            for order in orders:
                for pack_memory in (False, True):
                    candidate = build_from_order(order, threshold, pack_memory)
                    if candidate is None:
                        continue
                    value = score(candidate[1], candidate[2])
                    if threshold_value is None or value < threshold_value:
                        threshold_best = candidate
                        threshold_value = value

            if threshold_best is None:
                lower = threshold
            else:
                # The transformed threshold can be looser than the actual
                # maximum KVPR of the constructed placement.  Repack at that
                # measured pressure before continuing the search.
                tightened = min(threshold, threshold_value[0])
                tightened_best = threshold_best
                tightened_value = threshold_value

                if tightened < threshold - EPS:
                    for order in orders:
                        for pack_memory in (False, True):
                            candidate = build_from_order(
                                order, tightened, pack_memory
                            )
                            if candidate is None:
                                continue
                            value = score(candidate[1], candidate[2])
                            if value < tightened_value:
                                tightened_best = candidate
                                tightened_value = value

                # A successful placement itself proves that its measured
                # pressure is feasible, even if the greedy retry found no
                # alternate arrangement at that exact bound.
                upper = tightened
                if tightened_value < best_value:
                    best = tightened_best
                    best_value = tightened_value

    assignment, gpu_loads, remaining = best

    def current_score():
        return score(gpu_loads, remaining)

    # Steepest descent over moves and exchanges.  The transformed phase finds
    # balanced global layouts; these operations remove residual fragmentation.
    for _ in range(32):
        old_value = current_score()
        candidate_value = old_value
        operation = None

        for item in range(count):
            source = assignment[item]
            size = sizes[item]
            item_load = loads_per_model[item]

            for target in range(gpu_num):
                if target == source or size > remaining[target] + EPS:
                    continue

                gpu_loads[source] -= item_load
                remaining[source] += size
                gpu_loads[target] += item_load
                remaining[target] -= size

                value = current_score()

                gpu_loads[source] += item_load
                remaining[source] -= size
                gpu_loads[target] -= item_load
                remaining[target] += size

                if value < candidate_value:
                    candidate_value = value
                    operation = ("move", item, target)

        for first in range(count):
            first_gpu = assignment[first]
            first_size = sizes[first]
            first_load = loads_per_model[first]

            for second in range(first + 1, count):
                second_gpu = assignment[second]
                if first_gpu == second_gpu:
                    continue

                second_size = sizes[second]
                second_load = loads_per_model[second]

                new_first_remaining = (
                    remaining[first_gpu] + first_size - second_size
                )
                new_second_remaining = (
                    remaining[second_gpu] + second_size - first_size
                )

                if new_first_remaining < -EPS or new_second_remaining < -EPS:
                    continue

                old_first_remaining = remaining[first_gpu]
                old_second_remaining = remaining[second_gpu]

                gpu_loads[first_gpu] += second_load - first_load
                gpu_loads[second_gpu] += first_load - second_load
                remaining[first_gpu] = new_first_remaining
                remaining[second_gpu] = new_second_remaining

                value = current_score()

                gpu_loads[first_gpu] += first_load - second_load
                gpu_loads[second_gpu] += second_load - first_load
                remaining[first_gpu] = old_first_remaining
                remaining[second_gpu] = old_second_remaining

                if value < candidate_value:
                    candidate_value = value
                    operation = ("swap", first, second)

        if operation is None:
            break

        if operation[0] == "move":
            _, item, target = operation
            source = assignment[item]
            assignment[item] = target
            gpu_loads[source] -= loads_per_model[item]
            remaining[source] += sizes[item]
            gpu_loads[target] += loads_per_model[item]
            remaining[target] -= sizes[item]
        else:
            _, first, second = operation
            first_gpu = assignment[first]
            second_gpu = assignment[second]

            assignment[first], assignment[second] = second_gpu, first_gpu

            gpu_loads[first_gpu] += loads_per_model[second] - loads_per_model[first]
            gpu_loads[second_gpu] += loads_per_model[first] - loads_per_model[second]
            remaining[first_gpu] += sizes[first] - sizes[second]
            remaining[second_gpu] += sizes[second] - sizes[first]

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    for item, gpu in enumerate(assignment):
        placement[gpu].append(models[item])

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