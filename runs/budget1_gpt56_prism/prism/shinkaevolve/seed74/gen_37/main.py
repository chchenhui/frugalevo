GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement minimizing maximum KV cache pressure.

    The algorithm combines:
      1. Multiple greedy initial placements.
      2. Binary search over transformed KVPR-feasible bin-packing bounds.
      3. Local search with moves, swaps, and bounded 2-for-1 exchanges.
    """

    EPS = 1e-12
    THRESHOLD_STEPS = 30
    LOCAL_SEARCH_STEPS = 80
    MAX_SEEDS_TO_IMPROVE = 8
    EXCHANGE_GPU_LIMIT = 2
    EXCHANGE_MODEL_LIMIT = 12

    if gpu_num <= 0:
        if models:
            raise ValueError("Unable to place models because gpu_num must be positive.")
        return {}

    def model_load(model):
        return model.req_rate / model.slo

    def kvpr(load, remaining):
        if remaining <= EPS:
            return float("inf") if load > EPS else 0.0
        return load / remaining

    def make_state():
        return (
            {gpu: [] for gpu in range(gpu_num)},
            [GPU_MEM_SIZE] * gpu_num,
            [0.0] * gpu_num,
        )

    def state_score(remaining, loads):
        values = sorted(
            (kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)),
            reverse=True,
        )
        return tuple(values) + (-max(remaining),)

    def greedy_pack(order, compact=False):
        placement, remaining, loads = make_state()

        for model in order:
            size = model.model_size
            load = model_load(model)
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + EPS:
                    continue

                new_remaining = remaining[gpu] - size
                new_load = loads[gpu] + load

                if compact:
                    key = (
                        new_remaining,
                        kvpr(new_load, new_remaining),
                        gpu,
                    )
                else:
                    projected = [
                        kvpr(loads[i], remaining[i])
                        for i in range(gpu_num)
                    ]
                    projected[gpu] = kvpr(new_load, new_remaining)
                    key = (
                        max(projected),
                        kvpr(new_load, new_remaining),
                        -new_remaining,
                        gpu,
                    )

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            remaining[best_gpu] -= size
            loads[best_gpu] += load

        return placement, remaining, loads

    def transformed_pack(bound, ordering):
        """
        For target pressure P, each GPU has transformed capacity P * memory.
        A model consumes load + P * size. Any transformed-feasible placement
        has actual KVPR at most P.
        """
        capacity = bound * GPU_MEM_SIZE
        if capacity <= EPS:
            return None

        items = [
            (model, model_load(model) + bound * model.model_size)
            for model in models
        ]

        if any(weight > capacity + EPS for _, weight in items):
            return None

        if ordering == 0:
            items.sort(key=lambda x: (x[1], x[0].model_size), reverse=True)
        elif ordering == 1:
            items.sort(key=lambda x: (x[0].model_size, x[1]), reverse=True)
        elif ordering == 2:
            items.sort(
                key=lambda x: (
                    model_load(x[0]) / max(x[0].model_size, EPS),
                    x[0].model_size,
                ),
                reverse=True,
            )
        else:
            items.sort(
                key=lambda x: (
                    model_load(x[0]) /
                    max(GPU_MEM_SIZE - x[0].model_size, EPS),
                    x[1],
                ),
                reverse=True,
            )

        placement, remaining, loads = make_state()
        used = [0.0] * gpu_num

        for model, weight in items:
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if used[gpu] + weight > capacity + EPS:
                    continue
                if model.model_size > remaining[gpu] + EPS:
                    continue

                new_used = used[gpu] + weight
                new_remaining = remaining[gpu] - model.model_size
                new_load = loads[gpu] + model_load(model)

                key = (
                    capacity - new_used,
                    kvpr(new_load, new_remaining),
                    gpu,
                )

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            remaining[best_gpu] -= model.model_size
            loads[best_gpu] += model_load(model)
            used[best_gpu] += weight

        return placement, remaining, loads

    def candidate_score(remaining, loads, changes):
        test_remaining = list(remaining)
        test_loads = list(loads)

        for gpu, new_remaining, new_load in changes:
            test_remaining[gpu] = new_remaining
            test_loads[gpu] = new_load

        return state_score(test_remaining, test_loads)

    def improve(state):
        placement, remaining, loads = state

        for _ in range(LOCAL_SEARCH_STEPS):
            current_score = state_score(remaining, loads)
            best_score = current_score
            best_action = None

            # Single-model relocation.
            for source in range(gpu_num):
                for model in list(placement[source]):
                    size = model.model_size
                    load = model_load(model)

                    for target in range(gpu_num):
                        if source == target or size > remaining[target] + EPS:
                            continue

                        score = candidate_score(
                            remaining,
                            loads,
                            [
                                (
                                    source,
                                    remaining[source] + size,
                                    loads[source] - load,
                                ),
                                (
                                    target,
                                    remaining[target] - size,
                                    loads[target] + load,
                                ),
                            ],
                        )

                        if score < best_score:
                            best_score = score
                            best_action = ("move", source, target, model)

            # Pairwise 1-for-1 swaps.
            for left in range(gpu_num):
                for right in range(left + 1, gpu_num):
                    for left_model in list(placement[left]):
                        left_size = left_model.model_size
                        left_load = model_load(left_model)

                        for right_model in list(placement[right]):
                            right_size = right_model.model_size
                            right_load = model_load(right_model)

                            new_left_remaining = (
                                remaining[left] + left_size - right_size
                            )
                            new_right_remaining = (
                                remaining[right] + right_size - left_size
                            )

                            if new_left_remaining < -EPS:
                                continue
                            if new_right_remaining < -EPS:
                                continue

                            score = candidate_score(
                                remaining,
                                loads,
                                [
                                    (
                                        left,
                                        new_left_remaining,
                                        loads[left] - left_load + right_load,
                                    ),
                                    (
                                        right,
                                        new_right_remaining,
                                        loads[right] - right_load + left_load,
                                    ),
                                ],
                            )

                            if score < best_score:
                                best_score = score
                                best_action = (
                                    "swap",
                                    left,
                                    right,
                                    left_model,
                                    right_model,
                                )

            # Bounded 2-for-1 exchanges, focused on highest-pressure GPUs.
            ranked_gpus = sorted(
                range(gpu_num),
                key=lambda g: kvpr(loads[g], remaining[g]),
                reverse=True,
            )[:EXCHANGE_GPU_LIMIT]

            for source in ranked_gpus:
                source_models = sorted(
                    placement[source],
                    key=lambda m: (m.model_size, -model_load(m)),
                )[:EXCHANGE_MODEL_LIMIT]

                if len(source_models) < 2:
                    continue

                for target in range(gpu_num):
                    if target == source or not placement[target]:
                        continue

                    target_models = sorted(
                        placement[target],
                        key=lambda m: (
                            m.model_size,
                            model_load(m),
                        ),
                        reverse=True,
                    )[:EXCHANGE_MODEL_LIMIT]

                    for first_index in range(len(source_models)):
                        first = source_models[first_index]
                        first_size = first.model_size
                        first_load = model_load(first)

                        for second_index in range(first_index + 1, len(source_models)):
                            second = source_models[second_index]
                            second_size = second.model_size
                            second_load = model_load(second)

                            outgoing_size = first_size + second_size
                            outgoing_load = first_load + second_load

                            for incoming in target_models:
                                incoming_size = incoming.model_size
                                incoming_load = model_load(incoming)

                                new_source_remaining = (
                                    remaining[source]
                                    + outgoing_size
                                    - incoming_size
                                )
                                new_target_remaining = (
                                    remaining[target]
                                    + incoming_size
                                    - outgoing_size
                                )

                                if new_source_remaining < -EPS:
                                    continue
                                if new_target_remaining < -EPS:
                                    continue

                                score = candidate_score(
                                    remaining,
                                    loads,
                                    [
                                        (
                                            source,
                                            new_source_remaining,
                                            loads[source]
                                            - outgoing_load
                                            + incoming_load,
                                        ),
                                        (
                                            target,
                                            new_target_remaining,
                                            loads[target]
                                            - incoming_load
                                            + outgoing_load,
                                        ),
                                    ],
                                )

                                if score < best_score:
                                    best_score = score
                                    best_action = (
                                        "exchange",
                                        source,
                                        target,
                                        first,
                                        second,
                                        incoming,
                                    )

            if best_action is None:
                break

            kind = best_action[0]

            if kind == "move":
                _, source, target, model = best_action
                placement[source].remove(model)
                placement[target].append(model)
                remaining[source] += model.model_size
                remaining[target] -= model.model_size
                loads[source] -= model_load(model)
                loads[target] += model_load(model)

            elif kind == "swap":
                _, left, right, left_model, right_model = best_action
                placement[left].remove(left_model)
                placement[right].remove(right_model)
                placement[left].append(right_model)
                placement[right].append(left_model)

                remaining[left] += left_model.model_size - right_model.model_size
                remaining[right] += right_model.model_size - left_model.model_size
                loads[left] += model_load(right_model) - model_load(left_model)
                loads[right] += model_load(left_model) - model_load(right_model)

            else:
                _, source, target, first, second, incoming = best_action

                placement[source].remove(first)
                placement[source].remove(second)
                placement[target].remove(incoming)

                placement[source].append(incoming)
                placement[target].append(first)
                placement[target].append(second)

                outgoing_size = first.model_size + second.model_size
                outgoing_load = model_load(first) + model_load(second)

                remaining[source] += outgoing_size - incoming.model_size
                remaining[target] += incoming.model_size - outgoing_size

                loads[source] += model_load(incoming) - outgoing_load
                loads[target] += outgoing_load - model_load(incoming)

        return placement, remaining, loads

    for model in models:
        if model.model_size > GPU_MEM_SIZE + EPS:
            raise ValueError(
                "Unable to place all models within GPU memory. "
                f"Each GPU has {GPU_MEM_SIZE} GB."
            )

    if not models:
        return {gpu: [] for gpu in range(gpu_num)}

    base_orders = [
        sorted(models, key=lambda m: (m.model_size, model_load(m)), reverse=True),
        sorted(models, key=lambda m: (model_load(m), m.model_size), reverse=True),
        sorted(
            models,
            key=lambda m: (
                model_load(m) / max(m.model_size, EPS),
                m.model_size,
            ),
            reverse=True,
        ),
        sorted(
            models,
            key=lambda m: (
                model_load(m) / max(GPU_MEM_SIZE - m.model_size, EPS),
                m.model_size,
            ),
            reverse=True,
        ),
    ]

    candidates = []
    for index, order in enumerate(base_orders):
        state = greedy_pack(order, compact=(index == 0))
        if state is not None:
            candidates.append(state)

    if not candidates:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    initial_best = min(
        candidates,
        key=lambda state: state_score(state[1], state[2]),
    )
    upper = max(
        kvpr(initial_best[2][gpu], initial_best[1][gpu])
        for gpu in range(gpu_num)
    )
    lower = 0.0

    for _ in range(THRESHOLD_STEPS):
        middle = (lower + upper) / 2.0
        successful = []

        for ordering in range(4):
            state = transformed_pack(middle, ordering)
            if state is not None:
                successful.append(state)
                candidates.append(state)

        if successful:
            upper = middle
        else:
            lower = middle

    candidates.sort(key=lambda state: state_score(state[1], state[2]))
    seeds = candidates[:MAX_SEEDS_TO_IMPROVE]

    best_result = None
    best_score = None

    for seed in seeds:
        result = improve(seed)
        result_score = state_score(result[1], result[2])

        if best_score is None or result_score < best_score:
            best_result = result
            best_score = result_score

    return best_result[0]

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