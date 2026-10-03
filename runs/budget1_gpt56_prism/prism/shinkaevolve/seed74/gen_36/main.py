GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models while minimizing the maximum KV cache pressure.

    A pressure threshold P transforms each model into a bin-packing item:
        transformed_weight = load + P * model_size
    with GPU capacity P * GPU_MEM_SIZE.

    Any transformed-feasible bin has:
        total_load / (GPU_MEM_SIZE - total_size) <= P
    and is memory feasible.
    """

    EPS = 1e-12
    THRESHOLD_STEPS = 24
    LOCAL_SEARCH_STEPS = 60
    MAX_SEEDS_TO_IMPROVE = 6
    EXCHANGE_GPU_LIMIT = 2
    EXCHANGE_MODEL_LIMIT = 8

    if gpu_num <= 0:
        if models:
            raise ValueError(
                "Unable to place models because gpu_num must be positive."
            )
        return {}

    def load_of(model):
        return model.req_rate / model.slo

    def pressure(load, remaining):
        if remaining <= EPS:
            return float("inf") if load > EPS else 0.0
        return load / remaining

    def pressures(loads, remaining):
        return [pressure(loads[g], remaining[g]) for g in range(gpu_num)]

    def score(loads, remaining):
        # Minimize highest pressure first, then all remaining pressure levels.
        # The final term prefers a larger free-memory block when tied.
        values = sorted(pressures(loads, remaining), reverse=True)
        return tuple(values) + (-max(remaining),)

    def make_state():
        return (
            {gpu: [] for gpu in range(gpu_num)},
            [GPU_MEM_SIZE] * gpu_num,
            [0.0] * gpu_num,
        )

    def actual_score(state):
        return score(state[2], state[1])

    def greedy_memory_pack(order, best_fit=False):
        """Memory-safe fallback used to establish a valid upper bound."""
        placement, remaining, loads = make_state()

        for model in order:
            size = model.model_size
            load = load_of(model)
            chosen = None
            chosen_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + EPS:
                    continue

                new_remaining = remaining[gpu] - size
                new_load = loads[gpu] + load

                if best_fit:
                    key = (
                        new_remaining,
                        pressure(new_load, new_remaining),
                        gpu,
                    )
                else:
                    projected = list(pressures(loads, remaining))
                    projected[gpu] = pressure(new_load, new_remaining)
                    key = (
                        max(projected),
                        pressure(new_load, new_remaining),
                        -new_remaining,
                        gpu,
                    )

                if chosen_key is None or key < chosen_key:
                    chosen_key = key
                    chosen = gpu

            if chosen is None:
                return None

            placement[chosen].append(model)
            remaining[chosen] -= size
            loads[chosen] += load

        return placement, remaining, loads

    def transformed_pack(bound, order_kind):
        """
        Best-fit transformed bin packing for a fixed KVPR bound.
        Different orderings are intentionally all evaluated because they can
        produce different valid real-pressure profiles.
        """
        capacity = bound * GPU_MEM_SIZE
        if capacity <= 0:
            return None

        weighted = [
            (model, load_of(model) + bound * model.model_size)
            for model in models
        ]

        if any(weight > capacity + EPS for _, weight in weighted):
            return None

        if order_kind == 0:
            weighted.sort(key=lambda x: (x[1], x[0].model_size), reverse=True)
        elif order_kind == 1:
            weighted.sort(
                key=lambda x: (x[0].model_size, x[1]),
                reverse=True,
            )
        elif order_kind == 2:
            weighted.sort(
                key=lambda x: (
                    load_of(x[0]) / max(x[0].model_size, EPS),
                    x[0].model_size,
                ),
                reverse=True,
            )
        else:
            weighted.sort(
                key=lambda x: (
                    load_of(x[0]) / max(
                        GPU_MEM_SIZE - x[0].model_size, EPS
                    ),
                    x[1],
                ),
                reverse=True,
            )

        placement, remaining, loads = make_state()
        transformed_used = [0.0] * gpu_num

        for model, weight in weighted:
            chosen = None
            chosen_key = None

            for gpu in range(gpu_num):
                if transformed_used[gpu] + weight > capacity + EPS:
                    continue
                if model.model_size > remaining[gpu] + EPS:
                    continue

                # Best fit packs transformed capacity tightly.  The pressure
                # tie-breaker selects the better real assignment among ties.
                new_used = transformed_used[gpu] + weight
                new_remaining = remaining[gpu] - model.model_size
                new_load = loads[gpu] + load_of(model)
                key = (
                    capacity - new_used,
                    pressure(new_load, new_remaining),
                    gpu,
                )

                if chosen_key is None or key < chosen_key:
                    chosen_key = key
                    chosen = gpu

            if chosen is None:
                return None

            placement[chosen].append(model)
            remaining[chosen] -= model.model_size
            loads[chosen] += load_of(model)
            transformed_used[chosen] += weight

        return placement, remaining, loads

    def improve(state):
        """Best relocation/swap using lexicographic GPU-pressure minimization."""
        placement, remaining, loads = state

        for _ in range(LOCAL_SEARCH_STEPS):
            best_state_score = score(loads, remaining)
            best_action = None

            def candidate_score(left_gpu, left_load, left_remaining,
                                right_gpu=None, right_load=None,
                                right_remaining=None):
                values = []
                for gpu in range(gpu_num):
                    if gpu == left_gpu:
                        values.append(pressure(left_load, left_remaining))
                    elif gpu == right_gpu:
                        values.append(pressure(right_load, right_remaining))
                    else:
                        values.append(pressure(loads[gpu], remaining[gpu]))
                values.sort(reverse=True)

                largest_remaining = max(
                    left_remaining if gpu == left_gpu else
                    right_remaining if gpu == right_gpu else
                    remaining[gpu]
                    for gpu in range(gpu_num)
                )
                return tuple(values) + (-largest_remaining,)

            for source in range(gpu_num):
                for model in list(placement[source]):
                    model_load = load_of(model)
                    model_size = model.model_size

                    for target in range(gpu_num):
                        if source == target:
                            continue
                        if model_size > remaining[target] + EPS:
                            continue

                        new_source_load = loads[source] - model_load
                        new_source_remaining = remaining[source] + model_size
                        new_target_load = loads[target] + model_load
                        new_target_remaining = remaining[target] - model_size

                        candidate = candidate_score(
                            source, new_source_load, new_source_remaining,
                            target, new_target_load, new_target_remaining,
                        )
                        if candidate < best_state_score:
                            best_state_score = candidate
                            best_action = ("move", source, target, model)

            for left in range(gpu_num):
                for right in range(left + 1, gpu_num):
                    for left_model in list(placement[left]):
                        for right_model in list(placement[right]):
                            new_left_remaining = (
                                remaining[left]
                                + left_model.model_size
                                - right_model.model_size
                            )
                            new_right_remaining = (
                                remaining[right]
                                + right_model.model_size
                                - left_model.model_size
                            )

                            if new_left_remaining < -EPS:
                                continue
                            if new_right_remaining < -EPS:
                                continue

                            new_left_load = (
                                loads[left]
                                - load_of(left_model)
                                + load_of(right_model)
                            )
                            new_right_load = (
                                loads[right]
                                - load_of(right_model)
                                + load_of(left_model)
                            )

                            candidate = candidate_score(
                                left, new_left_load, new_left_remaining,
                                right, new_right_load, new_right_remaining,
                            )
                            if candidate < best_state_score:
                                best_state_score = candidate
                                best_action = (
                                    "swap",
                                    left,
                                    right,
                                    left_model,
                                    right_model,
                                )

            # A two-for-one exchange can resolve fragmented memory layouts
            # which cannot improve through a relocation or ordinary swap.
            # Restrict this neighborhood to currently overloaded GPUs and
            # relatively small outgoing models to keep local search bounded.
            source_gpus = sorted(
                range(gpu_num),
                key=lambda gpu: pressure(loads[gpu], remaining[gpu]),
                reverse=True,
            )[:EXCHANGE_GPU_LIMIT]

            for source in source_gpus:
                source_models = sorted(
                    placement[source],
                    key=lambda model: model.model_size,
                )[:EXCHANGE_MODEL_LIMIT]

                for first_index in range(len(source_models)):
                    first_model = source_models[first_index]
                    first_load = load_of(first_model)

                    for second_model in source_models[first_index + 1:]:
                        second_load = load_of(second_model)
                        outgoing_size = (
                            first_model.model_size + second_model.model_size
                        )
                        outgoing_load = first_load + second_load

                        for target in range(gpu_num):
                            if target == source:
                                continue

                            target_models = sorted(
                                placement[target],
                                key=lambda model: model.model_size,
                                reverse=True,
                            )[:EXCHANGE_MODEL_LIMIT]

                            for return_model in target_models:
                                new_source_remaining = (
                                    remaining[source]
                                    + outgoing_size
                                    - return_model.model_size
                                )
                                new_target_remaining = (
                                    remaining[target]
                                    + return_model.model_size
                                    - outgoing_size
                                )

                                if new_source_remaining < -EPS:
                                    continue
                                if new_target_remaining < -EPS:
                                    continue

                                new_source_load = (
                                    loads[source]
                                    - outgoing_load
                                    + load_of(return_model)
                                )
                                new_target_load = (
                                    loads[target]
                                    - load_of(return_model)
                                    + outgoing_load
                                )

                                candidate = candidate_score(
                                    source,
                                    new_source_load,
                                    new_source_remaining,
                                    target,
                                    new_target_load,
                                    new_target_remaining,
                                )
                                if candidate < best_state_score:
                                    best_state_score = candidate
                                    best_action = (
                                        "exchange",
                                        source,
                                        target,
                                        first_model,
                                        second_model,
                                        return_model,
                                    )

            if best_action is None:
                break

            if best_action[0] == "move":
                _, source, target, model = best_action
                placement[source].remove(model)
                placement[target].append(model)
                remaining[source] += model.model_size
                remaining[target] -= model.model_size
                loads[source] -= load_of(model)
                loads[target] += load_of(model)
            elif best_action[0] == "swap":
                _, left, right, left_model, right_model = best_action
                placement[left].remove(left_model)
                placement[right].remove(right_model)
                placement[left].append(right_model)
                placement[right].append(left_model)

                remaining[left] += (
                    left_model.model_size - right_model.model_size
                )
                remaining[right] += (
                    right_model.model_size - left_model.model_size
                )
                loads[left] += load_of(right_model) - load_of(left_model)
                loads[right] += load_of(left_model) - load_of(right_model)
            else:
                (
                    _,
                    source,
                    target,
                    first_model,
                    second_model,
                    return_model,
                ) = best_action
                placement[source].remove(first_model)
                placement[source].remove(second_model)
                placement[target].remove(return_model)
                placement[target].extend([first_model, second_model])
                placement[source].append(return_model)

                outgoing_size = (
                    first_model.model_size + second_model.model_size
                )
                outgoing_load = load_of(first_model) + load_of(second_model)
                remaining[source] += outgoing_size - return_model.model_size
                remaining[target] += return_model.model_size - outgoing_size
                loads[source] += load_of(return_model) - outgoing_load
                loads[target] += outgoing_load - load_of(return_model)

        return placement, remaining, loads

    for model in models:
        if model.model_size > GPU_MEM_SIZE + EPS:
            raise ValueError(
                "Unable to place all models within GPU memory. "
                f"Each GPU has {GPU_MEM_SIZE} GB."
            )

    base_orders = [
        sorted(models, key=lambda m: (m.model_size, load_of(m)), reverse=True),
        sorted(models, key=lambda m: (load_of(m), m.model_size), reverse=True),
        sorted(
            models,
            key=lambda m: (
                load_of(m) / max(GPU_MEM_SIZE - m.model_size, EPS),
                m.model_size,
            ),
            reverse=True,
        ),
    ]

    candidates = []
    for index, order in enumerate(base_orders):
        state = greedy_memory_pack(order, best_fit=(index == 0))
        if state is not None:
            candidates.append(state)

    if not candidates:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    best_initial = min(candidates, key=actual_score)
    upper = max(pressures(best_initial[2], best_initial[1]))
    lower = 0.0

    # Search the transformed pressure bound.  Every successful packing is
    # retained because a higher bound can still yield a better real assignment.
    for _ in range(THRESHOLD_STEPS):
        middle = (lower + upper) / 2.0
        round_candidates = []

        for order_kind in range(4):
            state = transformed_pack(middle, order_kind)
            if state is not None:
                round_candidates.append(state)
                candidates.append(state)

        if round_candidates:
            upper = middle
        else:
            lower = middle

    # Improve only the strongest distinct seeds to retain low execution time.
    candidates.sort(key=actual_score)
    selected = candidates[:MAX_SEEDS_TO_IMPROVE]

    best_result = None
    best_result_score = None
    for state in selected:
        improved = improve(state)
        improved_score = actual_score(improved)
        if best_result_score is None or improved_score < best_result_score:
            best_result = improved
            best_result_score = improved_score

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