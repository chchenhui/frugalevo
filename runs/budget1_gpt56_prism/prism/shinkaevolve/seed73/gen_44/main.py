GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement that minimizes maximum KV cache pressure.

    Returns:
        Dictionary mapping GPU ids to lists of assigned models.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    def weight(model):
        if model.slo == 0:
            raise ValueError("model.slo must be non-zero")
        return model.req_rate / model.slo

    def ratio(load, remaining):
        if remaining > 0:
            return load / remaining
        return float("inf") if load > 0 else 0.0

    class PlacementState:
        def __init__(self):
            self.groups = [[] for _ in range(gpu_num)]
            self.loads = [0.0] * gpu_num
            self.remaining = [float(GPU_MEM_SIZE)] * gpu_num

        def score(self):
            return max(
                ratio(self.loads[gpu], self.remaining[gpu])
                for gpu in range(gpu_num)
            )

        def projected_add_score(self, gpu, model):
            model_weight = weight(model)
            projected = [
                ratio(self.loads[i], self.remaining[i])
                for i in range(gpu_num)
            ]
            projected[gpu] = ratio(
                self.loads[gpu] + model_weight,
                self.remaining[gpu] - model.model_size,
            )
            return max(projected)

        def add(self, gpu, model):
            self.groups[gpu].append(model)
            self.loads[gpu] += weight(model)
            self.remaining[gpu] -= model.model_size

        def move_score(self, source, target, model):
            model_weight = weight(model)
            values = [
                ratio(self.loads[i], self.remaining[i])
                for i in range(gpu_num)
            ]
            values[source] = ratio(
                self.loads[source] - model_weight,
                self.remaining[source] + model.model_size,
            )
            values[target] = ratio(
                self.loads[target] + model_weight,
                self.remaining[target] - model.model_size,
            )
            return max(values)

        def move(self, source, target, model):
            self.groups[source].remove(model)
            self.groups[target].append(model)
            model_weight = weight(model)
            self.loads[source] -= model_weight
            self.remaining[source] += model.model_size
            self.loads[target] += model_weight
            self.remaining[target] -= model.model_size

        def swap_score(self, left_gpu, right_gpu, left_model, right_model):
            left_weight = weight(left_model)
            right_weight = weight(right_model)
            values = [
                ratio(self.loads[i], self.remaining[i])
                for i in range(gpu_num)
            ]
            values[left_gpu] = ratio(
                self.loads[left_gpu] - left_weight + right_weight,
                self.remaining[left_gpu] + left_model.model_size - right_model.model_size,
            )
            values[right_gpu] = ratio(
                self.loads[right_gpu] - right_weight + left_weight,
                self.remaining[right_gpu] + right_model.model_size - left_model.model_size,
            )
            return max(values)

        def swap(self, left_gpu, right_gpu, left_model, right_model):
            left_index = self.groups[left_gpu].index(left_model)
            right_index = self.groups[right_gpu].index(right_model)
            self.groups[left_gpu][left_index] = right_model
            self.groups[right_gpu][right_index] = left_model

            left_weight = weight(left_model)
            right_weight = weight(right_model)

            self.loads[left_gpu] += right_weight - left_weight
            self.loads[right_gpu] += left_weight - right_weight
            self.remaining[left_gpu] += left_model.model_size - right_model.model_size
            self.remaining[right_gpu] += right_model.model_size - left_model.model_size

        def exchange_score(
            self, left_gpu, right_gpu, left_model, right_model_a, right_model_b
        ):
            left_weight = weight(left_model)
            right_weight = weight(right_model_a) + weight(right_model_b)
            values = [
                ratio(self.loads[i], self.remaining[i])
                for i in range(gpu_num)
            ]
            values[left_gpu] = ratio(
                self.loads[left_gpu] - left_weight + right_weight,
                self.remaining[left_gpu]
                + left_model.model_size
                - right_model_a.model_size
                - right_model_b.model_size,
            )
            values[right_gpu] = ratio(
                self.loads[right_gpu] - right_weight + left_weight,
                self.remaining[right_gpu]
                + right_model_a.model_size
                + right_model_b.model_size
                - left_model.model_size,
            )
            return max(values)

        def exchange(
            self, left_gpu, right_gpu, left_model, right_model_a, right_model_b
        ):
            self.groups[left_gpu].remove(left_model)
            self.groups[right_gpu].remove(right_model_a)
            self.groups[right_gpu].remove(right_model_b)
            self.groups[left_gpu].extend([right_model_a, right_model_b])
            self.groups[right_gpu].append(left_model)

            left_weight = weight(left_model)
            right_weight = weight(right_model_a) + weight(right_model_b)
            self.loads[left_gpu] += right_weight - left_weight
            self.loads[right_gpu] += left_weight - right_weight
            size_delta = (
                left_model.model_size
                - right_model_a.model_size
                - right_model_b.model_size
            )
            self.remaining[left_gpu] += size_delta
            self.remaining[right_gpu] -= size_delta

        def result(self):
            return {gpu: list(self.groups[gpu]) for gpu in range(gpu_num)}

    if any(model.model_size > GPU_MEM_SIZE for model in models):
        too_large = next(model for model in models if model.model_size > GPU_MEM_SIZE)
        raise ValueError(
            f"Unable to place model of size {too_large.model_size} GB on a "
            f"{GPU_MEM_SIZE} GB GPU"
        )

    # Different orderings address both KVPR balancing and memory fragmentation.
    ordered_inputs = [
        sorted(models, key=lambda m: (weight(m), m.model_size), reverse=True),
        sorted(models, key=lambda m: (m.model_size, weight(m)), reverse=True),
        sorted(models, key=lambda m: weight(m) * m.model_size, reverse=True),
        sorted(
            models,
            key=lambda m: weight(m) / m.model_size if m.model_size > 0 else float("inf"),
            reverse=True,
        ),
        sorted(models, key=lambda m: (m.model_size, -weight(m)), reverse=True),
    ]

    def construct(ordered_models, pack_ties):
        state = PlacementState()

        for model in ordered_models:
            candidates = []
            for gpu in range(gpu_num):
                if model.model_size <= state.remaining[gpu]:
                    projected_score = state.projected_add_score(gpu, model)

                    # Packing ties protects empty GPUs for future large models.
                    # Spreading ties keeps KV pressure more evenly distributed.
                    tie_memory = (
                        state.remaining[gpu] - model.model_size
                        if pack_ties
                        else -state.remaining[gpu]
                    )
                    candidates.append((projected_score, tie_memory, gpu))

            if not candidates:
                return None

            _, _, selected_gpu = min(candidates)
            state.add(selected_gpu, model)

        return state

    def improve(state):
        epsilon = 1e-12
        limit = max(10, len(models) * 3)

        for _ in range(limit):
            current_score = state.score()
            current_values = [
                ratio(state.loads[gpu], state.remaining[gpu])
                for gpu in range(gpu_num)
            ]
            # A lexicographic pressure vector allows useful neutral changes:
            # reduce one of several tied bottlenecks before lowering their max.
            best_key = tuple(sorted(current_values, reverse=True))
            bottlenecks = {
                gpu
                for gpu in range(gpu_num)
                if current_values[gpu] + epsilon >= current_score
            }
            best_action = None

            def key_after(*updates):
                values = list(current_values)
                for gpu, load_delta, memory_delta in updates:
                    values[gpu] = ratio(
                        state.loads[gpu] + load_delta,
                        state.remaining[gpu] + memory_delta,
                    )
                return tuple(sorted(values, reverse=True))

            # Relocations out of a current bottleneck are most likely to
            # improve the maximum pressure.
            for source in bottlenecks:
                for model in state.groups[source]:
                    for target in range(gpu_num):
                        if source == target or model.model_size > state.remaining[target]:
                            continue

                        model_weight = weight(model)
                        candidate_key = key_after(
                            (source, -model_weight, model.model_size),
                            (target, model_weight, -model.model_size),
                        )
                        if candidate_key < best_key:
                            best_key = candidate_key
                            best_action = ("move", source, target, model)

            # Swaps can improve balance when no single model can move.
            for left_gpu in range(gpu_num):
                for right_gpu in range(left_gpu + 1, gpu_num):
                    if left_gpu not in bottlenecks and right_gpu not in bottlenecks:
                        continue
                    for left_model in state.groups[left_gpu]:
                        for right_model in state.groups[right_gpu]:
                            left_after = (
                                state.remaining[left_gpu]
                                + left_model.model_size
                                - right_model.model_size
                            )
                            right_after = (
                                state.remaining[right_gpu]
                                + right_model.model_size
                                - left_model.model_size
                            )

                            if left_after < 0 or right_after < 0:
                                continue

                            left_weight = weight(left_model)
                            right_weight = weight(right_model)
                            candidate_key = key_after(
                                (
                                    left_gpu,
                                    right_weight - left_weight,
                                    left_model.model_size - right_model.model_size,
                                ),
                                (
                                    right_gpu,
                                    left_weight - right_weight,
                                    right_model.model_size - left_model.model_size,
                                ),
                            )
                            if candidate_key < best_key:
                                best_key = candidate_key
                                best_action = (
                                    "swap",
                                    left_gpu,
                                    right_gpu,
                                    left_model,
                                    right_model,
                                )

            # When ordinary local improvements are exhausted, try a compound
            # one-for-two exchange.  This can cross memory-fragmentation barriers
            # that make both a move and a one-for-one swap impossible.
            if best_action is None:
                for first_gpu in range(gpu_num):
                    for second_gpu in range(first_gpu + 1, gpu_num):
                        if (
                            first_gpu not in bottlenecks
                            and second_gpu not in bottlenecks
                        ):
                            continue
                        for left_gpu, right_gpu in (
                            (first_gpu, second_gpu),
                            (second_gpu, first_gpu),
                        ):
                            for left_model in state.groups[left_gpu]:
                                right_models = state.groups[right_gpu]
                                for first_index in range(len(right_models)):
                                    for second_index in range(
                                        first_index + 1, len(right_models)
                                    ):
                                        right_model_a = right_models[first_index]
                                        right_model_b = right_models[second_index]
                                        left_after = (
                                            state.remaining[left_gpu]
                                            + left_model.model_size
                                            - right_model_a.model_size
                                            - right_model_b.model_size
                                        )
                                        right_after = (
                                            state.remaining[right_gpu]
                                            + right_model_a.model_size
                                            + right_model_b.model_size
                                            - left_model.model_size
                                        )
                                        if left_after < 0 or right_after < 0:
                                            continue

                                        left_weight = weight(left_model)
                                        right_weight = (
                                            weight(right_model_a)
                                            + weight(right_model_b)
                                        )
                                        size_delta = (
                                            left_model.model_size
                                            - right_model_a.model_size
                                            - right_model_b.model_size
                                        )
                                        candidate_key = key_after(
                                            (
                                                left_gpu,
                                                right_weight - left_weight,
                                                size_delta,
                                            ),
                                            (
                                                right_gpu,
                                                left_weight - right_weight,
                                                -size_delta,
                                            ),
                                        )
                                        if candidate_key < best_key:
                                            best_key = candidate_key
                                            best_action = (
                                                "exchange",
                                                left_gpu,
                                                right_gpu,
                                                left_model,
                                                right_model_a,
                                                right_model_b,
                                            )

            # A three-GPU ejection chain can make room on a good target GPU:
            # move one target item elsewhere, then move a bottleneck item in.
            # Restricting the source list keeps this fragmentation repair cheap.
            if best_action is None:
                for source in bottlenecks:
                    source_items = sorted(
                        state.groups[source],
                        key=lambda model: (
                            weight(model),
                            weight(model) / max(model.model_size, epsilon),
                        ),
                        reverse=True,
                    )[:6]

                    for model in source_items:
                        model_weight = weight(model)
                        for target in range(gpu_num):
                            if target == source:
                                continue

                            for displaced in state.groups[target]:
                                displaced_weight = weight(displaced)
                                if (
                                    model.model_size
                                    > state.remaining[target]
                                    + displaced.model_size
                                    + epsilon
                                ):
                                    continue

                                for destination in range(gpu_num):
                                    if (
                                        destination == source
                                        or destination == target
                                        or displaced.model_size
                                        > state.remaining[destination] + epsilon
                                    ):
                                        continue

                                    candidate_key = key_after(
                                        (
                                            source,
                                            -model_weight,
                                            model.model_size,
                                        ),
                                        (
                                            target,
                                            model_weight - displaced_weight,
                                            displaced.model_size - model.model_size,
                                        ),
                                        (
                                            destination,
                                            displaced_weight,
                                            -displaced.model_size,
                                        ),
                                    )
                                    if candidate_key < best_key:
                                        best_key = candidate_key
                                        best_action = (
                                            "chain",
                                            source,
                                            target,
                                            destination,
                                            model,
                                            displaced,
                                        )

            if best_action is None:
                break

            if best_action[0] == "move":
                _, source, target, model = best_action
                state.move(source, target, model)
            elif best_action[0] == "swap":
                _, left_gpu, right_gpu, left_model, right_model = best_action
                state.swap(left_gpu, right_gpu, left_model, right_model)
            elif best_action[0] == "exchange":
                (
                    _,
                    left_gpu,
                    right_gpu,
                    left_model,
                    right_model_a,
                    right_model_b,
                ) = best_action
                state.exchange(
                    left_gpu,
                    right_gpu,
                    left_model,
                    right_model_a,
                    right_model_b,
                )
            else:
                _, source, target, destination, model, displaced = best_action
                state.move(target, destination, displaced)
                state.move(source, target, model)

        return state

    best_state = None
    best_score = float("inf")

    for ordered_models in ordered_inputs:
        for pack_ties in (False, True):
            candidate = construct(ordered_models, pack_ties)
            if candidate is None:
                continue

            candidate = improve(candidate)
            candidate_score = candidate.score()

            if candidate_score < best_score:
                best_score = candidate_score
                best_state = candidate

    if best_state is None:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB each"
        )

    return best_state.result()

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