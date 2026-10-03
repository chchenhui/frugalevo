GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement minimizing the maximum KV cache pressure.
    """

    def model_load(model):
        return model.req_rate / model.slo

    def kvpr(load, remaining):
        if remaining <= 0:
            return float("inf") if load > 0 else 0.0
        return load / remaining

    def score(load, remaining):
        # Lexicographic balancing: first minimize max KVPR, then second highest,
        # and so on.  This is a useful deterministic tie breaker for max KVPR.
        return tuple(
            sorted(
                (kvpr(load[gpu], remaining[gpu]) for gpu in range(gpu_num)),
                reverse=True,
            )
        )

    indexed_models = list(enumerate(models))
    count = len(indexed_models)

    def ordered_by(key, rotation=0):
        return [
            model
            for index, model in sorted(
                indexed_models,
                key=lambda item: (
                    key(item[1]),
                    -((item[0] + rotation) % max(1, count)),
                ),
                reverse=True,
            )
        ]

    # Different orderings help avoid greedy packing failures and produce
    # complementary starting points for local search.
    orders = [
        ordered_by(lambda m: (model_load(m), m.model_size)),
        ordered_by(lambda m: (m.model_size, model_load(m))),
        ordered_by(
            lambda m: (
                model_load(m) / max(1e-9, GPU_MEM_SIZE - m.model_size),
                m.model_size,
            )
        ),
        ordered_by(
            lambda m: (
                model_load(m) / max(1e-9, m.model_size),
                m.model_size,
            )
        ),
        ordered_by(
            lambda m: (
                m.model_size / max(1e-9, model_load(m)),
                model_load(m),
            )
        ),
        ordered_by(lambda m: (model_load(m) * m.model_size, m.model_size)),
    ]

    # Rotated deterministic tie-breaking is useful when several models have
    # nearly identical pressure and size characteristics.
    if count > 1:
        orders.append(
            ordered_by(
                lambda m: (
                    model_load(m) / max(1e-9, GPU_MEM_SIZE - m.model_size),
                    m.model_size,
                ),
                count // 2,
            )
        )
        orders.append(
            ordered_by(
                lambda m: (m.model_size, model_load(m)),
                count // 3,
            )
        )

    def build_seed(ordered_models):
        placement = {gpu: [] for gpu in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        for model in ordered_models:
            size = model.model_size
            weight = model_load(model)
            best_gpu = None
            best_score = None

            for gpu in range(gpu_num):
                if size > remaining[gpu]:
                    continue

                trial_load = load[:]
                trial_remaining = remaining[:]
                trial_load[gpu] += weight
                trial_remaining[gpu] -= size
                trial_score = score(trial_load, trial_remaining)

                if best_score is None or trial_score < best_score:
                    best_score = trial_score
                    best_gpu = gpu

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            load[best_gpu] += weight
            remaining[best_gpu] -= size

        return placement, remaining, load

    seeds = []
    for order in orders:
        seed = build_seed(order)
        if seed is not None:
            placement, remaining, load = seed
            seeds.append((score(load, remaining), placement, remaining, load))

    if not seeds:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB"
        )

    # Refine several good, different seeds.  Restricting this to a few starts
    # retains the low execution cost of the original algorithm.
    seeds.sort(key=lambda item: item[0])
    seeds = seeds[:min(4, len(seeds))]

    def improve(placement, remaining, load):
        while True:
            current_score = score(load, remaining)
            best_action = None
            best_score = current_score

            # Single-model relocation.
            for source in range(gpu_num):
                for model in placement[source]:
                    size = model.model_size
                    weight = model_load(model)

                    for target in range(gpu_num):
                        if source == target or size > remaining[target]:
                            continue

                        trial_load = load[:]
                        trial_remaining = remaining[:]
                        trial_load[source] -= weight
                        trial_remaining[source] += size
                        trial_load[target] += weight
                        trial_remaining[target] -= size
                        trial_score = score(trial_load, trial_remaining)

                        if trial_score < best_score:
                            best_score = trial_score
                            best_action = ("move", source, target, model, weight)

            # Pairwise swaps can escape memory-constrained relocation minima.
            for left in range(gpu_num):
                for right in range(left + 1, gpu_num):
                    for left_model in placement[left]:
                        left_size = left_model.model_size
                        left_weight = model_load(left_model)

                        for right_model in placement[right]:
                            right_size = right_model.model_size
                            right_weight = model_load(right_model)

                            left_remaining = (
                                remaining[left] + left_size - right_size
                            )
                            right_remaining = (
                                remaining[right] + right_size - left_size
                            )

                            if left_remaining < 0 or right_remaining < 0:
                                continue

                            trial_load = load[:]
                            trial_remaining = remaining[:]
                            trial_load[left] += right_weight - left_weight
                            trial_load[right] += left_weight - right_weight
                            trial_remaining[left] = left_remaining
                            trial_remaining[right] = right_remaining
                            trial_score = score(trial_load, trial_remaining)

                            if trial_score < best_score:
                                best_score = trial_score
                                best_action = (
                                    "swap",
                                    left,
                                    right,
                                    left_model,
                                    right_model,
                                    left_weight,
                                    right_weight,
                                )

            if best_action is None:
                break

            if best_action[0] == "move":
                _, source, target, model, weight = best_action
                placement[source].remove(model)
                placement[target].append(model)
                load[source] -= weight
                load[target] += weight
                remaining[source] += model.model_size
                remaining[target] -= model.model_size
            else:
                (
                    _,
                    left,
                    right,
                    left_model,
                    right_model,
                    left_weight,
                    right_weight,
                ) = best_action

                placement[left].remove(left_model)
                placement[right].remove(right_model)
                placement[left].append(right_model)
                placement[right].append(left_model)

                load[left] += right_weight - left_weight
                load[right] += left_weight - right_weight
                remaining[left] += left_model.model_size - right_model.model_size
                remaining[right] += right_model.model_size - left_model.model_size

        return placement, remaining, load

    best_result = None
    best_result_score = None

    for _, placement, remaining, load in seeds:
        placement, remaining, load = improve(placement, remaining, load)
        final_score = score(load, remaining)

        if best_result_score is None or final_score < best_result_score:
            best_result_score = final_score
            best_result = placement

    return best_result

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