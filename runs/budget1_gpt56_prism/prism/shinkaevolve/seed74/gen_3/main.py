GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible model placement that minimizes maximum KVPR.

    The implementation uses:
    1. Multiple deterministic greedy construction orders.
    2. Selection of the best complete candidate placement.
    3. Local improvement using relocations and pairwise swaps.
    """

    EPS = 1e-12

    def model_load(model):
        return model.req_rate / model.slo

    def pressure(load, remaining):
        if remaining <= 0:
            return float("inf") if load > 0 else 0.0
        return load / remaining

    def max_pressure(loads, remaining):
        return max(
            pressure(loads[gpu], remaining[gpu])
            for gpu in range(gpu_num)
        ) if gpu_num else 0.0

    def build_candidate(ordered_models, packing_mode=False):
        placement = {gpu: [] for gpu in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        loads = [0.0] * gpu_num

        for model in ordered_models:
            size = model.model_size
            load = model_load(model)
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu]:
                    continue

                next_remaining = remaining[gpu] - size
                next_load = loads[gpu] + load

                if packing_mode:
                    # Best-fit decreasing style fallback: preserve large blocks
                    # of free memory while favoring lower projected pressure.
                    key = (
                        next_remaining,
                        pressure(next_load, next_remaining),
                        gpu,
                    )
                else:
                    candidate_peak = 0.0
                    for other_gpu in range(gpu_num):
                        if other_gpu == gpu:
                            value = pressure(next_load, next_remaining)
                        else:
                            value = pressure(loads[other_gpu], remaining[other_gpu])
                        if value > candidate_peak:
                            candidate_peak = value

                    key = (
                        candidate_peak,
                        pressure(next_load, next_remaining),
                        -next_remaining,
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

    def improve(placement, remaining, loads):
        """
        Apply the best strictly improving relocation or swap until stable.
        """
        max_iterations = 40

        for _ in range(max_iterations):
            current_peak = max_pressure(loads, remaining)
            best_action = None
            best_peak = current_peak

            # Search relocations.
            for source in range(gpu_num):
                for model in placement[source]:
                    size = model.model_size
                    load = model_load(model)

                    for target in range(gpu_num):
                        if source == target or size > remaining[target]:
                            continue

                        candidate_peak = 0.0
                        for gpu in range(gpu_num):
                            if gpu == source:
                                value = pressure(
                                    loads[gpu] - load,
                                    remaining[gpu] + size,
                                )
                            elif gpu == target:
                                value = pressure(
                                    loads[gpu] + load,
                                    remaining[gpu] - size,
                                )
                            else:
                                value = pressure(loads[gpu], remaining[gpu])
                            candidate_peak = max(candidate_peak, value)

                        if candidate_peak < best_peak - EPS:
                            best_peak = candidate_peak
                            best_action = ("move", source, target, model)

            # Search pairwise swaps. Swaps can repair placements that cannot
            # improve through a single feasible relocation.
            for left_gpu in range(gpu_num):
                for right_gpu in range(left_gpu + 1, gpu_num):
                    for left_model in placement[left_gpu]:
                        left_size = left_model.model_size
                        left_load = model_load(left_model)

                        for right_model in placement[right_gpu]:
                            right_size = right_model.model_size
                            right_load = model_load(right_model)

                            new_left_remaining = (
                                remaining[left_gpu] + left_size - right_size
                            )
                            new_right_remaining = (
                                remaining[right_gpu] + right_size - left_size
                            )

                            if new_left_remaining < 0 or new_right_remaining < 0:
                                continue

                            new_left_load = (
                                loads[left_gpu] - left_load + right_load
                            )
                            new_right_load = (
                                loads[right_gpu] - right_load + left_load
                            )

                            candidate_peak = 0.0
                            for gpu in range(gpu_num):
                                if gpu == left_gpu:
                                    value = pressure(
                                        new_left_load,
                                        new_left_remaining,
                                    )
                                elif gpu == right_gpu:
                                    value = pressure(
                                        new_right_load,
                                        new_right_remaining,
                                    )
                                else:
                                    value = pressure(loads[gpu], remaining[gpu])
                                candidate_peak = max(candidate_peak, value)

                            if candidate_peak < best_peak - EPS:
                                best_peak = candidate_peak
                                best_action = (
                                    "swap",
                                    left_gpu,
                                    right_gpu,
                                    left_model,
                                    right_model,
                                )

            if best_action is None:
                break

            if best_action[0] == "move":
                _, source, target, model = best_action
                size = model.model_size
                load = model_load(model)

                placement[source].remove(model)
                placement[target].append(model)
                remaining[source] += size
                remaining[target] -= size
                loads[source] -= load
                loads[target] += load
            else:
                _, left_gpu, right_gpu, left_model, right_model = best_action
                left_size = left_model.model_size
                right_size = right_model.model_size
                left_load = model_load(left_model)
                right_load = model_load(right_model)

                placement[left_gpu].remove(left_model)
                placement[right_gpu].remove(right_model)
                placement[left_gpu].append(right_model)
                placement[right_gpu].append(left_model)

                remaining[left_gpu] += left_size - right_size
                remaining[right_gpu] += right_size - left_size
                loads[left_gpu] += right_load - left_load
                loads[right_gpu] += left_load - right_load

        return placement, remaining, loads

    if gpu_num <= 0:
        if models:
            raise ValueError("Unable to place models because gpu_num must be positive.")
        return {}

    # Different orderings reduce sensitivity to a single greedy ordering.
    orderings = [
        sorted(
            models,
            key=lambda m: (model_load(m), m.model_size),
            reverse=True,
        ),
        sorted(
            models,
            key=lambda m: (m.model_size, model_load(m)),
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
        sorted(
            models,
            key=lambda m: (
                model_load(m) * m.model_size,
                model_load(m),
            ),
            reverse=True,
        ),
    ]

    best_result = None
    best_score = float("inf")

    for index, ordering in enumerate(orderings):
        candidate = build_candidate(ordering, packing_mode=(index == 1))
        if candidate is None:
            continue

        placement, remaining, loads = candidate
        score = max_pressure(loads, remaining)

        if score < best_score:
            best_score = score
            best_result = (placement, remaining, loads)

    if best_result is None:
        raise ValueError(
            "Unable to place all models within GPU memory. "
            f"Each GPU has {GPU_MEM_SIZE} GB."
        )

    placement, remaining, loads = best_result
    placement, remaining, loads = improve(placement, remaining, loads)

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