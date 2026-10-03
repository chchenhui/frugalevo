GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models on GPUs while minimizing the sorted KVPR profile.

    KVPR(gpu) = sum(req_rate / slo) / remaining_memory.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    model_list = list(models)
    n = len(model_list)
    eps = 1e-10

    if any(model.model_size > GPU_MEM_SIZE + eps for model in model_list):
        bad = next(model for model in model_list
                   if model.model_size > GPU_MEM_SIZE + eps)
        raise ValueError(
            f"Unable to place model of size {bad.model_size} GB on any GPU."
        )

    sizes = [float(model.model_size) for model in model_list]
    weights = [float(model.req_rate / model.slo) for model in model_list]

    def pressure(weight, used):
        remaining = GPU_MEM_SIZE - used
        if remaining <= eps:
            return float("inf") if weight > eps else 0.0
        return weight / remaining

    def profile(used, load):
        return tuple(sorted(
            (pressure(load[g], used[g]) for g in range(gpu_num)),
            reverse=True
        ))

    def state_key(used, load):
        # GPU labels are interchangeable during construction.
        return tuple(sorted(
            (round(used[g], 8), round(load[g], 8))
            for g in range(gpu_num)
        ))

    def beam_construct(order, beam_width=10):
        # Each entry is (bins, used_memory, pressure_weight).
        beam = [([[] for _ in range(gpu_num)],
                 [0.0] * gpu_num,
                 [0.0] * gpu_num)]

        for idx in order:
            candidates = []
            seen = set()

            for bins, used, load in beam:
                gpu_choices = []
                for g in range(gpu_num):
                    if used[g] + sizes[idx] <= GPU_MEM_SIZE + eps:
                        new_used = used[:]
                        new_load = load[:]
                        new_used[g] += sizes[idx]
                        new_load[g] += weights[idx]

                        new_bins = [bucket[:] for bucket in bins]
                        new_bins[g].append(idx)

                        p = profile(new_used, new_load)
                        # The second component mildly favors compact residual
                        # memory layouts only when KVPR profiles are identical.
                        residual_shape = tuple(sorted(
                            (GPU_MEM_SIZE - value for value in new_used)
                        ))
                        gpu_choices.append(
                            (p, residual_shape, new_bins, new_used, new_load)
                        )

                candidates.extend(gpu_choices)

            if not candidates:
                return None

            candidates.sort(key=lambda item: (item[0], item[1]))
            next_beam = []

            for _, _, bins, used, load in candidates:
                key = state_key(used, load)
                if key in seen:
                    continue
                seen.add(key)
                next_beam.append((bins, used, load))
                if len(next_beam) >= beam_width:
                    break

            beam = next_beam

        return min(beam, key=lambda state: profile(state[1], state[2]))

    # Different orderings expose different memory/pressure tradeoffs.
    orders = [
        sorted(range(n), key=lambda i: (-sizes[i], -weights[i], i)),
        sorted(range(n), key=lambda i: (-weights[i], -sizes[i], i)),
        sorted(range(n), key=lambda i: (
            -(weights[i] / max(sizes[i], eps)), -sizes[i], i
        )),
        sorted(range(n), key=lambda i: (
            -(weights[i] * sizes[i]), -sizes[i], i
        )),
    ]

    best_state = None
    best_profile = None

    for order in orders:
        state = beam_construct(order, beam_width=10)
        if state is None:
            continue
        candidate = profile(state[1], state[2])
        if best_profile is None or candidate < best_profile:
            best_state = state
            best_profile = candidate

    if best_state is None:
        # Feasibility-oriented fallback.
        bins = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num

        for idx in sorted(range(n), key=lambda i: (-sizes[i], i)):
            choices = [
                g for g in range(gpu_num)
                if used[g] + sizes[idx] <= GPU_MEM_SIZE + eps
            ]
            if not choices:
                remaining = [GPU_MEM_SIZE - value for value in used]
                raise ValueError(
                    f"Unable to place model of size {sizes[idx]} GB on any GPU. "
                    f"Remaining per-GPU memory: {remaining}"
                )

            g = min(
                choices,
                key=lambda x: (GPU_MEM_SIZE - used[x] - sizes[idx], x)
            )
            bins[g].append(idx)
            used[g] += sizes[idx]
            load[g] += weights[idx]
    else:
        bins, used, load = best_state

    current = profile(used, load)

    def evaluate(changes):
        """Return profile after {gpu: (memory_delta, load_delta)} changes."""
        trial_used = used[:]
        trial_load = load[:]
        for g, (mem_delta, load_delta) in changes.items():
            trial_used[g] += mem_delta
            trial_load[g] += load_delta
        return profile(trial_used, trial_load)

    # Strict-improvement local search.  Focused exchange neighborhoods avoid
    # spending time exploring changes unrelated to the current bottleneck.
    for _ in range(35):
        best_action = None
        best_value = current

        pressures = [pressure(load[g], used[g]) for g in range(gpu_num)]
        bottleneck = max(pressures)
        source_gpus = [
            g for g in range(gpu_num)
            if pressures[g] >= bottleneck - 1e-12 and bins[g]
        ]

        # Single relocations from bottleneck GPUs.
        for source in source_gpus:
            for idx in bins[source]:
                for target in range(gpu_num):
                    if target == source:
                        continue
                    if used[target] + sizes[idx] > GPU_MEM_SIZE + eps:
                        continue

                    candidate = evaluate({
                        source: (-sizes[idx], -weights[idx]),
                        target: (sizes[idx], weights[idx]),
                    })
                    if candidate < best_value:
                        best_value = candidate
                        best_action = ("move", source, target, idx)

        # One-for-one swaps involving a bottleneck GPU.
        if best_action is None and n <= 180:
            for source in source_gpus:
                for target in range(gpu_num):
                    if target == source:
                        continue
                    for left in bins[source]:
                        for right in bins[target]:
                            source_mem = used[source] - sizes[left] + sizes[right]
                            target_mem = used[target] - sizes[right] + sizes[left]
                            if (source_mem > GPU_MEM_SIZE + eps or
                                    target_mem > GPU_MEM_SIZE + eps):
                                continue

                            candidate = evaluate({
                                source: (
                                    sizes[right] - sizes[left],
                                    weights[right] - weights[left],
                                ),
                                target: (
                                    sizes[left] - sizes[right],
                                    weights[left] - weights[right],
                                ),
                            })
                            if candidate < best_value:
                                best_value = candidate
                                best_action = (
                                    "swap", source, target, left, right
                                )

        # Asymmetric exchanges: one bottleneck model for two target models,
        # and the reverse two-for-one direction.
        if best_action is None and n <= 100:
            for source in source_gpus:
                for target in range(gpu_num):
                    if target == source:
                        continue

                    # One source model exchanged for two target models.
                    for one in bins[source]:
                        for a_pos in range(len(bins[target])):
                            for b_pos in range(a_pos + 1, len(bins[target])):
                                first = bins[target][a_pos]
                                second = bins[target][b_pos]

                                source_mem = (
                                    used[source] - sizes[one]
                                    + sizes[first] + sizes[second]
                                )
                                target_mem = (
                                    used[target] - sizes[first] - sizes[second]
                                    + sizes[one]
                                )
                                if (source_mem > GPU_MEM_SIZE + eps or
                                        target_mem > GPU_MEM_SIZE + eps):
                                    continue

                                candidate = evaluate({
                                    source: (
                                        -sizes[one] + sizes[first] + sizes[second],
                                        -weights[one] + weights[first] + weights[second],
                                    ),
                                    target: (
                                        sizes[one] - sizes[first] - sizes[second],
                                        weights[one] - weights[first] - weights[second],
                                    ),
                                })
                                if candidate < best_value:
                                    best_value = candidate
                                    best_action = (
                                        "one_for_two", source, target,
                                        one, first, second
                                    )

                    # Two source models exchanged for one target model.
                    for a_pos in range(len(bins[source])):
                        for b_pos in range(a_pos + 1, len(bins[source])):
                            first = bins[source][a_pos]
                            second = bins[source][b_pos]
                            for one in bins[target]:
                                source_mem = (
                                    used[source] - sizes[first] - sizes[second]
                                    + sizes[one]
                                )
                                target_mem = (
                                    used[target] - sizes[one]
                                    + sizes[first] + sizes[second]
                                )
                                if (source_mem > GPU_MEM_SIZE + eps or
                                        target_mem > GPU_MEM_SIZE + eps):
                                    continue

                                candidate = evaluate({
                                    source: (
                                        sizes[one] - sizes[first] - sizes[second],
                                        weights[one] - weights[first] - weights[second],
                                    ),
                                    target: (
                                        -sizes[one] + sizes[first] + sizes[second],
                                        -weights[one] + weights[first] + weights[second],
                                    ),
                                })
                                if candidate < best_value:
                                    best_value = candidate
                                    best_action = (
                                        "two_for_one", source, target,
                                        first, second, one
                                    )

        if best_action is None:
            break

        kind = best_action[0]

        if kind == "move":
            _, source, target, idx = best_action
            bins[source].remove(idx)
            bins[target].append(idx)
            used[source] -= sizes[idx]
            load[source] -= weights[idx]
            used[target] += sizes[idx]
            load[target] += weights[idx]

        elif kind == "swap":
            _, source, target, left, right = best_action
            bins[source].remove(left)
            bins[target].remove(right)
            bins[source].append(right)
            bins[target].append(left)
            used[source] += sizes[right] - sizes[left]
            load[source] += weights[right] - weights[left]
            used[target] += sizes[left] - sizes[right]
            load[target] += weights[left] - weights[right]

        elif kind == "one_for_two":
            _, source, target, one, first, second = best_action
            bins[source].remove(one)
            bins[target].remove(first)
            bins[target].remove(second)
            bins[source].extend([first, second])
            bins[target].append(one)

            used[source] += sizes[first] + sizes[second] - sizes[one]
            load[source] += weights[first] + weights[second] - weights[one]
            used[target] += sizes[one] - sizes[first] - sizes[second]
            load[target] += weights[one] - weights[first] - weights[second]

        else:  # two_for_one
            _, source, target, first, second, one = best_action
            bins[source].remove(first)
            bins[source].remove(second)
            bins[target].remove(one)
            bins[source].append(one)
            bins[target].extend([first, second])

            used[source] += sizes[one] - sizes[first] - sizes[second]
            load[source] += weights[one] - weights[first] - weights[second]
            used[target] += sizes[first] + sizes[second] - sizes[one]
            load[target] += weights[first] + weights[second] - weights[one]

        current = best_value

    return {
        gpu_id: [model_list[idx] for idx in bins[gpu_id]]
        for gpu_id in range(gpu_num)
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