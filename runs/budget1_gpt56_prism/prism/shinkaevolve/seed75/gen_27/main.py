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

    sizes = [float(model.model_size) for model in model_list]
    weights = [float(model.req_rate / model.slo) for model in model_list]

    for size in sizes:
        if size > GPU_MEM_SIZE + eps:
            raise ValueError(
                f"Unable to place model of size {size} GB on any GPU."
            )

    def pressure(load, used):
        remaining = GPU_MEM_SIZE - used
        if remaining <= eps:
            return float("inf") if load > eps else 0.0
        return load / remaining

    def profile(used, load):
        return tuple(sorted(
            (pressure(load[g], used[g]) for g in range(gpu_num)),
            reverse=True,
        ))

    def state_key(used, load):
        return tuple(sorted(
            (round(used[g], 7), round(load[g], 7))
            for g in range(gpu_num)
        ))

    def construct(order, beam_width):
        beam = [([[] for _ in range(gpu_num)],
                 [0.0] * gpu_num,
                 [0.0] * gpu_num)]

        for idx in order:
            candidates = []
            seen = set()

            for bins, used, load in beam:
                for g in range(gpu_num):
                    if used[g] + sizes[idx] > GPU_MEM_SIZE + eps:
                        continue

                    next_used = used[:]
                    next_load = load[:]
                    next_used[g] += sizes[idx]
                    next_load[g] += weights[idx]

                    next_bins = [bucket[:] for bucket in bins]
                    next_bins[g].append(idx)

                    candidate_profile = profile(next_used, next_load)
                    residuals = tuple(sorted(
                        GPU_MEM_SIZE - value for value in next_used
                    ))

                    candidates.append((
                        candidate_profile,
                        residuals,
                        next_bins,
                        next_used,
                        next_load,
                    ))

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

    orders = [
        sorted(range(n), key=lambda i: (-sizes[i], -weights[i], i)),
        sorted(range(n), key=lambda i: (-weights[i], -sizes[i], i)),
        sorted(range(n), key=lambda i: (
            -(weights[i] / max(sizes[i], eps)), -sizes[i], i
        )),
        sorted(range(n), key=lambda i: (
            -(weights[i] * sizes[i]), -sizes[i], i
        )),
        sorted(range(n), key=lambda i: (
            -(weights[i] / max(GPU_MEM_SIZE - sizes[i], eps)),
            -sizes[i],
            i,
        )),
        sorted(range(n), key=lambda i: (
            -max(weights[i], sizes[i] / GPU_MEM_SIZE),
            -weights[i],
            -sizes[i],
            i,
        )),
    ]

    best_state = None
    best_profile = None

    for order in orders:
        state = construct(order, beam_width=12)
        if state is None:
            continue

        candidate_profile = profile(state[1], state[2])
        if best_profile is None or candidate_profile < best_profile:
            best_state = state
            best_profile = candidate_profile

    if best_state is None:
        bins = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num

        for idx in sorted(range(n), key=lambda i: (-sizes[i], -weights[i], i)):
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

            target = min(
                choices,
                key=lambda g: (
                    GPU_MEM_SIZE - used[g] - sizes[idx],
                    pressure(load[g] + weights[idx], used[g] + sizes[idx]),
                    g,
                ),
            )
            bins[target].append(idx)
            used[target] += sizes[idx]
            load[target] += weights[idx]
    else:
        bins, used, load = best_state

    current_profile = profile(used, load)

    def candidate_profile(changes):
        trial_used = used[:]
        trial_load = load[:]

        for gpu, mem_delta, load_delta in changes:
            trial_used[gpu] += mem_delta
            trial_load[gpu] += load_delta

        return profile(trial_used, trial_load)

    def apply_move(source, target, idx):
        bins[source].remove(idx)
        bins[target].append(idx)
        used[source] -= sizes[idx]
        load[source] -= weights[idx]
        used[target] += sizes[idx]
        load[target] += weights[idx]

    def apply_swap(left_gpu, right_gpu, left_idx, right_idx):
        bins[left_gpu].remove(left_idx)
        bins[right_gpu].remove(right_idx)
        bins[left_gpu].append(right_idx)
        bins[right_gpu].append(left_idx)

        used[left_gpu] += sizes[right_idx] - sizes[left_idx]
        load[left_gpu] += weights[right_idx] - weights[left_idx]
        used[right_gpu] += sizes[left_idx] - sizes[right_idx]
        load[right_gpu] += weights[left_idx] - weights[right_idx]

    max_rounds = min(60, max(20, n * 2))

    for _ in range(max_rounds):
        pressures = [pressure(load[g], used[g]) for g in range(gpu_num)]
        maximum = max(pressures)

        threshold = maximum - max(1e-12, abs(maximum) * 0.05)
        source_gpus = [
            g for g in range(gpu_num)
            if bins[g] and pressures[g] >= threshold
        ]

        if len(source_gpus) < min(3, gpu_num):
            ranked = sorted(
                (g for g in range(gpu_num) if bins[g]),
                key=lambda g: pressures[g],
                reverse=True,
            )
            source_gpus = list(dict.fromkeys(
                source_gpus + ranked[:min(3, len(ranked))]
            ))

        best_action = None
        best_value = current_profile

        for source in source_gpus:
            for idx in bins[source]:
                for target in range(gpu_num):
                    if target == source:
                        continue
                    if used[target] + sizes[idx] > GPU_MEM_SIZE + eps:
                        continue

                    value = candidate_profile([
                        (source, -sizes[idx], -weights[idx]),
                        (target, sizes[idx], weights[idx]),
                    ])

                    if value < best_value:
                        best_value = value
                        best_action = ("move", source, target, idx)

        if best_action is None and n <= 220:
            for source in source_gpus:
                for target in range(gpu_num):
                    if target == source:
                        continue

                    for left_idx in bins[source]:
                        for right_idx in bins[target]:
                            source_used = (
                                used[source] - sizes[left_idx] + sizes[right_idx]
                            )
                            target_used = (
                                used[target] - sizes[right_idx] + sizes[left_idx]
                            )

                            if (source_used > GPU_MEM_SIZE + eps or
                                    target_used > GPU_MEM_SIZE + eps):
                                continue

                            value = candidate_profile([
                                (
                                    source,
                                    sizes[right_idx] - sizes[left_idx],
                                    weights[right_idx] - weights[left_idx],
                                ),
                                (
                                    target,
                                    sizes[left_idx] - sizes[right_idx],
                                    weights[left_idx] - weights[right_idx],
                                ),
                            ])

                            if value < best_value:
                                best_value = value
                                best_action = (
                                    "swap", source, target, left_idx, right_idx
                                )

        if best_action is None and n <= 110:
            for source in source_gpus:
                for target in range(gpu_num):
                    if target == source:
                        continue

                    source_models = bins[source]
                    target_models = bins[target]

                    for first_pos in range(len(source_models)):
                        first = source_models[first_pos]
                        for second_pos in range(first_pos + 1, len(source_models)):
                            second = source_models[second_pos]

                            outgoing_size = sizes[first] + sizes[second]
                            outgoing_weight = weights[first] + weights[second]

                            for incoming in target_models:
                                source_used = (
                                    used[source] - outgoing_size + sizes[incoming]
                                )
                                target_used = (
                                    used[target] - sizes[incoming] + outgoing_size
                                )

                                if (source_used > GPU_MEM_SIZE + eps or
                                        target_used > GPU_MEM_SIZE + eps):
                                    continue

                                value = candidate_profile([
                                    (
                                        source,
                                        sizes[incoming] - outgoing_size,
                                        weights[incoming] - outgoing_weight,
                                    ),
                                    (
                                        target,
                                        outgoing_size - sizes[incoming],
                                        outgoing_weight - weights[incoming],
                                    ),
                                ])

                                if value < best_value:
                                    best_value = value
                                    best_action = (
                                        "two_for_one",
                                        source,
                                        target,
                                        first,
                                        second,
                                        incoming,
                                    )

        if best_action is None and n <= 110:
            for source in source_gpus:
                for target in range(gpu_num):
                    if target == source:
                        continue

                    for outgoing in bins[source]:
                        target_models = bins[target]
                        for first_pos in range(len(target_models)):
                            first = target_models[first_pos]
                            for second_pos in range(
                                first_pos + 1, len(target_models)
                            ):
                                second = target_models[second_pos]

                                incoming_size = sizes[first] + sizes[second]
                                incoming_weight = (
                                    weights[first] + weights[second]
                                )

                                source_used = (
                                    used[source] - sizes[outgoing]
                                    + incoming_size
                                )
                                target_used = (
                                    used[target] - incoming_size
                                    + sizes[outgoing]
                                )

                                if (source_used > GPU_MEM_SIZE + eps or
                                        target_used > GPU_MEM_SIZE + eps):
                                    continue

                                value = candidate_profile([
                                    (
                                        source,
                                        incoming_size - sizes[outgoing],
                                        incoming_weight - weights[outgoing],
                                    ),
                                    (
                                        target,
                                        sizes[outgoing] - incoming_size,
                                        weights[outgoing] - incoming_weight,
                                    ),
                                ])

                                if value < best_value:
                                    best_value = value
                                    best_action = (
                                        "one_for_two",
                                        source,
                                        target,
                                        outgoing,
                                        first,
                                        second,
                                    )

        if best_action is None:
            break

        if best_action[0] == "move":
            _, source, target, idx = best_action
            apply_move(source, target, idx)

        elif best_action[0] == "swap":
            _, source, target, left_idx, right_idx = best_action
            apply_swap(source, target, left_idx, right_idx)

        elif best_action[0] == "two_for_one":
            _, source, target, first, second, incoming = best_action

            bins[source].remove(first)
            bins[source].remove(second)
            bins[target].remove(incoming)

            bins[source].append(incoming)
            bins[target].extend([first, second])

            outgoing_size = sizes[first] + sizes[second]
            outgoing_weight = weights[first] + weights[second]

            used[source] += sizes[incoming] - outgoing_size
            load[source] += weights[incoming] - outgoing_weight
            used[target] += outgoing_size - sizes[incoming]
            load[target] += outgoing_weight - weights[incoming]

        else:  # one_for_two
            _, source, target, outgoing, first, second = best_action

            bins[source].remove(outgoing)
            bins[target].remove(first)
            bins[target].remove(second)

            bins[source].extend([first, second])
            bins[target].append(outgoing)

            incoming_size = sizes[first] + sizes[second]
            incoming_weight = weights[first] + weights[second]

            used[source] += incoming_size - sizes[outgoing]
            load[source] += incoming_weight - weights[outgoing]
            used[target] += sizes[outgoing] - incoming_size
            load[target] += weights[outgoing] - incoming_weight

        current_profile = best_value

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