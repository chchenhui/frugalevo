GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place models while minimizing the maximum KV-cache pressure ratio."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    EPS = 1e-12
    model_list = list(models)
    n = len(model_list)
    sizes = [model.model_size for model in model_list]
    weights = [model.req_rate / model.slo for model in model_list]

    if any(size >= GPU_MEM_SIZE for size in sizes):
        bad_size = next(size for size in sizes if size >= GPU_MEM_SIZE)
        raise ValueError(
            f"Unable to place model of size {bad_size} GB while preserving "
            "positive KV-cache memory"
        )

    def ratio(load, remaining):
        return load / remaining if remaining > EPS else float("inf")

    def profile(loads, remaining):
        return tuple(sorted(
            (ratio(loads[g], remaining[g]) for g in range(gpu_num)),
            reverse=True,
        ))

    def changed_profile(current, changes):
        return tuple(sorted(
            (changes[g] if g in changes else current[g] for g in range(gpu_num)),
            reverse=True,
        ))

    def solve(order):
        bins = [[] for _ in range(gpu_num)]
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        loads = [0.0] * gpu_num

        # Greedy placement minimizes the resulting global KVPR bottleneck.
        for idx in order:
            size = sizes[idx]
            weight = weights[idx]
            current = [ratio(loads[g], remaining[g]) for g in range(gpu_num)]
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                free_after = remaining[gpu] - size
                if free_after <= EPS:
                    continue

                new_ratio = ratio(loads[gpu] + weight, free_after)
                candidate = current[:]
                candidate[gpu] = new_ratio
                key = (
                    max(candidate),
                    tuple(sorted(candidate, reverse=True)),
                    -free_after,
                    gpu,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            bins[best_gpu].append(idx)
            remaining[best_gpu] -= size
            loads[best_gpu] += weight

        # Best-improvement local search over moves, swaps, and small exchanges.
        for _ in range(max(1, 2 * n)):
            current = [ratio(loads[g], remaining[g]) for g in range(gpu_num)]
            best_profile = tuple(sorted(current, reverse=True))
            best_action = None

            for source in range(gpu_num):
                for source_pos, idx in enumerate(bins[source]):
                    size = sizes[idx]
                    weight = weights[idx]

                    # One-model relocation.
                    for target in range(gpu_num):
                        if target == source or remaining[target] - size <= EPS:
                            continue

                        source_ratio = ratio(
                            loads[source] - weight, remaining[source] + size
                        )
                        target_ratio = ratio(
                            loads[target] + weight, remaining[target] - size
                        )
                        candidate = changed_profile(
                            current, {source: source_ratio, target: target_ratio}
                        )
                        if candidate < best_profile:
                            best_profile = candidate
                            best_action = ("move", source, source_pos, target)

                    # One-for-one model swap.
                    for target in range(source + 1, gpu_num):
                        for target_pos, other in enumerate(bins[target]):
                            source_free = remaining[source] + size - sizes[other]
                            target_free = remaining[target] + sizes[other] - size
                            if source_free <= EPS or target_free <= EPS:
                                continue

                            source_ratio = ratio(
                                loads[source] - weight + weights[other], source_free
                            )
                            target_ratio = ratio(
                                loads[target] - weights[other] + weight, target_free
                            )
                            candidate = changed_profile(
                                current, {source: source_ratio, target: target_ratio}
                            )
                            if candidate < best_profile:
                                best_profile = candidate
                                best_action = (
                                    "swap", source, source_pos, target, target_pos
                                )

            # A bottleneck may improve only by exchanging two small models for
            # one larger/lower-pressure model.
            if best_action is None and n <= 70:
                bottleneck = max(current)
                for source in range(gpu_num):
                    if current[source] < bottleneck - EPS:
                        continue
                    source_bin = bins[source]
                    for first_pos in range(len(source_bin)):
                        first = source_bin[first_pos]
                        for second_pos in range(first_pos + 1, len(source_bin)):
                            second = source_bin[second_pos]
                            for target in range(gpu_num):
                                if target == source:
                                    continue
                                for other in bins[target]:
                                    source_free = (
                                        remaining[source] + sizes[first] + sizes[second]
                                        - sizes[other]
                                    )
                                    target_free = (
                                        remaining[target] + sizes[other]
                                        - sizes[first] - sizes[second]
                                    )
                                    if source_free <= EPS or target_free <= EPS:
                                        continue

                                    source_ratio = ratio(
                                        loads[source] - weights[first] - weights[second]
                                        + weights[other],
                                        source_free,
                                    )
                                    target_ratio = ratio(
                                        loads[target] - weights[other]
                                        + weights[first] + weights[second],
                                        target_free,
                                    )
                                    candidate = changed_profile(
                                        current,
                                        {source: source_ratio, target: target_ratio},
                                    )
                                    if candidate < best_profile:
                                        best_profile = candidate
                                        best_action = (
                                            "two_for_one",
                                            source, target, first, second, other,
                                        )

            # Evacuate a target model to a third GPU, then move a bottleneck
            # model into the newly created space.
            if best_action is None and n <= 160:
                for source in range(gpu_num):
                    for source_pos, idx in enumerate(bins[source]):
                        size = sizes[idx]
                        weight = weights[idx]
                        source_ratio = ratio(
                            loads[source] - weight, remaining[source] + size
                        )

                        for target in range(gpu_num):
                            if target == source:
                                continue
                            for target_pos, other in enumerate(bins[target]):
                                target_free = remaining[target] + sizes[other] - size
                                if target_free <= EPS:
                                    continue

                                for destination in range(gpu_num):
                                    if destination == source or destination == target:
                                        continue
                                    destination_free = (
                                        remaining[destination] - sizes[other]
                                    )
                                    if destination_free <= EPS:
                                        continue

                                    target_ratio = ratio(
                                        loads[target] - weights[other] + weight,
                                        target_free,
                                    )
                                    destination_ratio = ratio(
                                        loads[destination] + weights[other],
                                        destination_free,
                                    )
                                    candidate = changed_profile(
                                        current,
                                        {
                                            source: source_ratio,
                                            target: target_ratio,
                                            destination: destination_ratio,
                                        },
                                    )
                                    if candidate < best_profile:
                                        best_profile = candidate
                                        best_action = (
                                            "chain",
                                            source, source_pos,
                                            target, target_pos,
                                            destination,
                                        )

            if best_action is None:
                break

            action = best_action[0]
            if action == "move":
                _, source, source_pos, target = best_action
                idx = bins[source].pop(source_pos)
                bins[target].append(idx)
                loads[source] -= weights[idx]
                loads[target] += weights[idx]
                remaining[source] += sizes[idx]
                remaining[target] -= sizes[idx]

            elif action == "swap":
                _, source, source_pos, target, target_pos = best_action
                idx = bins[source][source_pos]
                other = bins[target][target_pos]
                bins[source][source_pos] = other
                bins[target][target_pos] = idx
                loads[source] += weights[other] - weights[idx]
                loads[target] += weights[idx] - weights[other]
                remaining[source] += sizes[idx] - sizes[other]
                remaining[target] += sizes[other] - sizes[idx]

            elif action == "two_for_one":
                _, source, target, first, second, other = best_action
                bins[source].remove(first)
                bins[source].remove(second)
                bins[target].remove(other)
                bins[source].append(other)
                bins[target].extend([first, second])
                loads[source] += weights[other] - weights[first] - weights[second]
                loads[target] += weights[first] + weights[second] - weights[other]
                remaining[source] += sizes[first] + sizes[second] - sizes[other]
                remaining[target] += sizes[other] - sizes[first] - sizes[second]

            else:
                _, source, source_pos, target, target_pos, destination = best_action
                idx = bins[source].pop(source_pos)
                other = bins[target].pop(target_pos)
                bins[target].append(idx)
                bins[destination].append(other)
                loads[source] -= weights[idx]
                loads[target] += weights[idx] - weights[other]
                loads[destination] += weights[other]
                remaining[source] += sizes[idx]
                remaining[target] += sizes[other] - sizes[idx]
                remaining[destination] -= sizes[other]

        return profile(loads, remaining), bins

    # The aggregate lower bound is the smallest possible pressure even if
    # memory and request load could be divided fractionally.  At this pressure,
    # feasibility has the transformed bin capacity form:
    # demand + bound * size <= bound * GPU_MEM_SIZE.
    capacity_gap = max(
        EPS, gpu_num * GPU_MEM_SIZE - sum(sizes)
    )
    pressure_lower_bound = sum(weights) / capacity_gap

    orderings = [
        sorted(range(n), key=lambda i: (-weights[i], -sizes[i], i)),
        sorted(range(n), key=lambda i: (-sizes[i], -weights[i], i)),
        sorted(
            range(n),
            key=lambda i: (-weights[i] / max(sizes[i], EPS), -sizes[i], i),
        ),
        sorted(
            range(n),
            key=lambda i: (
                -sizes[i] / GPU_MEM_SIZE,
                -weights[i] / max(GPU_MEM_SIZE - sizes[i], EPS),
                i,
            ),
        ),
        sorted(
            range(n),
            key=lambda i: (
                -(sizes[i] > GPU_MEM_SIZE / 2),
                -sizes[i],
                -weights[i],
                i,
            ),
        ),
        # Largest transformed item first at the theoretical pressure bound.
        sorted(
            range(n),
            key=lambda i: (
                -(weights[i] + pressure_lower_bound * sizes[i]),
                -sizes[i],
                -weights[i],
                i,
            ),
        ),
    ]

    best_profile = None
    best_bins = None
    for order in orderings:
        result = solve(order)
        if result is not None and (
            best_profile is None or result[0] < best_profile
        ):
            best_profile, best_bins = result

    # Memory-first fallback preserves feasible-placement behavior when all
    # pressure-oriented greedy starts encounter bin-packing fragmentation.
    if best_bins is None:
        bins = [[] for _ in range(gpu_num)]
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        for idx in sorted(range(n), key=lambda i: (-sizes[i], i)):
            choices = [
                gpu for gpu in range(gpu_num)
                if remaining[gpu] - sizes[idx] > EPS
            ]
            if not choices:
                raise ValueError(
                    f"Unable to place model of size {sizes[idx]} GB on any GPU. "
                    f"Remaining per-GPU memory: {remaining}"
                )
            gpu = min(choices, key=lambda g: (remaining[g] - sizes[idx], g))
            bins[gpu].append(idx)
            remaining[gpu] -= sizes[idx]
        best_bins = bins

    return {
        gpu: [model_list[idx] for idx in best_bins[gpu]]
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