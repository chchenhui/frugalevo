GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Assign models to GPUs while minimizing the lexicographically sorted KVPR
    profile: lowest maximum KVPR first, then lowest secondary pressure.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    model_list = list(models)
    n = len(model_list)
    eps = 1e-10

    sizes = [model.model_size for model in model_list]
    weights = [model.req_rate / model.slo for model in model_list]

    for size in sizes:
        if size > GPU_MEM_SIZE + eps:
            raise ValueError(
                f"Unable to place model of size {size} GB on any GPU."
            )

    def pressure(weight, used):
        remaining = GPU_MEM_SIZE - used
        if remaining <= eps:
            return float("inf") if weight > eps else 0.0
        return weight / remaining

    def make_profile(used_mem, gpu_weight):
        return tuple(sorted(
            (pressure(gpu_weight[g], used_mem[g]) for g in range(gpu_num)),
            reverse=True,
        ))

    def state_profile_after(used_mem, gpu_weight, changes):
        """
        changes maps GPU id to (new_used_memory, new_weight).
        Only changed GPU pressures need recomputation.
        """
        values = []
        for g in range(gpu_num):
            if g in changes:
                used, weight = changes[g]
                values.append(pressure(weight, used))
            else:
                values.append(pressure(gpu_weight[g], used_mem[g]))
        return tuple(sorted(values, reverse=True))

    def construct(order):
        bins = [[] for _ in range(gpu_num)]
        used_mem = [0.0] * gpu_num
        gpu_weight = [0.0] * gpu_num

        for idx in order:
            best_gpu = None
            best_key = None

            for g in range(gpu_num):
                new_used = used_mem[g] + sizes[idx]
                if new_used > GPU_MEM_SIZE + eps:
                    continue

                new_weight = gpu_weight[g] + weights[idx]
                candidate = state_profile_after(
                    used_mem,
                    gpu_weight,
                    {g: (new_used, new_weight)},
                )

                key = (
                    candidate,
                    GPU_MEM_SIZE - new_used,
                    g,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = g

            if best_gpu is None:
                return None

            bins[best_gpu].append(idx)
            used_mem[best_gpu] += sizes[idx]
            gpu_weight[best_gpu] += weights[idx]

        return bins, used_mem, gpu_weight

    indices = list(range(n))
    orderings = [
        sorted(indices, key=lambda i: (-weights[i], -sizes[i], i)),
        sorted(indices, key=lambda i: (-sizes[i], -weights[i], i)),
        sorted(
            indices,
            key=lambda i: (
                -(weights[i] / max(sizes[i], eps)),
                -sizes[i],
                -weights[i],
                i,
            ),
        ),
        sorted(
            indices,
            key=lambda i: (
                -(weights[i] * sizes[i]),
                -sizes[i],
                i,
            ),
        ),
    ]

    best_state = None
    best_profile = None
    for order in orderings:
        state = construct(order)
        if state is None:
            continue
        candidate_profile = make_profile(state[1], state[2])
        if best_profile is None or candidate_profile < best_profile:
            best_state = state
            best_profile = candidate_profile

    if best_state is None:
        bins = [[] for _ in range(gpu_num)]
        used_mem = [0.0] * gpu_num
        gpu_weight = [0.0] * gpu_num

        for idx in sorted(indices, key=lambda i: (-sizes[i], i)):
            feasible = [
                g for g in range(gpu_num)
                if used_mem[g] + sizes[idx] <= GPU_MEM_SIZE + eps
            ]
            if not feasible:
                remaining = [GPU_MEM_SIZE - used for used in used_mem]
                raise ValueError(
                    f"Unable to place model of size {sizes[idx]} GB on any GPU. "
                    f"Remaining per-GPU memory: {remaining}"
                )

            target = min(
                feasible,
                key=lambda g: (GPU_MEM_SIZE - used_mem[g] - sizes[idx], g),
            )
            bins[target].append(idx)
            used_mem[target] += sizes[idx]
            gpu_weight[target] += weights[idx]
    else:
        bins, used_mem, gpu_weight = best_state

    current_profile = make_profile(used_mem, gpu_weight)

    def apply_move(source, target, idx):
        bins[source].remove(idx)
        bins[target].append(idx)
        used_mem[source] -= sizes[idx]
        used_mem[target] += sizes[idx]
        gpu_weight[source] -= weights[idx]
        gpu_weight[target] += weights[idx]

    def apply_swap(left_gpu, right_gpu, left_idx, right_idx):
        bins[left_gpu].remove(left_idx)
        bins[right_gpu].remove(right_idx)
        bins[left_gpu].append(right_idx)
        bins[right_gpu].append(left_idx)

        used_mem[left_gpu] += sizes[right_idx] - sizes[left_idx]
        used_mem[right_gpu] += sizes[left_idx] - sizes[right_idx]
        gpu_weight[left_gpu] += weights[right_idx] - weights[left_idx]
        gpu_weight[right_gpu] += weights[left_idx] - weights[right_idx]

    for _ in range(30):
        best_profile = current_profile
        best_action = None

        # Best feasible single-model relocation.
        for source in range(gpu_num):
            for idx in bins[source]:
                for target in range(gpu_num):
                    if target == source:
                        continue

                    target_used = used_mem[target] + sizes[idx]
                    if target_used > GPU_MEM_SIZE + eps:
                        continue

                    candidate = state_profile_after(
                        used_mem,
                        gpu_weight,
                        {
                            source: (
                                used_mem[source] - sizes[idx],
                                gpu_weight[source] - weights[idx],
                            ),
                            target: (
                                target_used,
                                gpu_weight[target] + weights[idx],
                            ),
                        },
                    )
                    if candidate < best_profile:
                        best_profile = candidate
                        best_action = ("move", source, target, idx)

        if best_action is not None:
            _, source, target, idx = best_action
            apply_move(source, target, idx)
            current_profile = best_profile
            continue

        # Swaps can improve pressure where a direct move cannot fit.
        if n <= 160:
            for left_gpu in range(gpu_num):
                for right_gpu in range(left_gpu + 1, gpu_num):
                    for left_idx in bins[left_gpu]:
                        for right_idx in bins[right_gpu]:
                            left_used = (
                                used_mem[left_gpu]
                                - sizes[left_idx]
                                + sizes[right_idx]
                            )
                            right_used = (
                                used_mem[right_gpu]
                                - sizes[right_idx]
                                + sizes[left_idx]
                            )
                            if (left_used > GPU_MEM_SIZE + eps or
                                    right_used > GPU_MEM_SIZE + eps):
                                continue

                            candidate = state_profile_after(
                                used_mem,
                                gpu_weight,
                                {
                                    left_gpu: (
                                        left_used,
                                        gpu_weight[left_gpu]
                                        - weights[left_idx]
                                        + weights[right_idx],
                                    ),
                                    right_gpu: (
                                        right_used,
                                        gpu_weight[right_gpu]
                                        - weights[right_idx]
                                        + weights[left_idx],
                                    ),
                                },
                            )
                            if candidate < best_profile:
                                best_profile = candidate
                                best_action = (
                                    "swap",
                                    left_gpu,
                                    right_gpu,
                                    left_idx,
                                    right_idx,
                                )

        if best_action is not None:
            _, left_gpu, right_gpu, left_idx, right_idx = best_action
            apply_swap(left_gpu, right_gpu, left_idx, right_idx)
            current_profile = best_profile
            continue

        # A bounded two-model relocation escapes packing local minima.  It is
        # intentionally enabled only for moderate workloads.
        if n <= 80:
            for source in range(gpu_num):
                source_items = bins[source]
                for first_pos in range(len(source_items)):
                    first = source_items[first_pos]
                    for second_pos in range(first_pos + 1, len(source_items)):
                        second = source_items[second_pos]
                        moved_size = sizes[first] + sizes[second]
                        moved_weight = weights[first] + weights[second]

                        for target in range(gpu_num):
                            if target == source:
                                continue

                            target_used = used_mem[target] + moved_size
                            if target_used > GPU_MEM_SIZE + eps:
                                continue

                            candidate = state_profile_after(
                                used_mem,
                                gpu_weight,
                                {
                                    source: (
                                        used_mem[source] - moved_size,
                                        gpu_weight[source] - moved_weight,
                                    ),
                                    target: (
                                        target_used,
                                        gpu_weight[target] + moved_weight,
                                    ),
                                },
                            )
                            if candidate < best_profile:
                                best_profile = candidate
                                best_action = (
                                    "pair_move",
                                    source,
                                    target,
                                    first,
                                    second,
                                )

        if best_action is None:
            break

        _, source, target, first, second = best_action
        apply_move(source, target, first)
        apply_move(source, target, second)
        current_profile = best_profile

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
