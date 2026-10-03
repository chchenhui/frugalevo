GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models on GPUs while minimizing the maximum KV cache pressure.

    GPU pressure:
        sum(req_rate / slo) / remaining_memory
    """

    EPS = 1e-12

    if not models:
        return {gpu: [] for gpu in range(gpu_num)}

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive when models are provided.")

    n = len(models)
    sizes = [model.model_size for model in models]
    demands = [model.req_rate / model.slo for model in models]

    if any(size > GPU_MEM_SIZE + EPS for size in sizes):
        raise ValueError("A model exceeds the memory capacity of a GPU.")

    def kvpr(load, free_memory):
        if free_memory > EPS:
            return load / free_memory
        return float("inf") if load > EPS else 0.0

    def score_profile(scores):
        return tuple(sorted(scores, reverse=True))

    def strictly_better(candidate, current):
        for left, right in zip(candidate, current):
            if left < right - EPS:
                return True
            if left > right + EPS:
                return False
        return False

    def build_state(owner):
        bins = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        remaining = [GPU_MEM_SIZE] * gpu_num

        for index, gpu in enumerate(owner):
            bins[gpu].append(index)
            loads[gpu] += demands[index]
            remaining[gpu] -= sizes[index]

        scores = [kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)]
        return bins, loads, remaining, scores

    def greedy(order, compact=False):
        owner = [-1] * n
        loads = [0.0] * gpu_num
        remaining = [GPU_MEM_SIZE] * gpu_num
        scores = [0.0] * gpu_num

        for index in order:
            size = sizes[index]
            demand = demands[index]
            best_gpu = None
            best_key = None

            for gpu in range(gpu_num):
                if size > remaining[gpu] + EPS:
                    continue

                new_remaining = remaining[gpu] - size
                new_score = kvpr(loads[gpu] + demand, new_remaining)

                if compact:
                    key = (
                        new_remaining,
                        loads[gpu],
                        gpu,
                    )
                else:
                    candidate_scores = scores[:]
                    candidate_scores[gpu] = new_score
                    key = (
                        score_profile(candidate_scores),
                        new_score,
                        new_remaining,
                        gpu,
                    )

                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu

            if best_gpu is None:
                return None

            owner[index] = best_gpu
            loads[best_gpu] += demand
            remaining[best_gpu] -= size
            scores[best_gpu] = kvpr(loads[best_gpu], remaining[best_gpu])

        return owner

    # Multiple complementary construction orders improve resistance to both
    # high-demand imbalance and large-model fragmentation.
    indices = list(range(n))
    orders = [
        sorted(indices, key=lambda i: demands[i], reverse=True),
        sorted(indices, key=lambda i: sizes[i], reverse=True),
        sorted(indices, key=lambda i: demands[i] / max(sizes[i], EPS), reverse=True),
        sorted(
            indices,
            key=lambda i: demands[i] / max(GPU_MEM_SIZE - sizes[i], EPS),
            reverse=True,
        ),
        sorted(indices, key=lambda i: (demands[i], sizes[i]), reverse=True),
        sorted(indices, key=lambda i: (sizes[i], demands[i]), reverse=True),
        sorted(
            indices,
            key=lambda i: (
                0.70 * demands[i] / max(GPU_MEM_SIZE - sizes[i], EPS)
                + 0.30 * sizes[i] / GPU_MEM_SIZE
            ),
            reverse=True,
        ),
    ]

    candidates = []
    for order in orders:
        result = greedy(order, compact=False)
        if result is not None:
            candidates.append(result)

    for order in (orders[1], orders[5], orders[3]):
        result = greedy(order, compact=True)
        if result is not None:
            candidates.append(result)

    if not candidates:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs "
            f"with {GPU_MEM_SIZE} GB each."
        )

    owner = min(
        candidates,
        key=lambda candidate: score_profile(build_state(candidate)[3]),
    )

    bins, loads, remaining, scores = build_state(owner)

    def evaluate_changes(changes):
        """
        changes is {model_index: destination_gpu}.
        Return a profile and affected state values, or None if infeasible.
        """
        changed_gpus = set()
        next_loads = {}
        next_remaining = {}

        for model_index, destination in changes.items():
            source = owner[model_index]
            if source == destination:
                continue

            if source not in next_loads:
                next_loads[source] = loads[source]
                next_remaining[source] = remaining[source]
            if destination not in next_loads:
                next_loads[destination] = loads[destination]
                next_remaining[destination] = remaining[destination]

            next_loads[source] -= demands[model_index]
            next_remaining[source] += sizes[model_index]
            next_loads[destination] += demands[model_index]
            next_remaining[destination] -= sizes[model_index]
            changed_gpus.add(source)
            changed_gpus.add(destination)

        if not changed_gpus:
            return None

        for gpu in changed_gpus:
            if next_remaining[gpu] < -EPS:
                return None

        candidate_scores = scores[:]
        for gpu in changed_gpus:
            candidate_scores[gpu] = kvpr(
                next_loads[gpu],
                max(0.0, next_remaining[gpu]),
            )

        return score_profile(candidate_scores)

    def consider(changes, best_profile, best_changes):
        candidate = evaluate_changes(changes)
        if candidate is not None and strictly_better(candidate, best_profile):
            return candidate, changes
        return best_profile, best_changes

    # Model limits retain fast execution when a GPU hosts many small models.
    EXCHANGE_LIMIT = 14
    EJECTION_LIMIT = 10
    THIRD_GPU_LIMIT = 4

    while True:
        current_profile = score_profile(scores)
        best_profile = current_profile
        best_changes = None

        gpu_desc = sorted(range(gpu_num), key=lambda gpu: scores[gpu], reverse=True)
        gpu_asc = list(reversed(gpu_desc))

        ranked_models = [
            sorted(
                bins[gpu],
                key=lambda i: (
                    kvpr(
                        demands[i],
                        max(EPS, GPU_MEM_SIZE - sizes[i]),
                    ),
                    demands[i],
                    sizes[i],
                ),
                reverse=True,
            )
            for gpu in range(gpu_num)
        ]

        # Direct relocations.
        for source in gpu_desc:
            for model_index in ranked_models[source]:
                for target in range(gpu_num):
                    if target == source:
                        continue
                    if sizes[model_index] <= remaining[target] + EPS:
                        best_profile, best_changes = consider(
                            {model_index: target},
                            best_profile,
                            best_changes,
                        )

        # Pairwise swaps.
        for first in range(gpu_num):
            for second in range(first + 1, gpu_num):
                for a in ranked_models[first][:EXCHANGE_LIMIT]:
                    for b in ranked_models[second][:EXCHANGE_LIMIT]:
                        if sizes[b] > remaining[first] + sizes[a] + EPS:
                            continue
                        if sizes[a] > remaining[second] + sizes[b] + EPS:
                            continue
                        best_profile, best_changes = consider(
                            {a: second, b: first},
                            best_profile,
                            best_changes,
                        )

        # Asymmetric exchanges centered on high-pressure source GPUs.
        for source in gpu_desc:
            source_models = ranked_models[source][:EXCHANGE_LIMIT]
            if not source_models:
                continue

            for target in range(gpu_num):
                if target == source:
                    continue

                target_models = ranked_models[target][:EXCHANGE_LIMIT]

                # Two source models exchanged for one target model.
                for a_pos in range(len(source_models)):
                    for b_pos in range(a_pos + 1, len(source_models)):
                        a = source_models[a_pos]
                        b = source_models[b_pos]
                        for incoming in target_models:
                            best_profile, best_changes = consider(
                                {
                                    a: target,
                                    b: target,
                                    incoming: source,
                                },
                                best_profile,
                                best_changes,
                            )

                # One source model exchanged for two target models.
                for outgoing in source_models:
                    for a_pos in range(len(target_models)):
                        for b_pos in range(a_pos + 1, len(target_models)):
                            a = target_models[a_pos]
                            b = target_models[b_pos]
                            best_profile, best_changes = consider(
                                {
                                    outgoing: target,
                                    a: source,
                                    b: source,
                                },
                                best_profile,
                                best_changes,
                            )

        # Three-GPU ejection chain:
        # source model -> target, target small model -> third GPU.
        for source in gpu_desc:
            source_models = ranked_models[source][:EJECTION_LIMIT]

            for target in gpu_asc:
                if target == source:
                    continue

                ejected_models = sorted(
                    bins[target],
                    key=lambda i: (sizes[i], demands[i]),
                )[:EJECTION_LIMIT]

                for outgoing in source_models:
                    if sizes[outgoing] <= remaining[target] + EPS:
                        continue

                    for ejected in ejected_models:
                        if sizes[outgoing] > remaining[target] + sizes[ejected] + EPS:
                            continue

                        third_candidates = [
                            gpu for gpu in gpu_asc
                            if gpu != source and gpu != target
                        ][:THIRD_GPU_LIMIT]

                        for third in third_candidates:
                            if sizes[ejected] > remaining[third] + EPS:
                                continue

                            best_profile, best_changes = consider(
                                {
                                    outgoing: target,
                                    ejected: third,
                                },
                                best_profile,
                                best_changes,
                            )

        if best_changes is None:
            break

        for model_index, destination in best_changes.items():
            owner[model_index] = destination

        bins, loads, remaining, scores = build_state(owner)

    return {
        gpu: [models[index] for index in bins[gpu]]
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