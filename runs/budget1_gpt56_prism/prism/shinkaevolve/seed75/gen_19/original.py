GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place models to minimize the maximum KV-cache pressure ratio."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    EPS = 1e-12
    model_list = list(models)

    def demand(model):
        return model.req_rate / model.slo

    def ratio(load, memory):
        if memory <= 0:
            return float("inf") if load > 0 else 0.0
        return load / memory

    def solve(order):
        placement = {gpu: [] for gpu in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        loads = [0.0] * gpu_num

        # Greedy construction: choose the placement with the lowest resulting
        # global maximum KVPR, not merely the lowest current GPU pressure.
        for model in order:
            size = model.model_size
            weight = demand(model)
            best_gpu = None
            best_value = float("inf")
            best_remaining = -1.0

            for gpu in range(gpu_num):
                # Positive remaining KV-cache memory is required.
                if size >= remaining[gpu]:
                    continue

                candidate_value = 0.0
                for other in range(gpu_num):
                    if other == gpu:
                        value = ratio(
                            loads[other] + weight,
                            remaining[other] - size,
                        )
                    else:
                        value = ratio(loads[other], remaining[other])
                    if value > candidate_value:
                        candidate_value = value

                # Preserve larger free regions when objective values tie.
                free_after = remaining[gpu] - size
                if (candidate_value < best_value - EPS or
                        (abs(candidate_value - best_value) <= EPS and
                         free_after > best_remaining)):
                    best_gpu = gpu
                    best_value = candidate_value
                    best_remaining = free_after

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            loads[best_gpu] += weight
            remaining[best_gpu] -= size

        # Strictly improving move and swap local search.
        for _ in range(max(1, 2 * len(model_list))):
            current_ratios = [
                ratio(loads[gpu], remaining[gpu])
                for gpu in range(gpu_num)
            ]
            current_max = max(current_ratios)
            best_max = current_max
            best_action = None

            for source in range(gpu_num):
                for source_index, model in enumerate(placement[source]):
                    size = model.model_size
                    weight = demand(model)

                    # Relocate one model.
                    for target in range(gpu_num):
                        if target == source or size >= remaining[target]:
                            continue

                        source_value = ratio(
                            loads[source] - weight,
                            remaining[source] + size,
                        )
                        target_value = ratio(
                            loads[target] + weight,
                            remaining[target] - size,
                        )
                        value = max(source_value, target_value)
                        for other in range(gpu_num):
                            if other != source and other != target:
                                value = max(value, current_ratios[other])

                        if value < best_max - EPS:
                            best_max = value
                            best_action = ("move", source, source_index, target)

                    # Exchange models between two GPUs.
                    for target in range(source + 1, gpu_num):
                        for target_index, other_model in enumerate(placement[target]):
                            other_size = other_model.model_size
                            other_weight = demand(other_model)

                            source_memory = (
                                remaining[source] + size - other_size
                            )
                            target_memory = (
                                remaining[target] + other_size - size
                            )
                            if source_memory <= 0 or target_memory <= 0:
                                continue

                            source_value = ratio(
                                loads[source] - weight + other_weight,
                                source_memory,
                            )
                            target_value = ratio(
                                loads[target] - other_weight + weight,
                                target_memory,
                            )
                            value = max(source_value, target_value)
                            for gpu in range(gpu_num):
                                if gpu != source and gpu != target:
                                    value = max(value, current_ratios[gpu])

                            if value < best_max - EPS:
                                best_max = value
                                best_action = (
                                    "swap", source, source_index,
                                    target, target_index,
                                )

            if best_action is None:
                break

            if best_action[0] == "move":
                _, source, source_index, target = best_action
                model = placement[source].pop(source_index)
                placement[target].append(model)
                loads[source] -= demand(model)
                loads[target] += demand(model)
                remaining[source] += model.model_size
                remaining[target] -= model.model_size
            else:
                _, source, source_index, target, target_index = best_action
                model = placement[source][source_index]
                other_model = placement[target][target_index]

                placement[source][source_index] = other_model
                placement[target][target_index] = model

                loads[source] += demand(other_model) - demand(model)
                loads[target] += demand(model) - demand(other_model)
                remaining[source] += model.model_size - other_model.model_size
                remaining[target] += other_model.model_size - model.model_size

        objective = max(
            ratio(loads[gpu], remaining[gpu])
            for gpu in range(gpu_num)
        )
        return objective, placement

    # Diverse deterministic starts reduce sensitivity to greedy construction.
    orderings = [
        sorted(
            model_list,
            key=lambda m: (demand(m), m.model_size),
            reverse=True,
        ),
        sorted(
            model_list,
            key=lambda m: (m.model_size, demand(m)),
            reverse=True,
        ),
        sorted(
            model_list,
            key=lambda m: (
                demand(m) / max(m.model_size, 1e-9),
                demand(m),
                m.model_size,
            ),
            reverse=True,
        ),
        sorted(
            model_list,
            key=lambda m: (
                m.model_size / GPU_MEM_SIZE,
                demand(m) / max(GPU_MEM_SIZE - m.model_size, 1e-9),
                demand(m),
            ),
            reverse=True,
        ),
        sorted(
            model_list,
            key=lambda m: (
                m.model_size > GPU_MEM_SIZE / 2,
                m.model_size,
                demand(m),
            ),
            reverse=True,
        ),
    ]

    best_placement = None
    best_objective = float("inf")

    for order in orderings:
        result = solve(order)
        if result is not None:
            objective, placement = result
            if objective < best_objective - EPS:
                best_objective = objective
                best_placement = placement

    if best_placement is None:
        raise ValueError(
            "Unable to place all models while preserving positive KV-cache memory"
        )

    return best_placement

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