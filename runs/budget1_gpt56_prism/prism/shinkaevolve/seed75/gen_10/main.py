GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place models while minimizing the maximum post-placement KVPR."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # Place the models that are hardest to accommodate first.
    sorted_models = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo, m.model_size),
        reverse=True,
    )

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
    pressure = [0.0 for _ in range(gpu_num)]

    # Select by the resulting, not current, KVPR.
    for model in sorted_models:
        demand = model.req_rate / model.slo
        candidates = [
            gpu_id for gpu_id in range(gpu_num)
            if model.model_size < remaining[gpu_id]
        ]
        if not candidates:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        best_idx = min(
            candidates,
            key=lambda gpu_id: (
                (pressure[gpu_id] + demand) /
                (remaining[gpu_id] - model.model_size),
                -remaining[gpu_id],
            ),
        )
        placement[best_idx].append(model)
        pressure[best_idx] += demand
        remaining[best_idx] -= model.model_size

    # Improve the bottleneck with relocations and memory-unlocking swaps.
    for _ in range(len(models)):
        ratios = [
            pressure[gpu_id] / remaining[gpu_id]
            for gpu_id in range(gpu_num)
        ]
        source = max(range(gpu_num), key=lambda gpu_id: ratios[gpu_id])
        best_max = ratios[source]
        best_action = None

        for model in placement[source]:
            demand = model.req_rate / model.slo
            source_ratio = (pressure[source] - demand) / (
                remaining[source] + model.model_size
            )
            for target in range(gpu_num):
                if target == source:
                    continue

                if model.model_size < remaining[target]:
                    target_ratio = (pressure[target] + demand) / (
                        remaining[target] - model.model_size
                    )
                    candidate_max = max(
                        source_ratio,
                        target_ratio,
                        *(ratios[gpu_id] for gpu_id in range(gpu_num)
                          if gpu_id != source and gpu_id != target),
                    )
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_action = ("move", model, target, demand)

                for other in placement[target]:
                    other_demand = other.req_rate / other.slo
                    new_source_remaining = (
                        remaining[source] + model.model_size - other.model_size
                    )
                    new_target_remaining = (
                        remaining[target] + other.model_size - model.model_size
                    )
                    if new_source_remaining <= 0 or new_target_remaining <= 0:
                        continue
                    source_swap_ratio = (
                        pressure[source] - demand + other_demand
                    ) / new_source_remaining
                    target_swap_ratio = (
                        pressure[target] - other_demand + demand
                    ) / new_target_remaining
                    candidate_max = max(
                        source_swap_ratio,
                        target_swap_ratio,
                        *(ratios[gpu_id] for gpu_id in range(gpu_num)
                          if gpu_id != source and gpu_id != target),
                    )
                    if candidate_max < best_max:
                        best_max = candidate_max
                        best_action = (
                            "swap", model, other, target, demand, other_demand
                        )

        # A direct move may be blocked by target memory even when the target
        # can first evacuate one of its models to a third GPU.
        if best_action is None:
            for model in placement[source]:
                demand = model.req_rate / model.slo
                source_ratio = (pressure[source] - demand) / (
                    remaining[source] + model.model_size
                )
                for target in range(gpu_num):
                    if target == source:
                        continue
                    for other in placement[target]:
                        other_demand = other.req_rate / other.slo
                        target_remaining = (
                            remaining[target] + other.model_size -
                            model.model_size
                        )
                        if target_remaining <= 0:
                            continue
                        target_ratio = (
                            pressure[target] - other_demand + demand
                        ) / target_remaining
                        for destination in range(gpu_num):
                            if destination == source or destination == target:
                                continue
                            if other.model_size >= remaining[destination]:
                                continue
                            destination_ratio = (
                                pressure[destination] + other_demand
                            ) / (
                                remaining[destination] - other.model_size
                            )
                            candidate_max = max(
                                source_ratio,
                                target_ratio,
                                destination_ratio,
                                *(
                                    ratios[gpu_id]
                                    for gpu_id in range(gpu_num)
                                    if gpu_id not in (
                                        source, target, destination
                                    )
                                ),
                            )
                            if candidate_max < best_max:
                                best_max = candidate_max
                                best_action = (
                                    "chain", model, target, other,
                                    destination, demand, other_demand,
                                )

        if best_action is None:
            break

        if best_action[0] == "move":
            _, model, target, demand = best_action
            placement[source].remove(model)
            placement[target].append(model)
            remaining[source] += model.model_size
            remaining[target] -= model.model_size
            pressure[source] -= demand
            pressure[target] += demand
        elif best_action[0] == "swap":
            _, model, other, target, demand, other_demand = best_action
            placement[source].remove(model)
            placement[target].remove(other)
            placement[source].append(other)
            placement[target].append(model)
            remaining[source] += model.model_size - other.model_size
            remaining[target] += other.model_size - model.model_size
            pressure[source] += other_demand - demand
            pressure[target] += demand - other_demand
        else:
            _, model, target, other, destination, demand, other_demand = (
                best_action
            )
            placement[source].remove(model)
            placement[target].remove(other)
            placement[target].append(model)
            placement[destination].append(other)
            remaining[source] += model.model_size
            remaining[target] += other.model_size - model.model_size
            remaining[destination] -= other.model_size
            pressure[source] -= demand
            pressure[target] += demand - other_demand
            pressure[destination] += other_demand

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