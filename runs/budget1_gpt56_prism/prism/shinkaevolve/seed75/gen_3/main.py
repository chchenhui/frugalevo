GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # A placement decision must use the KVPR after adding the model, rather
    # than the GPU's current KVPR.  Otherwise large models are repeatedly
    # assigned to apparently empty GPUs without accounting for their memory use.
    sorted_models = sorted(
        models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True
    )

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
    loads = [0.0 for _ in range(gpu_num)]

    def pressure(load, memory):
        if memory > 0:
            return load / memory
        return 0.0 if load == 0 else float("inf")

    def candidate_max(updates):
        update_map = {gpu_id: (load, memory) for gpu_id, load, memory in updates}
        return max(
            pressure(*update_map[gpu_id])
            if gpu_id in update_map
            else pressure(loads[gpu_id], remaining[gpu_id])
            for gpu_id in range(gpu_num)
        )

    # Build an initial placement by minimizing the resulting global maximum.
    for model in sorted_models:
        weight = model.req_rate / model.slo
        best_gpu = None
        best_max = float("inf")
        best_remaining = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size <= remaining[gpu_id]:
                new_remaining = remaining[gpu_id] - model.model_size
                new_load = loads[gpu_id] + weight
                new_max = candidate_max(
                    [(gpu_id, new_load, new_remaining)]
                )

                # On an objective tie, pack more tightly and preserve a large
                # contiguous memory region for later models.
                if (new_max < best_max or
                        (new_max == best_max and new_remaining < best_remaining)):
                    best_gpu = gpu_id
                    best_max = new_max
                    best_remaining = new_remaining

        if best_gpu is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        placement[best_gpu].append(model)
        loads[best_gpu] += weight
        remaining[best_gpu] -= model.model_size

    # Improve the greedy result with moves and swaps involving a bottleneck GPU.
    # Only strictly improving changes are accepted, so this always terminates.
    for _ in range(len(models) * max(gpu_num, 1)):
        current_max = max(
            pressure(loads[gpu_id], remaining[gpu_id])
            for gpu_id in range(gpu_num)
        )
        best_change = None
        best_max = current_max

        for source in range(gpu_num):
            if pressure(loads[source], remaining[source]) < current_max - 1e-12:
                continue

            for source_index, model in enumerate(placement[source]):
                weight = model.req_rate / model.slo
                size = model.model_size

                # Try moving a bottleneck model to another GPU.
                for target in range(gpu_num):
                    if target == source or size > remaining[target]:
                        continue
                    new_max = candidate_max([
                        (source, loads[source] - weight, remaining[source] + size),
                        (target, loads[target] + weight, remaining[target] - size),
                    ])
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        best_change = ("move", source, source_index, target)

                # Try swapping it with a model on another GPU.
                for target in range(gpu_num):
                    if target == source:
                        continue
                    for target_index, other in enumerate(placement[target]):
                        other_weight = other.req_rate / other.slo
                        other_size = other.model_size
                        source_memory = remaining[source] + size - other_size
                        target_memory = remaining[target] + other_size - size
                        if source_memory < 0 or target_memory < 0:
                            continue

                        new_max = candidate_max([
                            (source, loads[source] - weight + other_weight,
                             source_memory),
                            (target, loads[target] - other_weight + weight,
                             target_memory),
                        ])
                        if new_max < best_max - 1e-12:
                            best_max = new_max
                            best_change = (
                                "swap", source, source_index, target, target_index
                            )

        if best_change is None:
            break

        if best_change[0] == "move":
            _, source, source_index, target = best_change
            model = placement[source].pop(source_index)
            weight = model.req_rate / model.slo
            placement[target].append(model)
            loads[source] -= weight
            loads[target] += weight
            remaining[source] += model.model_size
            remaining[target] -= model.model_size
        else:
            _, source, source_index, target, target_index = best_change
            model = placement[source][source_index]
            other = placement[target][target_index]
            placement[source][source_index] = other
            placement[target][target_index] = model
            loads[source] += other.req_rate / other.slo - model.req_rate / model.slo
            loads[target] += model.req_rate / model.slo - other.req_rate / other.slo
            remaining[source] += model.model_size - other.model_size
            remaining[target] += other.model_size - model.model_size

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