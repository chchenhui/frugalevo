GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Compute a feasible placement with minimum possible maximum KVPR."""
    sorted_models = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo, m.model_size),
        reverse=True,
    )

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
    load = [0.0 for _ in range(gpu_num)]

    def ratio(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    # Choose the GPU with the lowest KVPR after, not before, the assignment.
    for model in sorted_models:
        weight = model.req_rate / model.slo
        best_gpu = None
        best_key = None
        for gpu_id in range(gpu_num):
            if model.model_size <= remaining[gpu_id]:
                new_memory = remaining[gpu_id] - model.model_size
                key = (ratio(load[gpu_id] + weight, new_memory), -new_memory)
                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = gpu_id

        if best_gpu is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        placement[best_gpu].append(model)
        load[best_gpu] += weight
        remaining[best_gpu] -= model.model_size

    # Strictly improve the bottleneck, with a few bounded plateau escapes.
    plateau_moves_left = 2
    allow_plateau = False
    while True:
        current = [ratio(load[gpu_id], remaining[gpu_id]) for gpu_id in range(gpu_num)]
        current_max = max(current)
        bottlenecks = [
            gpu_id for gpu_id in range(gpu_num)
            if abs(current[gpu_id] - current_max) <= 1e-12
        ]
        best_action = None
        best_key = None

        for src in bottlenecks:
            for model in placement[src]:
                weight = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if dst == src or model.model_size > remaining[dst]:
                        continue
                    candidate = list(current)
                    candidate[src] = ratio(
                        load[src] - weight, remaining[src] + model.model_size
                    )
                    candidate[dst] = ratio(
                        load[dst] + weight, remaining[dst] - model.model_size
                    )
                    candidate_max = max(candidate)
                    strict = candidate_max < current_max - 1e-12
                    plateau = (
                        allow_plateau and
                        candidate_max <= current_max + 1e-12 and
                        candidate[src] < current[src] - 1e-12
                    )
                    if strict or plateau:
                        key = (candidate_max, candidate[src])
                        if best_key is None or key < best_key:
                            best_key = key
                            best_action = ("move", src, dst, model, plateau)

                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    for other in placement[dst]:
                        if (remaining[src] + model.model_size - other.model_size < 0 or
                                remaining[dst] + other.model_size - model.model_size < 0):
                            continue
                        other_weight = other.req_rate / other.slo
                        candidate = list(current)
                        candidate[src] = ratio(
                            load[src] - weight + other_weight,
                            remaining[src] + model.model_size - other.model_size,
                        )
                        candidate[dst] = ratio(
                            load[dst] - other_weight + weight,
                            remaining[dst] + other.model_size - model.model_size,
                        )
                        candidate_max = max(candidate)
                        strict = candidate_max < current_max - 1e-12
                        plateau = (
                            allow_plateau and
                            candidate_max <= current_max + 1e-12 and
                            candidate[src] < current[src] - 1e-12
                        )
                        if strict or plateau:
                            key = (candidate_max, candidate[src])
                            if best_key is None or key < best_key:
                                best_key = key
                                best_action = ("swap", src, dst, model, other, plateau)

        if best_action is None:
            if not allow_plateau and plateau_moves_left:
                allow_plateau = True
                continue
            break

        was_plateau = best_action[-1]
        if was_plateau:
            plateau_moves_left -= 1
        allow_plateau = False

        if best_action[0] == "move":
            _, src, dst, model, _ = best_action
            weight = model.req_rate / model.slo
            placement[src].remove(model)
            placement[dst].append(model)
            load[src] -= weight
            load[dst] += weight
            remaining[src] += model.model_size
            remaining[dst] -= model.model_size
        else:
            _, src, dst, model, other, _ = best_action
            weight = model.req_rate / model.slo
            other_weight = other.req_rate / other.slo
            placement[src].remove(model)
            placement[dst].remove(other)
            placement[src].append(other)
            placement[dst].append(model)
            load[src] += other_weight - weight
            load[dst] += weight - other_weight
            remaining[src] += model.model_size - other.model_size
            remaining[dst] += other.model_size - model.model_size

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