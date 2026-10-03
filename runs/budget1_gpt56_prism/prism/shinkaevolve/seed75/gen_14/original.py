GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Compute a memory-feasible placement minimizing maximum KVPR."""

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    pressure = [0.0 for _ in range(gpu_num)]

    def kvpr(load, free_memory):
        if free_memory <= 0:
            return float("inf") if load > 0 else 0.0
        return load / free_memory

    def maximum_kvpr():
        return max(kvpr(pressure[gpu_id], remaining[gpu_id])
                   for gpu_id in range(gpu_num))

    # Place high-pressure models first so their large impact is balanced early.
    sorted_models = sorted(
        models,
        key=lambda model: (model.req_rate / model.slo, model.model_size),
        reverse=True,
    )

    for model in sorted_models:
        model_pressure = model.req_rate / model.slo
        best_gpu = None
        best_score = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size > remaining[gpu_id]:
                continue

            candidate_score = 0.0
            for other_gpu in range(gpu_num):
                if other_gpu == gpu_id:
                    score = kvpr(
                        pressure[other_gpu] + model_pressure,
                        remaining[other_gpu] - model.model_size,
                    )
                else:
                    score = kvpr(pressure[other_gpu], remaining[other_gpu])
                candidate_score = max(candidate_score, score)

            if candidate_score < best_score:
                best_score = candidate_score
                best_gpu = gpu_id

        if best_gpu is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        placement[best_gpu].append(model)
        remaining[best_gpu] -= model.model_size
        pressure[best_gpu] += model_pressure

    # Greedy placement can leave an avoidable hotspot.  Apply improving moves
    # until no single memory-feasible relocation lowers the global maximum.
    while True:
        current_score = maximum_kvpr()
        best_move = None
        best_score = current_score

        for source_gpu in range(gpu_num):
            for model in placement[source_gpu]:
                model_pressure = model.req_rate / model.slo
                for target_gpu in range(gpu_num):
                    if target_gpu == source_gpu:
                        continue
                    if model.model_size > remaining[target_gpu]:
                        continue

                    candidate_score = 0.0
                    for gpu_id in range(gpu_num):
                        if gpu_id == source_gpu:
                            score = kvpr(
                                pressure[gpu_id] - model_pressure,
                                remaining[gpu_id] + model.model_size,
                            )
                        elif gpu_id == target_gpu:
                            score = kvpr(
                                pressure[gpu_id] + model_pressure,
                                remaining[gpu_id] - model.model_size,
                            )
                        else:
                            score = kvpr(pressure[gpu_id], remaining[gpu_id])
                        candidate_score = max(candidate_score, score)

                    if candidate_score < best_score - 1e-12:
                        best_score = candidate_score
                        best_move = (model, source_gpu, target_gpu, model_pressure)

        if best_move is None:
            break

        model, source_gpu, target_gpu, model_pressure = best_move
        placement[source_gpu].remove(model)
        placement[target_gpu].append(model)
        remaining[source_gpu] += model.model_size
        remaining[target_gpu] -= model.model_size
        pressure[source_gpu] -= model_pressure
        pressure[target_gpu] += model_pressure

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