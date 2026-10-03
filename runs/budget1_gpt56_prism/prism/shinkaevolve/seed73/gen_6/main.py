GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a memory-feasible placement that minimizes the maximum KVPR.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    def demand(model):
        return model.req_rate / model.slo

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_gpu = None
            best_max_kvpr = float("inf")

            for gpu_id in range(gpu_num):
                if model.model_size > remaining[gpu_id]:
                    continue

                new_remaining = remaining[gpu_id] - model.model_size
                new_load = load[gpu_id] + demand(model)
                projected = (
                    new_load / new_remaining
                    if new_remaining > 0
                    else float("inf")
                )
                max_kvpr = projected
                for other_gpu in range(gpu_num):
                    if other_gpu != gpu_id and remaining[other_gpu] > 0:
                        max_kvpr = max(
                            max_kvpr, load[other_gpu] / remaining[other_gpu]
                        )

                if max_kvpr < best_max_kvpr:
                    best_max_kvpr = max_kvpr
                    best_gpu = gpu_id

            if best_gpu is None:
                return None

            placement[best_gpu].append(model)
            remaining[best_gpu] -= model.model_size
            load[best_gpu] += demand(model)

        score = max(
            (load[gpu_id] / remaining[gpu_id])
            if remaining[gpu_id] > 0
            else float("inf")
            for gpu_id in range(gpu_num)
        )
        return score, placement

    orders = [
        sorted(models, key=demand, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: demand(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (demand(m), m.model_size), reverse=True),
    ]

    best = None
    for order in orders:
        candidate = greedy(order)
        if candidate is not None and (best is None or candidate[0] < best[0]):
            best = candidate

    if best is None:
        raise ValueError(
            f"Unable to place all models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB"
        )

    return best[1]

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