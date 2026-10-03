GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy placement (hardest models first, each to the GPU yielding the
    lowest resulting KVPR), followed by a local-search refinement that
    repeatedly moves a model off the current worst-KVPR GPU when doing so
    lowers the maximum KVPR.
    """

    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]

    def kvpr(gpu_id):
        if shared_kv[gpu_id] >= GPU_MEM_SIZE:
            return 0.0
        return weighted_req_rate[gpu_id] / shared_kv[gpu_id]

    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                new_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining
                if new_ratio < best_ratio:
                    best_ratio = new_ratio
                    best_idx = gpu_id

        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # Local search: move models away from the worst GPU to reduce max KVPR
    for _ in range(len(models)):
        worst = max(range(gpu_num), key=kvpr)
        if not placement[worst] or kvpr(worst) == 0.0:
            break
        cur_max = max(kvpr(g) for g in range(gpu_num))
        moved = False
        # try moving the model with the largest pressure contribution first
        for model in sorted(placement[worst],
                            key=lambda m: m.req_rate / m.slo, reverse=True):
            for gpu_id in range(gpu_num):
                if gpu_id == worst:
                    continue
                if shared_kv[gpu_id] - model.model_size <= 0:
                    continue
                # simulate move
                weighted_req_rate[worst] -= model.req_rate / model.slo
                shared_kv[worst] += model.model_size
                weighted_req_rate[gpu_id] += model.req_rate / model.slo
                shared_kv[gpu_id] -= model.model_size
                new_max = max(kvpr(g) for g in range(gpu_num))
                if new_max < cur_max:
                    placement[worst].remove(model)
                    placement[gpu_id].append(model)
                    moved = True
                    break
                # revert
                weighted_req_rate[worst] += model.req_rate / model.slo
                shared_kv[worst] -= model.model_size
                weighted_req_rate[gpu_id] -= model.req_rate / model.slo
                shared_kv[gpu_id] += model.model_size
            if moved:
                break
        if not moved:
            break

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
