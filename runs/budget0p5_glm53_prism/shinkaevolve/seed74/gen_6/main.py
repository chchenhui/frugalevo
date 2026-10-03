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

    # Greedy KVPR-minimizing placement based on Algorithm 1 (without τ check)
    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # 4) Local refinement: move models away from the highest-KVPR GPU
    #    whenever it lowers the maximum KVPR across all GPUs.
    def kvpr(gpu_id):
        mem = GPU_MEM_SIZE - shared_kv[gpu_id]
        if mem <= 0:
            return float('inf')
        return weighted_req_rate[gpu_id] / mem

    improved = True
    while improved:
        improved = False
        max_gpu = max(range(gpu_num), key=lambda g: kvpr(g))
        max_kv = kvpr(max_gpu)
        for m_idx, model in enumerate(placement[max_gpu]):
            for g in range(gpu_num):
                if g == max_gpu:
                    continue
                if model.model_size > shared_kv[g]:
                    continue
                # simulate the move
                old_other = kvpr(g)
                new_max_gpu = (weighted_req_rate[max_gpu] - model.req_rate / model.slo) / (GPU_MEM_SIZE - shared_kv[max_gpu] + model.model_size)
                new_other = (weighted_req_rate[g] + model.req_rate / model.slo) / (GPU_MEM_SIZE - shared_kv[g] - model.model_size)
                if max(new_max_gpu, new_other) < max_kv:
                    # apply the move
                    placement[max_gpu].pop(m_idx)
                    placement[g].append(model)
                    weighted_req_rate[max_gpu] -= model.req_rate / model.slo
                    shared_kv[max_gpu] += model.model_size
                    weighted_req_rate[g] += model.req_rate / model.slo
                    shared_kv[g] -= model.model_size
                    improved = True
                    break
            if improved:
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