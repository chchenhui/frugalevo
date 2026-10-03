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

        req_per_slo = model.req_rate / model.slo
        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                # KVPR of this GPU after placing the model
                new_ratio = (weighted_req_rate[gpu_id] + req_per_slo) / remaining
                if new_ratio < best_ratio:
                    best_ratio = new_ratio
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

    # 4) Local search: move models away from the highest-KVPR GPU when it helps
    def kvpr(gpu_id):
        return weighted_req_rate[gpu_id] / shared_kv[gpu_id]

    improved = True
    while improved:
        improved = False
        worst = max(range(gpu_num), key=kvpr)
        worst_kvpr = kvpr(worst)
        # second highest KVPR (that would become the max after reducing `worst`)
        other_kvprs = [kvpr(g) for g in range(gpu_num) if g != worst]
        second_kvpr = max(other_kvprs) if other_kvprs else 0.0

        best_move = None
        best_new_max = worst_kvpr
        for model in placement[worst]:
            req_per_slo = model.req_rate / model.slo
            src_after = (weighted_req_rate[worst] - req_per_slo) / (shared_kv[worst] + model.model_size)
            for dst in range(gpu_num):
                if dst == worst:
                    continue
                if model.model_size > shared_kv[dst]:
                    continue
                dst_after = (weighted_req_rate[dst] + req_per_slo) / (shared_kv[dst] - model.model_size)
                new_max = max(second_kvpr, src_after, dst_after)
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_move = (model, dst)

        if best_move is not None:
            model, dst = best_move
            placement[worst].remove(model)
            placement[dst].append(model)
            weighted_req_rate[worst] -= model.req_rate / model.slo
            shared_kv[worst] += model.model_size
            weighted_req_rate[dst] += model.req_rate / model.slo
            shared_kv[dst] -= model.model_size
            improved = True

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