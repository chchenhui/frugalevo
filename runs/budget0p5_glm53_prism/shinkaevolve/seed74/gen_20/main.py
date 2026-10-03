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

    # 4) Local-search refinement: move models off the hottest GPU when it lowers max KVPR
    def gpu_kvpr(gpu_id):
        den = GPU_MEM_SIZE - sum(m.model_size for m in placement[gpu_id])
        if den <= 0:
            return float('inf')
        return sum(m.req_rate / m.slo for m in placement[gpu_id]) / den

    for _ in range(200):
        kvs = [gpu_kvpr(g) for g in range(gpu_num)]
        cur_max = max(kvs)
        src = max(range(gpu_num), key=lambda g: kvs[g])
        improved = False
        for model in placement[src]:
            w = model.req_rate / model.slo
            for dst in range(gpu_num):
                if dst == src or model.model_size > shared_kv[dst]:
                    continue
                src_den = shared_kv[src] + model.model_size
                dst_den = shared_kv[dst] - model.model_size
                if src_den <= 0 or dst_den <= 0:
                    continue
                new_src = (weighted_req_rate[src] - w) / src_den
                new_dst = (weighted_req_rate[dst] + w) / dst_den
                others = max([kvs[i] for i in range(gpu_num) if i not in (src, dst)] or [0.0])
                if max(new_src, new_dst, others) < cur_max - 1e-12:
                    placement[src].remove(model)
                    placement[dst].append(model)
                    shared_kv[src] += model.model_size
                    shared_kv[dst] -= model.model_size
                    weighted_req_rate[src] -= w
                    weighted_req_rate[dst] += w
                    improved = True
                    break
            if improved:
                break
        if not improved:
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