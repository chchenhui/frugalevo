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

    # 3) Assign each model to the GPU that minimizes post-placement KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')
        w = model.req_rate / model.slo

        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                used_after = GPU_MEM_SIZE - (shared_kv[gpu_id] - model.model_size)
                post_kvpr = (weighted_req_rate[gpu_id] + w) / used_after
                if post_kvpr < best_ratio:
                    best_ratio = post_kvpr
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

    # 4) Local search: move models off the hottest GPU to reduce max KVPR
    def kvpr(g):
        denom = GPU_MEM_SIZE - shared_kv[g]
        return weighted_req_rate[g] / denom if denom > 0 else 0.0

    improved = True
    iterations = 0
    while improved and iterations < 30:
        improved = False
        iterations += 1
        kvprs = [kvpr(g) for g in range(gpu_num)]
        hot = max(range(gpu_num), key=lambda g: kvprs[g])
        hot_kvpr = kvprs[hot]
        for model in list(placement[hot]):
            w = model.req_rate / model.slo
            rest = [kvprs[g] for g in range(gpu_num) if g != hot]
            base = max(rest) if rest else 0.0
            best_gain, best_dst = 0.0, None
            for g in range(gpu_num):
                if g == hot or model.model_size > shared_kv[g]:
                    continue
                dst_used = GPU_MEM_SIZE - (shared_kv[g] - model.model_size)
                dst_kvpr = (weighted_req_rate[g] + w) / dst_used
                src_used = GPU_MEM_SIZE - (shared_kv[hot] + model.model_size)
                src_kvpr = (weighted_req_rate[hot] - w) / src_used if src_used > 0 else 0.0
                new_max = max(base, dst_kvpr, src_kvpr)
                if hot_kvpr - new_max > best_gain:
                    best_gain = hot_kvpr - new_max
                    best_dst = g
            if best_dst is not None:
                placement[hot].remove(model)
                placement[best_dst].append(model)
                weighted_req_rate[hot] -= w
                shared_kv[hot] += model.model_size
                weighted_req_rate[best_dst] += w
                shared_kv[best_dst] -= model.model_size
                improved = True
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