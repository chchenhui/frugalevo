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

        model_weight = model.req_rate / model.slo
        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                # KVPR of this GPU after placing the model
                resulting_ratio = (weighted_req_rate[gpu_id] + model_weight) / remaining
                if resulting_ratio < best_ratio:
                    best_ratio = resulting_ratio
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

    # 4) Local search: move models from the highest-KVPR GPU to reduce max KVPR
    def kvpr(gpu_id):
        if shared_kv[gpu_id] <= 0:
            return float('inf')
        return weighted_req_rate[gpu_id] / shared_kv[gpu_id]

    improved = True
    while improved:
        improved = False
        worst = max(range(gpu_num), key=kvpr)
        worst_kvpr = kvpr(worst)
        # order models on worst GPU by weight descending
        candidates = sorted(placement[worst], key=lambda m: (m.req_rate / m.slo), reverse=True)
        for model in candidates:
            w = model.req_rate / model.slo
            for gpu_id in range(gpu_num):
                if gpu_id == worst:
                    continue
                if model.model_size > shared_kv[gpu_id]:
                    continue
                # KVPR of source after removal
                src_rem = shared_kv[worst] + model.model_size
                src_kvpr = (weighted_req_rate[worst] - w) / src_rem if src_rem > 0 else float('inf')
                # KVPR of destination after placement
                dst_rem = shared_kv[gpu_id] - model.model_size
                dst_kvpr = (weighted_req_rate[gpu_id] + w) / dst_rem if dst_rem > 0 else float('inf')
                new_max = max(max(kvpr(g) for g in range(gpu_num) if g not in (worst, gpu_id)),
                              src_kvpr, dst_kvpr)
                if new_max < worst_kvpr:
                    placement[worst].remove(model)
                    placement[gpu_id].append(model)
                    weighted_req_rate[worst] -= w
                    shared_kv[worst] += model.model_size
                    weighted_req_rate[gpu_id] += w
                    shared_kv[gpu_id] -= model.model_size
                    improved = True
                    break
            if improved:
                break

        # 5) Swap phase: exchange a model on the worst GPU with one on another GPU
        if not improved:
            for model in list(placement[worst]):
                w_a = model.req_rate / model.slo
                for gpu_id in range(gpu_num):
                    if gpu_id == worst:
                        continue
                    for other in list(placement[gpu_id]):
                        w_b = other.req_rate / other.slo
                        # memory feasibility both directions
                        if model.model_size - other.model_size > shared_kv[gpu_id]:
                            continue
                        if other.model_size - model.model_size > shared_kv[worst]:
                            continue
                        src_rem = shared_kv[worst] + model.model_size - other.model_size
                        dst_rem = shared_kv[gpu_id] + other.model_size - model.model_size
                        if src_rem <= 0 or dst_rem <= 0:
                            continue
                        src_kvpr = (weighted_req_rate[worst] - w_a + w_b) / src_rem
                        dst_kvpr = (weighted_req_rate[gpu_id] - w_b + w_a) / dst_rem
                        new_max = max(
                            max(kvpr(g) for g in range(gpu_num) if g not in (worst, gpu_id)),
                            src_kvpr, dst_kvpr)
                        if new_max < worst_kvpr - 1e-12:
                            placement[worst].remove(model)
                            placement[worst].append(other)
                            placement[gpu_id].remove(other)
                            placement[gpu_id].append(model)
                            weighted_req_rate[worst] += w_b - w_a
                            shared_kv[worst] += model.model_size - other.model_size
                            weighted_req_rate[gpu_id] += w_a - w_b
                            shared_kv[gpu_id] += other.model_size - model.model_size
                            improved = True
                            break
                    if improved:
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