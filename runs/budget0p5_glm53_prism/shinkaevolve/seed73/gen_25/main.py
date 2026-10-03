GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

import copy

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    def greedy(order):
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]
        for model in order:
            best_idx = None
            best_ratio = float('inf')
            for gpu_id in range(gpu_num):
                remaining = shared_kv[gpu_id] - model.model_size
                if remaining > 0:
                    resulting_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining
                    if resulting_ratio < best_ratio:
                        best_ratio = resulting_ratio
                        best_idx = gpu_id
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size
        return placement

    def gpu_kvpr(placement, gpu_id):
        remaining = GPU_MEM_SIZE - sum(m.model_size for m in placement[gpu_id])
        if remaining <= 0:
            return float('inf')
        return sum(m.req_rate / m.slo for m in placement[gpu_id]) / remaining

    def refine(placement):
        for _ in range(100):
            kvprs = [gpu_kvpr(placement, g) for g in range(gpu_num)]
            src = max(range(gpu_num), key=lambda g: kvprs[g])
            best_max = kvprs[src]
            improved = False
            for m in placement[src]:
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    dst_remaining = GPU_MEM_SIZE - sum(x.model_size for x in placement[dst])
                    if dst_remaining - m.model_size <= 0:
                        continue
                    placement[src].remove(m)
                    placement[dst].append(m)
                    new_max = max(gpu_kvpr(placement, g) for g in range(gpu_num))
                    if new_max < best_max - 1e-12:
                        best_max = new_max
                        improved = True
                        break
                    placement[dst].remove(m)
                    placement[src].append(m)
                if improved:
                    break
            if not improved:
                break
        return placement

    # Candidate orderings: by r_j/s_j desc, and by model size desc
    orders = [
        sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
    ]

    best_placement = None
    best_score = float('inf')
    for order in orders:
        placement = greedy(order)
        if placement is None:
            continue
        placement = refine(placement)
        score = max(gpu_kvpr(placement, g) for g in range(gpu_num))
        if score < best_score:
            best_score = score
            best_placement = placement

    if best_placement is None:
        raise ValueError(
            "Unable to place models within GPU memory."
        )
    return best_placement

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes current KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                # KVPR of this GPU after placing the model on it
                resulting_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining
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