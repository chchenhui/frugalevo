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

    # Greedy: sort models by r_j/s_j descending, place each model on the GPU
    # that minimizes the KVPR *after* placement. Prefer GPUs that keep
    # strictly positive free memory to avoid zero denominators in KVPR.
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num
    w_rate = [0.0] * gpu_num  # sum of r_j / s_j per GPU

    for model in sorted_models:
        r = model.req_rate / model.slo
        best_idx, best_val = None, float('inf')
        # First pass: GPUs where the model leaves strictly positive free memory
        for g in range(gpu_num):
            if model.model_size < free_mem[g]:
                val = (w_rate[g] + r) / (free_mem[g] - model.model_size)
                if val < best_val:
                    best_val, best_idx = val, g
        # Fallback: allow exact fit if no strictly-positive option exists
        if best_idx is None:
            for g in range(gpu_num):
                if model.model_size == free_mem[g] and w_rate[g] == 0.0:
                    best_idx = g
                    break
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {free_mem}"
            )
        placement[best_idx].append(model)
        w_rate[best_idx] += r
        free_mem[best_idx] -= model.model_size

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
