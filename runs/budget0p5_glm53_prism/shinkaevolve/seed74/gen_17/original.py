GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Greedy placement: assign models (sorted by req_rate/slo descending) to the
    GPU that minimizes the resulting KVPR, skipping GPUs with no KV room left.
    """
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    rem_mem = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')
        model_weight = model.req_rate / model.slo

        for gpu_id in range(gpu_num):
            free_after = rem_mem[gpu_id] - model.model_size
            # Skip if model doesn't fit or leaves no room for KV cache
            if free_after <= 0:
                continue
            new_ratio = (load[gpu_id] + model_weight) / free_after
            if new_ratio < best_ratio:
                best_ratio = new_ratio
                best_idx = gpu_id

        # Fallback: allow filling a GPU exactly if nothing else works
        if best_idx is None:
            for gpu_id in range(gpu_num):
                if model.model_size <= rem_mem[gpu_id]:
                    best_idx = gpu_id
                    break

        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU."
            )

        placement[best_idx].append(model)
        load[best_idx] += model_weight
        rem_mem[best_idx] -= model.model_size

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