GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily place highest standalone-KVPR models on the GPU with lowest projected KVPR."""
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def standalone_kvpr(model):
        return (model.req_rate / model.slo) / max(GPU_MEM_SIZE - model.model_size, 1e-12)

    for model in sorted(models, key=lambda m: (standalone_kvpr(m), m.model_size), reverse=True):
        demand = model.req_rate / model.slo
        candidates = [
            ((load[gpu] + demand) / max(remaining[gpu] - model.model_size, 1e-12), gpu)
            for gpu in range(gpu_num)
            if model.model_size <= remaining[gpu]
        ]
        if not candidates:
            raise ValueError(f"Unable to place model of size {model.model_size} GB.")

        _, gpu = min(candidates)
        placement[gpu].append(model)
        load[gpu] += demand
        remaining[gpu] -= model.model_size

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
