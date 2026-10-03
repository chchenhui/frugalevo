GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _kvpr(load, free_mem):
    """KVPR given accumulated load (sum of r/s) and free memory."""
    denom = GPU_MEM_SIZE - free_mem
    if denom <= 0:
        return float('inf')
    return load / denom


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # --- Stage 1: greedy seed placement by resulting (post-placement) KVPR ---
    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    placement = {g: [] for g in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num  # sum of r_j / s_j

    for model in sorted_models:
        best_gpu, best_kvpr = None, float('inf')
        for g in range(gpu_num):
            if model.model_size > free_mem[g]:
                continue
            # KVPR if we placed model here
            new_kvpr = (load[g] + model.req_rate / model.slo) / (GPU_MEM_SIZE - (free_mem[g] - model.model_size))
            if new_kvpr < best_kvpr:
                best_kvpr = new_kvpr
                best_gpu = g
        if best_gpu is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU."
            )
        placement[best_gpu].append(model)
        load[best_gpu] += model.req_rate / model.slo
        free_mem[best_gpu] -= model.model_size

    # --- Stage 2: local search — move models to reduce max KVPR ---
    def gpu_kvprs():
        return [_kvpr(load[g], free_mem[g]) for g in range(gpu_num)]

    improved = True
    iterations = 0
    while improved and iterations < 20:
        improved = False
        iterations += 1
        kvprs = gpu_kvprs()
        hot = max(range(gpu_num), key=lambda g: kvprs[g])
        hot_kvpr = kvprs[hot]
        # try moving each model off the hottest GPU
        for model in list(placement[hot]):
            if len(placement[hot]) == 1 and all(len(placement[g]) == 0 for g in range(gpu_num) if g != hot):
                # only models exist on this GPU; can still move if others fit
                pass
            src_w = model.req_rate / model.slo
            src_kvpr_after = (load[hot] - src_w) / (GPU_MEM_SIZE - (free_mem[hot] + model.model_size)) if (free_mem[hot] + model.model_size) < GPU_MEM_SIZE else 0.0
            best_gain, best_dst = 0.0, None
            for g in range(gpu_num):
                if g == hot or model.model_size > free_mem[g]:
                    continue
                dst_kvpr_after = (load[g] + src_w) / (GPU_MEM_SIZE - (free_mem[g] - model.model_size))
                # moving helps if new max KVPR is lower than current hot KVPR
                new_max = max(max(kvprs[:g] + kvprs[g+1:g] + kvprs[g+1:]), dst_kvpr_after, src_kvpr_after)
                gain = hot_kvpr - new_max
                if gain > best_gain:
                    best_gain = gain
                    best_dst = g
            if best_dst is not None:
                placement[hot].remove(model)
                placement[best_dst].append(model)
                load[hot] -= src_w
                free_mem[hot] += model.model_size
                load[best_dst] += src_w
                free_mem[best_dst] -= model.model_size
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
