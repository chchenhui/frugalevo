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

    def kvpr_of(load, used_mem):
        denom = GPU_MEM_SIZE - used_mem
        if denom <= 0:
            return float('inf')
        return load / denom

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    def run_pipeline(order_key):
        # 1) Sort models by chosen ordering key in descending order
        sorted_models = sorted(models, key=order_key, reverse=True)

        # 2) Initialize per-GPU states
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
        load = [0.0 for _ in range(gpu_num)]

        # 3) Assign each model to the GPU minimizing resulting post-placement KVPR
        for model in sorted_models:
            w = model.req_rate / model.slo
            best_idx = None
            best_kvpr = float('inf')

            for gpu_id in range(gpu_num):
                if model.model_size <= free_mem[gpu_id]:
                    new_kvpr = kvpr_of(load[gpu_id] + w,
                                       GPU_MEM_SIZE - free_mem[gpu_id] + model.model_size)
                    if new_kvpr < best_kvpr:
                        best_kvpr = new_kvpr
                        best_idx = gpu_id

            if best_idx is None:
                return None, float('inf')

            placement[best_idx].append(model)
            load[best_idx] += w
            free_mem[best_idx] -= model.model_size

        # 4) Local search: move models off the hottest GPU to reduce max KVPR
        for _ in range(30):
            kvprs = [kvpr_of(load[g], GPU_MEM_SIZE - free_mem[g]) for g in range(gpu_num)]
            hot = max(range(gpu_num), key=lambda g: kvprs[g])
            hot_kvpr = kvprs[hot]
            others = [kvprs[g] for g in range(gpu_num) if g != hot]
            base_others = max(others) if others else 0.0

            best_gain, best_move = 0.0, None
            for model in placement[hot]:
                w = model.req_rate / model.slo
                # source GPU after removal
                new_used = GPU_MEM_SIZE - free_mem[hot] - model.model_size
                src_kvpr = kvpr_of(load[hot] - w, new_used) if new_used > 0 else 0.0
                for g in range(gpu_num):
                    if g == hot or model.model_size > free_mem[g]:
                        continue
                    dst_kvpr = kvpr_of(load[g] + w,
                                       GPU_MEM_SIZE - free_mem[g] + model.model_size)
                    new_max = max(base_others, src_kvpr, dst_kvpr)
                    gain = hot_kvpr - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (model, g)

            if best_move is None:
                break

            model, dst = best_move
            w = model.req_rate / model.slo
            placement[hot].remove(model)
            placement[dst].append(model)
            load[hot] -= w
            free_mem[hot] += model.model_size
            load[dst] += w
            free_mem[dst] -= model.model_size

        kvprs = [kvpr_of(load[g], GPU_MEM_SIZE - free_mem[g]) for g in range(gpu_num)]
        return placement, max(kvprs)

    orderings = [
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9),
        lambda m: (m.req_rate / m.slo, m.model_size),
    ]

    best_placement, best_max = None, float('inf')
    for key in orderings:
        placement, mx = run_pipeline(key)
        if placement is not None and mx < best_max:
            best_max, best_placement = mx, placement

    if best_placement is None:
        raise ValueError(
            f"Unable to place models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB each."
        )

    return best_placement

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