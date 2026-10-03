GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy placement (hardest models first, each to the GPU yielding the
    lowest resulting KVPR), followed by a local-search refinement that
    repeatedly moves a model off the current worst-KVPR GPU when doing so
    lowers the maximum KVPR.
    """

    # Greedy placement: assign each model to the GPU where placing it
    # yields the lowest resulting KVPR, processing models by descending
    # req_rate/slo (hardest models first).
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                # KVPR of this GPU after placing the model on it
                new_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining
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

    def kvpr(g):
        # Guard against zero (or negative) remaining memory to avoid
        # division-by-zero; empty GPUs have zero pressure.
        if shared_kv[g] <= 0:
            return 0.0 if weighted_req_rate[g] == 0.0 else float('inf')
        return weighted_req_rate[g] / shared_kv[g]

    def cur_max():
        return max(kvpr(g) for g in range(gpu_num))

    # Local search: repeatedly move or swap a model involving the
    # worst-KVPR GPU when doing so lowers the maximum KVPR.
    for _ in range(2 * len(models) + 4):
        worst = max(range(gpu_num), key=kvpr)
        if not placement[worst] or kvpr(worst) == 0.0:
            break
        target = cur_max()
        improved = False

        # Try moving a model off the worst GPU.
        for model in sorted(placement[worst],
                            key=lambda m: m.req_rate / m.slo, reverse=True):
            for gpu_id in range(gpu_num):
                if gpu_id == worst or shared_kv[gpu_id] - model.model_size <= 0:
                    continue
                weighted_req_rate[worst] -= model.req_rate / model.slo
                shared_kv[worst] += model.model_size
                weighted_req_rate[gpu_id] += model.req_rate / model.slo
                shared_kv[gpu_id] -= model.model_size
                if cur_max() < target:
                    placement[worst].remove(model)
                    placement[gpu_id].append(model)
                    improved = True
                    break
                weighted_req_rate[worst] += model.req_rate / model.slo
                shared_kv[worst] -= model.model_size
                weighted_req_rate[gpu_id] -= model.req_rate / model.slo
                shared_kv[gpu_id] += model.model_size
            if improved:
                break
        if improved:
            continue

        # Try swapping a model on the worst GPU with one on another GPU.
        for m1 in sorted(placement[worst],
                         key=lambda m: m.req_rate / m.slo, reverse=True):
            for gpu_id in range(gpu_num):
                if gpu_id == worst:
                    continue
                for m2 in list(placement[gpu_id]):
                    if (shared_kv[worst] + m1.model_size - m2.model_size <= 0 or
                            shared_kv[gpu_id] + m2.model_size - m1.model_size <= 0):
                        continue
                    # simulate swap
                    weighted_req_rate[worst] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                    shared_kv[worst] += m1.model_size - m2.model_size
                    weighted_req_rate[gpu_id] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
                    shared_kv[gpu_id] += m2.model_size - m1.model_size
                    if cur_max() < target:
                        placement[worst].remove(m1)
                        placement[worst].append(m2)
                        placement[gpu_id].remove(m2)
                        placement[gpu_id].append(m1)
                        improved = True
                        break
                    # revert
                    weighted_req_rate[worst] -= m2.req_rate / m2.slo - m1.req_rate / m1.slo
                    shared_kv[worst] -= m1.model_size - m2.model_size
                    weighted_req_rate[gpu_id] -= m1.req_rate / m1.slo - m2.req_rate / m2.slo
                    shared_kv[gpu_id] -= m2.model_size - m1.model_size
                if improved:
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
