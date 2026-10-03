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
            if model.model_size <= shared_kv[gpu_id]:
                remaining = shared_kv[gpu_id] - model.model_size
                prospective_ratio = (
                    (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining
                    if remaining > 0 else float('inf')
                )
                if prospective_ratio < best_ratio:
                    best_ratio = prospective_ratio
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

    # Improve the greedy solution with capacity-safe local exchanges.  Comparing
    # the full sorted pressure vector avoids trading one bottleneck for another.
    def pressure(load, remaining):
        return load / remaining if remaining > 0 else float('inf')

    def pressure_vector(loads, remaining):
        return tuple(sorted(pressure(loads[i], remaining[i]) for i in range(gpu_num),
                            reverse=True))

    for _ in range(100):
        base = pressure_vector(weighted_req_rate, shared_kv)
        bottlenecks = [
            i for i in range(gpu_num)
            if pressure(weighted_req_rate[i], shared_kv[i]) == base[0]
        ]
        best = None

        def consider(changes):
            nonlocal best
            loads = weighted_req_rate[:]
            remaining = shared_kv[:]
            for gpu, removed, added in changes:
                remaining[gpu] += sum(m.model_size for m in removed)
                remaining[gpu] -= sum(m.model_size for m in added)
                loads[gpu] -= sum(m.req_rate / m.slo for m in removed)
                loads[gpu] += sum(m.req_rate / m.slo for m in added)
            if min(remaining) < 0:
                return
            candidate = pressure_vector(loads, remaining)
            if candidate < base and (best is None or candidate < best[0]):
                best = (candidate, changes)

        for source in bottlenecks:
            for target in range(gpu_num):
                if source == target:
                    continue
                # Single-model moves and one-for-one swaps.
                for model in placement[source]:
                    consider([(source, (model,), ()), (target, (), (model,))])
                    for other in placement[target]:
                        consider([(source, (model,), (other,)),
                                  (target, (other,), (model,))])

                    # One bottleneck model exchanged for two target models.
                    for first_index, first in enumerate(placement[target]):
                        for second in placement[target][first_index + 1:]:
                            consider([(source, (model,), (first, second)),
                                      (target, (first, second), (model,))])

                # Reverse two-for-one exchange: two bottleneck models are
                # replaced by one target model.
                for first_index, first in enumerate(placement[source]):
                    for second in placement[source][first_index + 1:]:
                        for model in placement[target]:
                            consider([(source, (first, second), (model,)),
                                      (target, (model,), (first, second))])

        if best is None:
            break

        for gpu, removed, added in best[1]:
            for model in removed:
                placement[gpu].remove(model)
                shared_kv[gpu] += model.model_size
                weighted_req_rate[gpu] -= model.req_rate / model.slo
            for model in added:
                placement[gpu].append(model)
                shared_kv[gpu] -= model.model_size
                weighted_req_rate[gpu] += model.req_rate / model.slo

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