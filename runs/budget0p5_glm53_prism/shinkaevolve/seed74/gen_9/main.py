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

    # Binary-search-on-threshold placement: find the smallest lambda such that
    # all models can be packed with every GPU's KVPR <= lambda.
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    def try_pack(lam):
        """Pack models into at most gpu_num GPUs with KVPR <= lam.
        Returns assignment dict, or None if infeasible."""
        rem = [GPU_MEM_SIZE] * gpu_num
        W = [0.0] * gpu_num
        assign = {i: [] for i in range(gpu_num)}
        used = 0  # number of GPUs opened so far
        for m in sorted_models:
            w = m.req_rate / m.slo
            s = m.model_size
            best_i, best_val = None, None
            # Best-fit: among GPUs where model fits and KVPR stays <= lam,
            # pick the one with the smallest resulting KVPR (tightest fit).
            for i in range(used):
                if s <= rem[i]:
                    val = (W[i] + w) / (rem[i] - s)
                    if val <= lam + 1e-12 and (best_val is None or val < best_val):
                        best_val, best_i = val, i
            # Allow opening a fresh GPU if it satisfies the constraint
            if best_i is None and used < gpu_num and s <= GPU_MEM_SIZE:
                val = w / (GPU_MEM_SIZE - s) if (GPU_MEM_SIZE - s) > 0 else float('inf')
                if val <= lam + 1e-12:
                    best_i, best_val = used, val
            if best_i is None:
                return None
            rem[best_i] -= s
            W[best_i] += w
            assign[best_i].append(m)
            if best_i == used:
                used += 1
        return assign

    # Binary search the smallest feasible lambda
    lo, hi = 0.0, 1e9
    best = try_pack(hi)
    if best is None:
        # Infeasible under any threshold: fall back to memory-only first-fit
        rem = [GPU_MEM_SIZE] * gpu_num
        assign = {i: [] for i in range(gpu_num)}
        for m in sorted_models:
            placed = False
            for i in range(gpu_num):
                if m.model_size <= rem[i]:
                    rem[i] -= m.model_size
                    assign[i].append(m)
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU."
                )
        return assign

    for _ in range(100):
        mid = (lo + hi) / 2
        res = try_pack(mid)
        if res is not None:
            best = res
            hi = mid
        else:
            lo = mid

    return best

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