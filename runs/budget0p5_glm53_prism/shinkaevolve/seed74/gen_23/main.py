GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement minimizing the maximum KVPR across all GPUs
    using binary-search-on-threshold best-fit packing plus local-search refinement.
    """

    def safe_ratio(W, free):
        if free <= 1e-9:
            return float('inf')
        return W / free

    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    def try_pack(lam, record=False):
        """Best-fit pack models into at most gpu_num GPUs with KVPR <= lam.
        Returns (assign, rem, W) if record, True if feasible, None if infeasible."""
        rem = [GPU_MEM_SIZE] * gpu_num
        W = [0.0] * gpu_num
        assign = {i: [] for i in range(gpu_num)}
        used = 0  # number of GPUs opened so far
        for m in sorted_models:
            w = m.req_rate / m.slo
            s = m.model_size
            best_i, best_val = None, None
            # Best-fit: among opened GPUs where the model fits and KVPR <= lam,
            # pick the one with the smallest resulting KVPR (tightest fit).
            for i in range(used):
                if s <= rem[i]:
                    val = safe_ratio(W[i] + w, rem[i] - s)
                    if val <= lam + 1e-12 and (best_val is None or val < best_val):
                        best_val, best_i = val, i
            # Allow opening a fresh GPU if it satisfies the constraint
            if best_i is None and used < gpu_num and s <= GPU_MEM_SIZE:
                val = safe_ratio(w, GPU_MEM_SIZE - s)
                if val <= lam + 1e-12:
                    best_i, best_val = used, val
            if best_i is None:
                return None
            rem[best_i] -= s
            W[best_i] += w
            assign[best_i].append(m)
            if best_i == used:
                used += 1
        if record:
            return assign, rem, W
        return True

    # Binary search the smallest feasible lambda
    lo, hi = 0.0, 1e9
    if try_pack(hi) is None:
        # Infeasible under any threshold: fall back to memory-only best-fit
        placement = {i: [] for i in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        W = [0.0] * gpu_num
        for m in sorted_models:
            best_i, best_val = None, float('inf')
            for i in range(gpu_num):
                if m.model_size <= GPU_MEM_SIZE - used_mem[i]:
                    val = safe_ratio(W[i] + m.req_rate / m.slo,
                                     GPU_MEM_SIZE - used_mem[i] - m.model_size)
                    if val < best_val:
                        best_val, best_i = val, i
            if best_i is None:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU."
                )
            placement[best_i].append(m)
            used_mem[best_i] += m.model_size
            W[best_i] += m.req_rate / m.slo
        return placement

    best = None
    for _ in range(60):
        mid = (lo + hi) / 2
        if try_pack(mid):
            best = mid
            hi = mid
        else:
            lo = mid

    result = try_pack(hi, record=True)
    if result is None:
        result = try_pack(1e9, record=True)
    if result is None:
        raise ValueError("Unable to place models.")

    placement, rem, W = result
    used_mem = [GPU_MEM_SIZE - r for r in rem]

    # --- Local search refinement: move models off the hottest GPU ---
    def kvpr(i):
        return safe_ratio(W[i], GPU_MEM_SIZE - used_mem[i])

    improved = True
    guard = 0
    while improved and guard < 5000:
        improved = False
        guard += 1
        max_i = max(range(gpu_num), key=kvpr)
        max_val = kvpr(max_i)
        if max_val == float('inf'):
            break
        for m in list(placement[max_i]):
            w = m.req_rate / m.slo
            s = m.model_size
            for j in range(gpu_num):
                if j == max_i:
                    continue
                if s <= GPU_MEM_SIZE - used_mem[j]:
                    n1 = safe_ratio(W[max_i] - w, GPU_MEM_SIZE - used_mem[max_i] + s)
                    n2 = safe_ratio(W[j] + w, GPU_MEM_SIZE - used_mem[j] - s)
                    if gpu_num > 2:
                        others = max(kvpr(k) for k in range(gpu_num)
                                     if k not in (max_i, j))
                        nmax = max(n1, n2, others)
                    else:
                        nmax = max(n1, n2)
                    if nmax < max_val - 1e-12:
                        placement[max_i].remove(m)
                        placement[j].append(m)
                        W[max_i] -= w
                        used_mem[max_i] -= s
                        W[j] += w
                        used_mem[j] += s
                        improved = True
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
