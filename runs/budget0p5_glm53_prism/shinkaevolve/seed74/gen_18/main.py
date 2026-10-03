GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Strategy:
      1) Greedy: place models (sorted by pressure density r/s descending) onto
         the GPU that yields the smallest resulting KVPR while fitting memory.
      2) Local search: repeatedly pick the GPU with maximum KVPR and try to
         move one of its models to another GPU if that strictly lowers the
         overall maximum KVPR. Repeat until no improvement.
    """

    import random

    def pack(order):
        """Binary-search the smallest lambda such that a best-fit packing of
        `order` into at most gpu_num GPUs keeps every GPU's KVPR <= lambda.
        Returns the assignment dict, or None if infeasible."""

        def try_pack(lam):
            rem = [GPU_MEM_SIZE] * gpu_num
            W = [0.0] * gpu_num
            assign = {i: [] for i in range(gpu_num)}
            used = 0
            for m in order:
                w = m.req_rate / m.slo
                s = m.model_size
                best_i, best_val = None, None
                for i in range(used):
                    if s <= rem[i]:
                        free = rem[i] - s
                        val = (W[i] + w) / free if free > 1e-12 else float('inf')
                        if val <= lam + 1e-12 and (best_val is None or val < best_val):
                            best_val, best_i = val, i
                if best_i is None and used < gpu_num and s <= GPU_MEM_SIZE:
                    free = GPU_MEM_SIZE - s
                    val = w / free if free > 1e-12 else float('inf')
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

        lo, hi = 0.0, 1e9
        best = try_pack(hi)
        if best is None:
            return None
        for _ in range(60):
            mid = (lo + hi) / 2
            res = try_pack(mid)
            if res is not None:
                best = res
                hi = mid
            else:
                lo = mid
        return best

    def kvpr_of(assign):
        worst = 0.0
        for i in range(gpu_num):
            free = GPU_MEM_SIZE - sum(m.model_size for m in assign[i])
            Wsum = sum(m.req_rate / m.slo for m in assign[i])
            if free <= 1e-9:
                return float('inf')
            worst = max(worst, Wsum / free)
        return worst

    base = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    best_assign = None
    best_score = float('inf')

    # Randomized Las Vegas restarts with perturbed sort keys
    for restart in range(25):
        if restart == 0:
            order = base
        else:
            order = sorted(
                models,
                key=lambda m: (m.req_rate / m.slo) * (1.0 + random.uniform(-0.02, 0.02)),
                reverse=True)
        assign = pack(order)
        if assign is None:
            continue
        score = kvpr_of(assign)
        if score < best_score:
            best_score = score
            best_assign = assign

    if best_assign is not None:
        return best_assign

    # Fallback: memory-only first-fit (no KVPR constraint)
    rem = [GPU_MEM_SIZE] * gpu_num
    placement = {i: [] for i in range(gpu_num)}
    for m in base:
        placed = False
        for i in range(gpu_num):
            if m.model_size <= rem[i]:
                rem[i] -= m.model_size
                placement[i].append(m)
                placed = True
                break
        if not placed:
            raise ValueError(
                f"Unable to place model of size {m.model_size} GB on any GPU."
            )
    return placement

    # --- (unused below, kept unreachable) ---
    if False:
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        W = [0.0] * gpu_num
        best_i = None
        best_val = float('inf')
        placement[0].append(None)
        used[0] += 0
        W[0] += 0

    # Local-search phase removed; the binary-search packing with randomized
    # restarts above handles placement and returns early.
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