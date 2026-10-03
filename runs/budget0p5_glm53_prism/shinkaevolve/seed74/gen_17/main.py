GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses binary search on the target KVPR (lambda) with warm-started bounds:
      lo = max single-model pressure (theoretical lower bound),
      hi = KVPR achieved by a greedy memory-aware packing (always feasible).
    For each candidate lambda, try to pack all models so that every GPU's
    load / remaining memory stays <= lambda; return the best feasible packing.
    """

    def greedy_pack():
        """Memory-aware greedy packing; returns (placement, max_kvpr)."""
        placement = {g: [] for g in range(gpu_num)}
        rem_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        # Sort by pressure-per-memory descending to place hardest models first
        order = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
        for m in order:
            w = m.req_rate / m.slo
            best_g, best_ratio = None, float('inf')
            for g in range(gpu_num):
                free_after = rem_mem[g] - m.model_size
                if free_after <= 0:
                    continue
                ratio = (load[g] + w) / free_after
                if ratio < best_ratio:
                    best_ratio, best_g = ratio, g
            if best_g is None:
                # fallback: allow exact fill
                for g in range(gpu_num):
                    if m.model_size <= rem_mem[g]:
                        best_g = g
                        break
            if best_g is None:
                return None, float('inf')
            placement[best_g].append(m)
            load[best_g] += w
            rem_mem[best_g] -= m.model_size
        worst = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
            ld = sum(m.req_rate / m.slo for m in placement[g])
            worst = max(worst, ld / rem if rem > 0 else (float('inf') if ld > 0 else 0.0))
        return placement, worst

    def try_pack(lam, order):
        """Try to pack all models with per-GPU KVPR <= lam. Returns placement or None."""
        placement = {g: [] for g in range(gpu_num)}
        rem_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            w = m.req_rate / m.slo
            best_g, best_ratio = None, float('inf')
            for g in range(gpu_num):
                free_after = rem_mem[g] - m.model_size
                if free_after <= 0:
                    continue
                new_load = load[g] + w
                if new_load > lam * free_after + 1e-12:
                    continue
                ratio = new_load / free_after
                if ratio < best_ratio:
                    best_ratio, best_g = ratio, g
            if best_g is None:
                return None
            placement[best_g].append(m)
            load[best_g] += w
            rem_mem[best_g] -= m.model_size
        return placement

    # Warm-started bounds
    base_placement, hi = greedy_pack()
    if base_placement is None:
        raise ValueError("Unable to place all models in GPU memory")
    lo = 0.0
    for m in models:
        free = GPU_MEM_SIZE - m.model_size
        if free > 0:
            lo = max(lo, (m.req_rate / m.slo) / free)
    lo = min(lo, hi)

    # Multiple orderings as seeds for the feasibility check
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: -(m.req_rate / m.slo) / max(m.model_size, 1e-9)),
        sorted(models, key=lambda m: m.model_size, reverse=True),
    ]

    best_placement, best_val = base_placement, hi
    for _ in range(60):
        mid = (lo + hi) / 2.0
        packed = None
        for order in orders:
            packed = try_pack(mid, order)
            if packed is not None:
                break
        if packed is not None:
            # verify actual max kvpr
            worst = 0.0
            for g in packed:
                rem = GPU_MEM_SIZE - sum(m.model_size for m in packed[g])
                ld = sum(m.req_rate / m.slo for m in packed[g])
                worst = max(worst, ld / rem if rem > 0 else (float('inf') if ld > 0 else 0.0))
            if worst < best_val:
                best_val, best_placement = worst, packed
            hi = mid
        else:
            lo = mid

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