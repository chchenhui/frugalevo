GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Approach: best-fit decreasing construction followed by local-search
    rebalancing (move/swap of models) driven directly by the max-KVPR objective.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")
    if not models:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}

    # --- Helper state ---
    def gpu_mem(gpu_id):
        return sum(m.model_size for m in placement[gpu_id])

    def gpu_load(gpu_id):
        return sum(m.req_rate / m.slo for m in placement[gpu_id])

    def kvpr(gpu_id):
        remaining = GPU_MEM_SIZE - gpu_mem(gpu_id)
        if remaining <= 0:
            return float('inf')
        return gpu_load(gpu_id) / remaining

    def max_kvpr_gpus():
        vals = [(kvpr(g), g) for g in range(gpu_num)]
        worst = max(v for v, _ in vals)
        return worst, [g for v, g in vals if v == worst]

    # --- λ-feasibility packing with binary search over multiple orderings ---
    w = [m.req_rate / m.slo for m in models]

    def pack(order, lam):
        """First-fit packing under max-KVPR constraint lam. Returns placement or None."""
        pl = {g: [] for g in range(gpu_num)}
        mem = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for idx in order:
            m = models[idx]
            placed = False
            for g in range(gpu_num):
                new_mem = mem[g] + m.model_size
                if new_mem >= GPU_MEM_SIZE:
                    continue
                free = GPU_MEM_SIZE - new_mem
                if load[g] + w[idx] <= lam * free + 1e-12:
                    pl[g].append(m)
                    mem[g] = new_mem
                    load[g] += w[idx]
                    placed = True
                    break
            if not placed:
                return None
        return pl

    def true_max(pl):
        worst = 0.0
        for g in range(gpu_num):
            mem = sum(m.model_size for m in pl[g])
            free = GPU_MEM_SIZE - mem
            if free <= 1e-12:
                return float('inf')
            worst = max(worst, sum(m.req_rate / m.slo for m in pl[g]) / free)
        return worst

    orderings = [
        sorted(range(len(models)), key=lambda i: w[i] / models[i].model_size, reverse=True),
        sorted(range(len(models)), key=lambda i: models[i].model_size, reverse=True),
        sorted(range(len(models)), key=lambda i: w[i], reverse=True),
        sorted(range(len(models)), key=lambda i: w[i] + models[i].model_size, reverse=True),
    ]

    best_placement = None
    best_val = float('inf')
    for order in orderings:
        if pack(order, 1e9) is None:
            continue
        lo, hi = 0.0, 1e9
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if pack(order, mid) is not None:
                hi = mid
            else:
                lo = mid
        pl = pack(order, hi * (1 + 1e-9))
        if pl is not None:
            val = true_max(pl)
            if val < best_val:
                best_val = val
                best_placement = pl

    if best_placement is None:
        raise ValueError("Unable to place models on any GPU.")

    return best_placement

    # local search removed: replaced by λ binary-search packing above

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