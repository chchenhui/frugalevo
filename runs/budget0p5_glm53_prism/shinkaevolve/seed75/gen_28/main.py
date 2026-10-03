GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Best-of-N greedy construction with randomized tie-breaking, followed by
    local-search refinement. All KVPR ratios are computed via a zero-safe helper.
    """

    import random

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    def kvpr_of(load, free):
        denom = GPU_MEM_SIZE - free
        if denom <= 0:
            return float('inf') if load > 0 else 0.0
        return load / denom

    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    def run_greedy():
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        for model in sorted_models:
            w = model.req_rate / model.slo
            candidates = []
            best_val = float('inf')
            for g in range(gpu_num):
                if model.model_size > free_mem[g]:
                    continue
                new_free = free_mem[g] - model.model_size
                denom = GPU_MEM_SIZE - new_free
                if denom <= 0:
                    continue
                val = (load[g] + w) / denom
                if val < best_val - 1e-12:
                    best_val = val
                    candidates = [g]
                elif val <= best_val + 1e-9:
                    candidates.append(g)
            if not candidates:
                return None
            g = random.choice(candidates)
            placement[g].append(model)
            free_mem[g] -= model.model_size
            load[g] += w
        return placement, free_mem, load

    def refine(placement, free_mem, load):
        # Local search: move models off the worst GPU if it lowers max KVPR
        def kvpr(g):
            return kvpr_of(load[g], free_mem[g])

        def max_kvpr(exclude=None):
            vals = [kvpr(g) for g in range(gpu_num) if g != exclude]
            return max(vals) if vals else 0.0

        for _ in range(200):
            worst = max(range(gpu_num), key=lambda g: kvpr(g))
            cur_max = max_kvpr()
            improved = False
            for m_idx in range(len(placement[worst])):
                model = placement[worst][m_idx]
                w = model.req_rate / model.slo
                for g in range(gpu_num):
                    if g == worst or model.model_size > free_mem[g]:
                        continue
                    placement[worst].pop(m_idx)
                    free_mem[worst] += model.model_size
                    load[worst] -= w
                    free_mem[g] -= model.model_size
                    load[g] += w
                    if max_kvpr() < cur_max - 1e-12:
                        placement[g].append(model)
                        improved = True
                        break
                    # revert
                    free_mem[g] += model.model_size
                    load[g] -= w
                    free_mem[worst] -= model.model_size
                    load[worst] += w
                    placement[worst].insert(m_idx, model)
                if improved:
                    break
            if not improved:
                break
        return placement, free_mem, load

    best_placement = None
    best_max = float('inf')

    N = 12
    for _ in range(N):
        run = run_greedy()
        if run is None:
            # Infeasible under random tie-breaking; fall back to deterministic scan
            # to see if any feasible placement exists.
            placement = {g: [] for g in range(gpu_num)}
            free_mem = [GPU_MEM_SIZE] * gpu_num
            load = [0.0] * gpu_num
            feasible = True
            for model in sorted_models:
                w = model.req_rate / model.slo
                best_g, best_val = None, float('inf')
                for g in range(gpu_num):
                    if model.model_size > free_mem[g]:
                        continue
                    denom = GPU_MEM_SIZE - (free_mem[g] - model.model_size)
                    if denom <= 0:
                        continue
                    val = (load[g] + w) / denom
                    if val < best_val:
                        best_val, best_g = val, g
                if best_g is None:
                    feasible = False
                    break
                placement[best_g].append(model)
                free_mem[best_g] -= model.model_size
                load[best_g] += w
            if not feasible:
                raise ValueError(
                    f"Unable to place models on {gpu_num} GPUs. "
                    f"Free memory per GPU: {free_mem}"
                )
            run = (placement, free_mem, load)

        placement, free_mem, load = refine(*run)
        cur_max = max(kvpr_of(load[g], free_mem[g]) for g in range(gpu_num))
        if cur_max < best_max:
            best_max = cur_max
            best_placement = placement

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
