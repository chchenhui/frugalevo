GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _kvpr(mlist):
    """KVPR of a single GPU's model list."""
    free = GPU_MEM_SIZE - sum(m.model_size for m in mlist)
    if not mlist or free <= 0:
        return 0.0
    return sum(m.req_rate / m.slo for m in mlist) / free


def _greedy(gpu_num, sorted_models, use_resulting):
    """Place each model on the GPU with the lowest KVPR that fits it.
    If use_resulting, compare KVPR after placement; else compare current KVPR."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted_models:
        best_idx, best_val = None, float('inf')
        for g in range(gpu_num):
            if m.model_size <= free[g] and free[g] - m.model_size > 0:
                if use_resulting:
                    val = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                else:
                    val = load[g] / free[g]
                if val < best_val:
                    best_val, best_idx = val, g
        if best_idx is None:
            return None
        placement[best_idx].append(m)
        load[best_idx] += m.req_rate / m.slo
        free[best_idx] -= m.model_size
    return placement


def _refine(placement):
    """Hill-climb: move a model off the worst-KVPR GPU when it lowers max KVPR."""
    for _ in range(len(placement) * 4):
        k = {g: _kvpr(placement[g]) for g in placement}
        src = max(placement, key=lambda g: k[g])
        if k[src] <= 0:
            break
        improved = False
        for m in list(placement[src]):
            for dst in placement:
                if dst == src:
                    continue
                if m.model_size > GPU_MEM_SIZE - sum(x.model_size for x in placement[dst]):
                    continue
                placement[src].remove(m)
                placement[dst].append(m)
                if max(_kvpr(v) for v in placement.values()) < max(k.values()) - 1e-12:
                    improved = True
                    break
                placement[dst].remove(m)
                placement[src].append(m)
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """
    Greedy placement minimizing max KVPR: two greedy variants (lowest
    current KVPR, lowest resulting KVPR) are each tried over three model
    orderings (pressure desc, size desc, pressure/size desc); the best
    placement is refined by a local search moving models off the worst GPU.
    """
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
    ]
    best, best_max = None, float('inf')
    for ms in orders:
        for use_resulting in (True, False):
            p = _greedy(gpu_num, ms, use_resulting)
            if p is None:
                continue
            mx = max(_kvpr(v) for v in p.values())
            if mx < best_max:
                best, best_max = p, mx
    if best is None:
        raise ValueError("Unable to place all models within GPU memory")
    return _refine(best)

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
