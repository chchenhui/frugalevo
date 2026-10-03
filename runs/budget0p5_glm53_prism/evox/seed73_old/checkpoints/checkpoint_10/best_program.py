GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _greedy(gpu_num, sorted_models):
    """Place each model on the GPU whose resulting KVPR is lowest."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted_models:
        best_idx, best_kvpr = None, float('inf')
        for g in range(gpu_num):
            if m.model_size <= free[g] and free[g] - m.model_size > 0:
                kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best_idx = kvpr, g
        if best_idx is None:
            return None
        placement[best_idx].append(m)
        load[best_idx] += m.req_rate / m.slo
        free[best_idx] -= m.model_size
    return placement


def _greedy_fallback(gpu_num, sorted_models):
    """Fallback: place each model on the GPU with lowest current KVPR that fits."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num
    for m in sorted_models:
        best_idx, best_ratio = None, float('inf')
        for g in range(gpu_num):
            if m.model_size <= free[g] and free[g] > 0:
                ratio = load[g] / free[g]
                if ratio < best_ratio:
                    best_ratio, best_idx = ratio, g
        if best_idx is None:
            return None
        placement[best_idx].append(m)
        load[best_idx] += m.req_rate / m.slo
        free[best_idx] -= m.model_size
    return placement


def _gpu_kvpr(mlist):
    """KVPR of a single GPU's model list."""
    free = GPU_MEM_SIZE - sum(m.model_size for m in mlist)
    if not mlist or free <= 0:
        return 0.0
    return sum(m.req_rate / m.slo for m in mlist) / free


def _local_search(placement):
    """Hill-climbing: repeatedly move a model off the max-KVPR GPU if it
    lowers the overall max KVPR, until no improvement is found."""
    gpus = list(placement.keys())
    kvprs = {g: _gpu_kvpr(placement[g]) for g in gpus}
    improved = True
    while improved:
        improved = False
        src = max(gpus, key=lambda g: kvprs[g])
        for dst in gpus:
            if dst == src:
                continue
            src_free = GPU_MEM_SIZE - sum(m.model_size for m in placement[src])
            dst_free = GPU_MEM_SIZE - sum(m.model_size for m in placement[dst])
            for m in list(placement[src]):
                if m.model_size <= dst_free:
                    new_src = [x for x in placement[src] if x is not m]
                    new_dst = placement[dst] + [m]
                    new_max = max(_gpu_kvpr(new_src), _gpu_kvpr(new_dst),
                                  max(kvprs[g] for g in gpus if g not in (src, dst)))
                    if new_max < kvprs[src] - 1e-12:
                        placement[src] = new_src
                        placement[dst] = new_dst
                        kvprs[src] = _gpu_kvpr(new_src)
                        kvprs[dst] = _gpu_kvpr(new_dst)
                        improved = True
                        break
            if improved:
                break
    return placement


def compute_model_placement(gpu_num, models):
    """
    Greedy placement minimizing max KVPR: each model goes to the GPU where
    the resulting KVPR is lowest. Multiple model orderings are tried
    (pressure desc, size desc, pressure/size desc) and the best placement
    (lowest max KVPR) is returned. A local-search refinement then moves
    models off the most pressured GPU to further reduce max KVPR.
    Falls back to current-KVPR greedy if the resulting-KVPR greedy fails.
    """
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
    ]
    best, best_max = None, float('inf')
    for ms in orders:
        p = _greedy(gpu_num, ms)
        if p is None:
            continue
        mx = 0.0
        for g, mlist in p.items():
            free = GPU_MEM_SIZE - sum(m.model_size for m in mlist)
            if mlist and free > 0:
                mx = max(mx, sum(m.req_rate / m.slo for m in mlist) / free)
        if mx < best_max:
            best, best_max = p, mx
    if best is None:
        # Fallback: try the more permissive current-KVPR greedy
        for ms in orders:
            p = _greedy_fallback(gpu_num, ms)
            if p is not None:
                mx = max(_gpu_kvpr(mlist) for mlist in p.values())
                if mx < best_max:
                    best, best_max = p, mx
    if best is None:
        raise ValueError("Unable to place all models within GPU memory")
    return _local_search(best)

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
