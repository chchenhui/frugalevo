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
            if m.model_size < free[g]:
                kvpr = (load[g] + m.req_rate / m.slo) / (free[g] - m.model_size)
                if kvpr < best_kvpr:
                    best_kvpr, best_idx = kvpr, g
        if best_idx is None:
            return None
        placement[best_idx].append(m)
        load[best_idx] += m.req_rate / m.slo
        free[best_idx] -= m.model_size
    return placement


def _kvpr(mlist):
    """KVPR of a single GPU's model list."""
    free = GPU_MEM_SIZE - sum(m.model_size for m in mlist)
    if not mlist or free <= 0:
        return 0.0
    return sum(m.req_rate / m.slo for m in mlist) / free


def _refine(p):
    """Hill-climb: try a move or swap for every model (any source GPU,
    any destination) and accept the first step that strictly lowers the
    maximum KVPR. Sweeping all sources (not just the worst GPU) escapes
    local optima where the max-KVPR GPU has no feasible improving move.
    Repeats until no improving step exists or iteration budget is hit."""
    n_models = sum(len(v) for v in p.values())
    for _ in range(n_models * 4 + 8):
        k = {g: _kvpr(p[g]) for g in p}
        cur = max(k.values())
        if cur <= 0:
            break
        improved = False
        for src in sorted(p, key=lambda g: -k[g]):
            if improved:
                break
            src_free = GPU_MEM_SIZE - sum(x.model_size for x in p[src])
            src_load = sum(x.req_rate / x.slo for x in p[src])
            for m in list(p[src]):
                if improved:
                    break
                for dst in p:
                    if dst == src:
                        continue
                    dst_free = GPU_MEM_SIZE - sum(x.model_size for x in p[dst])
                    # try plain move
                    if m.model_size < dst_free:
                        p[src].remove(m)
                        p[dst].append(m)
                        nk = {g: _kvpr(p[g]) for g in p}
                        if max(nk.values()) < cur - 1e-12:
                            improved = True
                            break
                        p[dst].remove(m)
                        p[src].append(m)
                    # try swap with each model on dst
                    for n in list(p[dst]):
                        if (m.model_size - n.model_size < dst_free and
                                n.model_size - m.model_size < src_free):
                            p[src].remove(m)
                            p[dst].remove(n)
                            p[src].append(n)
                            p[dst].append(m)
                            nk = {g: _kvpr(p[g]) for g in p}
                            if max(nk.values()) < cur - 1e-12:
                                improved = True
                                break
                            p[dst].remove(m)
                            p[src].remove(n)
                            p[src].append(m)
                            p[dst].append(n)
                    if improved:
                        break
        if not improved:
            break
    return p


def compute_model_placement(gpu_num, models):
    """
    Greedy placement minimizing max KVPR: each model goes to the GPU where
    the resulting KVPR is lowest. Three model orderings are tried (pressure
    desc, size desc, pressure/size desc), each refined by a local-search
    pass moving models off the worst GPU, and the best placement returned.
    """
    import random
    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
    ]
    for seed in range(6):
        ms = list(models)
        random.Random(seed).shuffle(ms)
        orders.append(ms)
    best, best_max = None, float('inf')
    for ms in orders:
        p = _greedy(gpu_num, ms)
        if p is None:
            continue
        p = _refine(p)
        mx = max(_kvpr(mlist) for mlist in p.values())
        if mx < best_max:
            best, best_max = p, mx
    if best is None:
        raise ValueError("Unable to place all models within GPU memory")
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
