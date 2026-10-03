GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement over many model orderings, each refined by
    an exhaustive best-improvement local search over all single moves and
    all pairwise swaps; returns the placement with the lowest max KVPR.
    """

    def build(order):
        placement = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            best, best_v = None, float('inf')
            for g in range(gpu_num):
                if m.model_size <= free[g] and free[g] > 0:
                    v = (load[g] + m.req_rate / m.slo) / free[g]
                    if v < best_v:
                        best_v, best = v, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            free[best] -= m.model_size
        return placement

    def local_search(placement):
        free = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
                for g in range(gpu_num)]
        load = [sum(m.req_rate / m.slo for m in placement[g])
                for g in range(gpu_num)]

        def kv(g):
            return load[g] / free[g] if free[g] > 0 else float('inf')

        cur = max(kv(g) for g in range(gpu_num))
        while True:
            best_gain, best_op = 1e-9, None
            # try moving any model to any other GPU
            for src in range(gpu_num):
                for m in list(placement[src]):
                    r1, s1 = m.req_rate / m.slo, m.model_size
                    for dst in range(gpu_num):
                        if dst == src or s1 > free[dst] or free[dst] - s1 <= 0:
                            continue
                        others = max([kv(g) for g in range(gpu_num)
                                      if g != src and g != dst] or [0.0])
                        nm = max(others, (load[src] - r1) / (free[src] + s1),
                                 (load[dst] + r1) / (free[dst] - s1))
                        if cur - nm > best_gain:
                            best_gain, best_op = cur - nm, ('m', src, dst, m)
            # try swapping any pair of models between two GPUs
            for src in range(gpu_num):
                for m1 in list(placement[src]):
                    r1, s1 = m1.req_rate / m1.slo, m1.model_size
                    for dst in range(src + 1, gpu_num):
                        for m2 in list(placement[dst]):
                            r2, s2 = m2.req_rate / m2.slo, m2.model_size
                            fs, fd = free[src] + s1 - s2, free[dst] + s2 - s1
                            if fs <= 0 or fd <= 0:
                                continue
                            others = max([kv(g) for g in range(gpu_num)
                                          if g != src and g != dst] or [0.0])
                            nm = max(others, (load[src] - r1 + r2) / fs,
                                     (load[dst] - r2 + r1) / fd)
                            if cur - nm > best_gain:
                                best_gain, best_op = cur - nm, ('s', src, dst, m1, m2)
            if best_op is None:
                break
            if best_op[0] == 'm':
                _, src, dst, m = best_op
                placement[src].remove(m)
                placement[dst].append(m)
                free[src] += m.model_size
                free[dst] -= m.model_size
                load[src] -= m.req_rate / m.slo
                load[dst] += m.req_rate / m.slo
            else:
                _, src, dst, m1, m2 = best_op
                placement[src].remove(m1)
                placement[src].append(m2)
                placement[dst].remove(m2)
                placement[dst].append(m1)
                free[src] += m1.model_size - m2.model_size
                free[dst] += m2.model_size - m1.model_size
                load[src] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                load[dst] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
            cur -= best_gain
        return placement

    def score(placement):
        worst = 0.0
        for g in placement:
            free = GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
            if free <= 0:
                return float('inf')
            worst = max(worst, sum(m.req_rate / m.slo for m in placement[g]) / free)
        return worst

    r = lambda m: m.req_rate / m.slo
    orders = [
        sorted(models, key=r, reverse=True),
        sorted(models, key=r),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: r(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) / m.model_size),
    ]
    import random
    rng = random.Random(0)
    for _ in range(20):
        o = list(models)
        rng.shuffle(o)
        orders.append(o)

    best_placement, best_score = None, float('inf')
    for order in orders:
        p = build(order)
        if p is None:
            continue
        p = local_search(p)
        s = score(p)
        if s < best_score:
            best_score, best_placement = s, p
    if best_placement is None:
        raise ValueError("Unable to place models on any GPU")
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
