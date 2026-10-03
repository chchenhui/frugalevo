GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement (each model goes to the GPU yielding the
    lowest resulting KVPR), refined by a best-improvement local search over
    single-model moves and pairwise swaps. Several greedy starting orders
    (by pressure, model size, pressure density, req_rate) are tried and the
    placement with the lowest maximum KVPR is returned.
    """
    if not models:
        return {g: [] for g in range(gpu_num)}

    def p(m):
        return m.req_rate / m.slo

    starts = [
        sorted(models, key=p, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: p(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate, reverse=True),
    ]

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            best, best_r = None, float('inf')
            for g in range(gpu_num):
                r = rem[g] - m.model_size
                if r > 0:
                    ratio = (load[g] + p(m)) / r
                    if ratio < best_r:
                        best_r, best = ratio, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += p(m)
            rem[best] -= m.model_size
        return placement, rem, load

    def local_search(placement, rem, load):
        # Best-improvement: evaluate all moves and swaps, apply only the
        # single change that most reduces the maximum KVPR; repeat.
        while True:
            cur = [load[g] / rem[g] for g in range(gpu_num)]
            cur_max = max(cur)
            best_gain = 1e-12
            best_act = None
            # Candidate moves (only src and dst KVPR change)
            for src in range(gpu_num):
                for m in placement[src]:
                    r_src = rem[src] + m.model_size
                    l_src = load[src] - p(m)
                    for dst in range(gpu_num):
                        if dst == src or rem[dst] - m.model_size <= 0:
                            continue
                        others = max((cur[g] for g in range(gpu_num)
                                      if g != src and g != dst), default=0.0)
                        new_max = max(others, l_src / r_src,
                                      (load[dst] + p(m)) / (rem[dst] - m.model_size))
                        if cur_max - new_max > best_gain:
                            best_gain = cur_max - new_max
                            best_act = ('m', m, None, src, dst)
            # Candidate swaps
            for src in range(gpu_num):
                for m in placement[src]:
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        for o in placement[dst]:
                            r_src = rem[src] + m.model_size - o.model_size
                            r_dst = rem[dst] + o.model_size - m.model_size
                            if r_src <= 0 or r_dst <= 0:
                                continue
                            others = max((cur[g] for g in range(gpu_num)
                                          if g != src and g != dst), default=0.0)
                            new_max = max(others,
                                          (load[src] - p(m) + p(o)) / r_src,
                                          (load[dst] - p(o) + p(m)) / r_dst)
                            if cur_max - new_max > best_gain:
                                best_gain = cur_max - new_max
                                best_act = ('s', m, o, src, dst)
            if best_act is None:
                break
            _, m, o, src, dst = best_act
            placement[src].remove(m)
            if o is None:
                placement[dst].append(m)
                load[src] -= p(m)
                rem[src] += m.model_size
                load[dst] += p(m)
                rem[dst] -= m.model_size
            else:
                placement[src].append(o)
                placement[dst].remove(o)
                placement[dst].append(m)
                load[src] += p(o) - p(m)
                rem[src] += m.model_size - o.model_size
                load[dst] += p(m) - p(o)
                rem[dst] += o.model_size - m.model_size
        return placement

    best_place, best_max = None, float('inf')
    for order in starts:
        res = greedy(order)
        if res is None:
            continue
        placement, rem, load = res
        placement = local_search(placement, rem, load)
        cur_max = max(load[g] / rem[g] for g in range(gpu_num))
        if cur_max < best_max:
            best_max, best_place = cur_max, placement
    if best_place is None:
        raise ValueError("Unable to place all models within GPU memory")
    return best_place

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
