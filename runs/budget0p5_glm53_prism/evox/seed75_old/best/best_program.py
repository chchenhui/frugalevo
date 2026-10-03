GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement: several deterministic orderings plus seeded
    randomized restarts. Greedy assigns each model to the GPU minimizing the
    resulting KVPR. Each placement is improved by local search (best-improvement
    moves and swaps off the hottest GPU, then a full relocation pass from any
    GPU). Return the placement with the lowest max KVPR across all starts.
    """
    import random

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        pressure = [0.0] * gpu_num
        for model in order:
            best_idx, best_kvpr = None, float('inf')
            for g in range(gpu_num):
                rem = free_mem[g] - model.model_size
                if rem > 0:
                    kvpr = (pressure[g] + model.req_rate / model.slo) / rem
                    if kvpr < best_kvpr:
                        best_kvpr, best_idx = kvpr, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            pressure[best_idx] += model.req_rate / model.slo
            free_mem[best_idx] -= model.model_size
        return placement, pressure, free_mem

    def kvpr_list(pressure, free_mem):
        return [pressure[g] / free_mem[g] if free_mem[g] > 0 else float('inf')
                for g in range(gpu_num)]

    def sorted_profile(vals):
        return sorted(vals, reverse=True)

    def local_search(placement, pressure, free_mem):
        for _ in range(300):
            kvs = kvpr_list(pressure, free_mem)
            cur_prof = sorted_profile(kvs)
            hot = max(range(gpu_num), key=lambda g: kvs[g])
            # best-improvement move of a hot-GPU model (lexicographic)
            best = None
            for model in placement[hot]:
                for dst in range(gpu_num):
                    if dst == hot or model.model_size > free_mem[dst]:
                        continue
                    new_dst = (pressure[dst] + model.req_rate / model.slo) / (free_mem[dst] - model.model_size)
                    new_hot = (pressure[hot] - model.req_rate / model.slo) / (free_mem[hot] + model.model_size)
                    prof = sorted_profile([new_dst if g == dst else
                                           (new_hot if g == hot else kvs[g])
                                           for g in range(gpu_num)])
                    if prof < cur_prof and (best is None or prof < best[0]):
                        best = (prof, model, dst)
            if best is not None:
                _, model, dst = best
                placement[hot].remove(model)
                placement[dst].append(model)
                pressure[hot] -= model.req_rate / model.slo
                free_mem[hot] += model.model_size
                pressure[dst] += model.req_rate / model.slo
                free_mem[dst] -= model.model_size
                continue
            # full pairwise swap neighborhood (lexicographic best-improvement)
            best = None
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for m_a in list(placement[a]):
                        for m_b in list(placement[b]):
                            d = m_a.model_size - m_b.model_size
                            if free_mem[b] + m_b.model_size < m_a.model_size:
                                continue
                            if free_mem[a] + m_a.model_size < m_b.model_size:
                                continue
                            if free_mem[a] + d <= 0 or free_mem[b] - d <= 0:
                                continue
                            p_a = pressure[a] - m_a.req_rate / m_a.slo + m_b.req_rate / m_b.slo
                            p_b = pressure[b] - m_b.req_rate / m_b.slo + m_a.req_rate / m_a.slo
                            new_a = p_a / (free_mem[a] + d)
                            new_b = p_b / (free_mem[b] - d)
                            prof = sorted_profile([new_a if g == a else
                                                   (new_b if g == b else kvs[g])
                                                   for g in range(gpu_num)])
                            if prof < cur_prof and (best is None or prof < best[0]):
                                best = (prof, m_a, m_b, a, b)
            if best is not None:
                _, m_a, m_b, a, b = best
                d = m_a.model_size - m_b.model_size
                placement[a].remove(m_a); placement[b].remove(m_b)
                placement[a].append(m_b); placement[b].append(m_a)
                pressure[a] += m_b.req_rate / m_b.slo - m_a.req_rate / m_a.slo
                pressure[b] += m_a.req_rate / m_a.slo - m_b.req_rate / m_b.slo
                free_mem[a] += d
                free_mem[b] -= d
                continue
            # full relocation pass (lexicographic)
            best = None
            for src in range(gpu_num):
                for model in list(placement[src]):
                    for dst in range(gpu_num):
                        if dst == src or model.model_size > free_mem[dst]:
                            continue
                        new_dst = (pressure[dst] + model.req_rate / model.slo) / (free_mem[dst] - model.model_size)
                        new_src = (pressure[src] - model.req_rate / model.slo) / (free_mem[src] + model.model_size)
                        prof = sorted_profile([new_dst if g == dst else
                                               (new_src if g == src else kvs[g])
                                               for g in range(gpu_num)])
                        if prof < cur_prof and (best is None or prof < best[0]):
                            best = (prof, model, src, dst)
            if best is not None:
                _, model, src, dst = best
                placement[src].remove(model)
                placement[dst].append(model)
                pressure[src] -= model.req_rate / model.slo
                free_mem[src] += model.model_size
                pressure[dst] += model.req_rate / model.slo
                free_mem[dst] -= model.model_size
                continue
            break
        return placement

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9), reverse=True),
        list(models),
    ]
    rng = random.Random(1234)
    for _ in range(40):
        o = list(models)
        rng.shuffle(o)
        orders.append(o)

    best_placement, best_max = None, float('inf')
    for order in orders:
        result = greedy(order)
        if result is None:
            # try first-fit-decreasing by size as a feasibility fallback
            result = greedy(sorted(models, key=lambda m: m.model_size, reverse=True))
        if result is None:
            continue
        placement, pressure, free_mem = result
        placement = local_search(placement, pressure, free_mem)
        # recompute final state
        free_mem = [GPU_MEM_SIZE] * gpu_num
        pressure = [0.0] * gpu_num
        for g, mods in placement.items():
            for m in mods:
                free_mem[g] -= m.model_size
                pressure[g] += m.req_rate / m.slo
        cur_max = max((pressure[g] / free_mem[g]) if free_mem[g] > 0 else float('inf')
                      for g in range(gpu_num))
        if cur_max < best_max:
            best_max, best_placement = cur_max, placement

    if best_placement is None:
        raise ValueError("Cannot fit all models into GPU memory")
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
