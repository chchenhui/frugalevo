GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

import random

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses greedy with randomized tie-breaking, multiple restarts, local search,
    and returns the best of N restarts.
    """

    def kvpr(rate, free):
        return rate / free if free > 0 else float('inf')

    def greedy_construct(rng, tie_tol):
        # Multiple sort orders to diversify
        key_mode = rng.randint(0, 2)
        if key_mode == 0:
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True)
        elif key_mode == 1:
            sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9), reverse=True)
        else:
            sorted_models = sorted(models, key=lambda m: m.model_size, reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        rate = [0.0] * gpu_num

        for model in sorted_models:
            cands = []
            best_ratio = float('inf')
            for g in range(gpu_num):
                if model.model_size <= free_mem[g]:
                    nf = free_mem[g] - model.model_size
                    r = kvpr(rate[g] + model.req_rate / model.slo, nf)
                    if r < best_ratio - 1e-12:
                        best_ratio = r
                        cands = [g]
                    elif r < best_ratio * (1 + tie_tol):
                        cands.append(g)
            if not cands:
                return None
            g = rng.choice(cands)
            placement[g].append(model)
            free_mem[g] -= model.model_size
            rate[g] += model.req_rate / model.slo
        return placement, free_mem, rate

    def local_search(placement, free_mem, rate):
        improved = True
        while improved:
            improved = False
            ratios = [kvpr(rate[g], free_mem[g]) for g in range(gpu_num)]
            worst = max(range(gpu_num), key=lambda g: ratios[g])
            for m in list(placement[worst]):
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    if m.model_size <= free_mem[g]:
                        old_max = max(ratios)
                        new_r_worst = kvpr(rate[worst] - m.req_rate / m.slo, free_mem[worst] + m.model_size)
                        new_r_g = kvpr(rate[g] + m.req_rate / m.slo, free_mem[g] - m.model_size)
                        others = [ratios[k] for k in range(gpu_num) if k not in (worst, g)]
                        if max(others + [new_r_worst, new_r_g]) < old_max - 1e-12:
                            placement[worst].remove(m)
                            placement[g].append(m)
                            free_mem[worst] += m.model_size
                            free_mem[g] -= m.model_size
                            rate[worst] -= m.req_rate / m.slo
                            rate[g] += m.req_rate / m.slo
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # Swap phase
            for m_worst in list(placement[worst]):
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    for m_other in list(placement[g]):
                        r_w = m_worst.req_rate / m_worst.slo
                        r_o = m_other.req_rate / m_other.slo
                        fw = free_mem[worst] + m_worst.model_size - m_other.model_size
                        fg = free_mem[g] + m_other.model_size - m_worst.model_size
                        if fw <= 0 or fg <= 0:
                            continue
                        new_r_worst = (rate[worst] - r_w + r_o) / fw
                        new_r_g = (rate[g] - r_o + r_w) / fg
                        others = [ratios[k] for k in range(gpu_num) if k not in (worst, g)]
                        if max(others + [new_r_worst, new_r_g]) < max(ratios) - 1e-12:
                            placement[worst].remove(m_worst)
                            placement[g].remove(m_other)
                            placement[worst].append(m_other)
                            placement[g].append(m_worst)
                            free_mem[worst] = fw
                            free_mem[g] = fg
                            rate[worst] = rate[worst] - r_w + r_o
                            rate[g] = rate[g] - r_o + r_w
                            improved = True
                            break
                if improved:
                    break
        return placement, free_mem, rate

    def max_ratio(free_mem, rate):
        vals = [kvpr(rate[g], free_mem[g]) for g in range(gpu_num) if free_mem[g] > 0]
        return max(vals) if vals else 0.0

    N_RESTARTS = 20
    TIE_TOLS = [0.0, 0.05, 0.10, 0.20]
    best_result = None
    best_score = float('inf')

    # First, deterministic run (seed 0, no tie tolerance)
    for attempt in range(N_RESTARTS):
        rng = random.Random(attempt)
        tie_tol = TIE_TOLS[attempt % len(TIE_TOLS)]
        built = greedy_construct(rng, tie_tol)
        if built is None:
            raise ValueError("Unable to place all models on the given GPUs.")
        placement, free_mem, rate = local_search(*built)
        score = max_ratio(free_mem, rate)
        if score < best_score:
            best_score = score
            best_result = placement

    return best_result

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
