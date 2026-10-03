GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    """

    def kvpr(rate, free):
        return rate / free if free > 0 else float('inf')

    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num
    rate = [0.0] * gpu_num

    # Greedy seed
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if model.model_size <= free_mem[gpu_id]:
                new_ratio = kvpr(rate[gpu_id] + model.req_rate / model.slo,
                                 free_mem[gpu_id] - model.model_size)
                if new_ratio < best_ratio:
                    best_ratio = new_ratio
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU."
            )
        placement[best_idx].append(model)
        free_mem[best_idx] -= model.model_size
        rate[best_idx] += model.req_rate / model.slo

    ratios = [kvpr(rate[g], free_mem[g]) for g in range(gpu_num)]

    def try_moves_from(worst):
        """Try a single move or swap involving 'worst' that strictly lowers max KVPR."""
        for m in list(placement[worst]):
            for g in range(gpu_num):
                if g == worst:
                    continue
                if m.model_size <= free_mem[g]:
                    new_r_worst = kvpr(rate[worst] - m.req_rate / m.slo,
                                        free_mem[worst] + m.model_size)
                    new_r_g = kvpr(rate[g] + m.req_rate / m.slo,
                                   free_mem[g] - m.model_size)
                    others = [ratios[k] for k in range(gpu_num) if k not in (worst, g)]
                    if max(others + [new_r_worst, new_r_g]) < max(ratios) - 1e-12:
                        placement[worst].remove(m)
                        placement[g].append(m)
                        free_mem[worst] += m.model_size
                        free_mem[g] -= m.model_size
                        rate[worst] -= m.req_rate / m.slo
                        rate[g] += m.req_rate / m.slo
                        return True

        # swaps
        for m_worst in list(placement[worst]):
            for g in range(gpu_num):
                if g == worst:
                    continue
                for m_other in list(placement[g]):
                    r_w = m_worst.req_rate / m_worst.slo
                    r_o = m_other.req_rate / m_other.slo
                    free_worst = free_mem[worst] + m_worst.model_size - m_other.model_size
                    free_g = free_mem[g] + m_other.model_size - m_worst.model_size
                    if free_worst <= 0 or free_g <= 0:
                        continue
                    new_r_worst = (rate[worst] - r_w + r_o) / free_worst
                    new_r_g = (rate[g] - r_o + r_w) / free_g
                    others = [ratios[k] for k in range(gpu_num) if k not in (worst, g)]
                    if max(others + [new_r_worst, new_r_g]) < max(ratios) - 1e-12:
                        placement[worst].remove(m_worst)
                        placement[g].remove(m_other)
                        placement[worst].append(m_other)
                        placement[g].append(m_worst)
                        free_mem[worst] = free_worst
                        free_mem[g] = free_g
                        rate[worst] = rate[worst] - r_w + r_o
                        rate[g] = rate[g] - r_o + r_w
                        return True

        # 2-for-1 swaps: replace one model on 'worst' with two models from g
        for m_worst in list(placement[worst]):
            r_w = m_worst.req_rate / m_worst.slo
            for g in range(gpu_num):
                if g == worst:
                    continue
                others_g = list(placement[g])
                for i in range(len(others_g)):
                    for j in range(i + 1, len(others_g)):
                        m1, m2 = others_g[i], others_g[j]
                        r1 = m1.req_rate / m1.slo
                        r2 = m2.req_rate / m2.slo
                        free_worst = free_mem[worst] + m_worst.model_size - m1.model_size - m2.model_size
                        free_g = free_mem[g] + m1.model_size + m2.model_size - m_worst.model_size
                        if free_worst <= 0 or free_g <= 0:
                            continue
                        new_r_worst = (rate[worst] - r_w + r1 + r2) / free_worst
                        new_r_g = (rate[g] - r1 - r2 + r_w) / free_g
                        others = [ratios[k] for k in range(gpu_num) if k not in (worst, g)]
                        if max(others + [new_r_worst, new_r_g]) < max(ratios) - 1e-12:
                            placement[worst].remove(m_worst)
                            placement[g].remove(m1)
                            placement[g].remove(m2)
                            placement[worst].append(m1)
                            placement[worst].append(m2)
                            placement[g].append(m_worst)
                            free_mem[worst] = free_worst
                            free_mem[g] = free_g
                            rate[worst] = rate[worst] - r_w + r1 + r2
                            rate[g] = rate[g] - r1 - r2 + r_w
                            return True
        return False

    # Alternate targeting the worst and runner-up GPUs
    improved = True
    while improved:
        improved = False
        ratios = [kvpr(rate[g], free_mem[g]) for g in range(gpu_num)]
        order = sorted(range(gpu_num), key=lambda g: ratios[g], reverse=True)
        for target in order[:2]:
            if try_moves_from(target):
                improved = True
                break
        if improved:
            # refresh ratios after successful move
            ratios = [kvpr(rate[g], free_mem[g]) for g in range(gpu_num)]
            continue

        # Final attempt: direct swap between the two worst GPUs
        if gpu_num >= 2:
            w1, w2 = order[0], order[1]
            for m1 in list(placement[w1]):
                for m2 in list(placement[w2]):
                    r1 = m1.req_rate / m1.slo
                    r2 = m2.req_rate / m2.slo
                    f1 = free_mem[w1] + m1.model_size - m2.model_size
                    f2 = free_mem[w2] + m2.model_size - m1.model_size
                    if f1 <= 0 or f2 <= 0:
                        continue
                    new_r1 = (rate[w1] - r1 + r2) / f1
                    new_r2 = (rate[w2] - r2 + r1) / f2
                    others = [ratios[k] for k in range(gpu_num) if k not in (w1, w2)]
                    if max(others + [new_r1, new_r2]) < max(ratios) - 1e-12:
                        placement[w1].remove(m1)
                        placement[w2].remove(m2)
                        placement[w1].append(m2)
                        placement[w2].append(m1)
                        free_mem[w1] = f1
                        free_mem[w2] = f2
                        rate[w1] = rate[w1] - r1 + r2
                        rate[w2] = rate[w2] - r2 + r1
                        improved = True
                        break
                if improved:
                    break

    return placement

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