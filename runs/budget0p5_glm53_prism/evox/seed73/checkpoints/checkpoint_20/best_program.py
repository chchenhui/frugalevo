GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy KVPR-minimizing placement: try several model
    orderings with two GPU-selection criteria (lowest current KVPR or
    lowest resulting KVPR after placement), then refine with a local
    search that moves models off the worst GPU and additionally swaps
    models between the worst GPU and others whenever it lowers the
    overall maximum KVPR. Keep the best placement found.
    """

    def kvpr(gpu_models):
        free = GPU_MEM_SIZE - sum(m.model_size for m in gpu_models)
        if free <= 0:
            return float('inf')
        return sum(m.req_rate / m.slo for m in gpu_models) / free

    def greedy(order, use_resulting):
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float('inf')
            for g in range(gpu_num):
                if model.model_size <= free_mem[g] and free_mem[g] > 0:
                    ratio = (load[g] + (model.req_rate / model.slo
                                        if use_resulting else 0.0)) / free_mem[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            free_mem[best_idx] -= model.model_size
        return placement

    def refine(placement):
        """Move a model from the worst GPU elsewhere if it lowers max KVPR."""
        free_mem = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
                    for g in range(gpu_num)]
        improved = True
        while improved:
            improved = False
            worst = max(placement, key=lambda g: kvpr(placement[g]))
            old_max = max(kvpr(placement[g]) for g in range(gpu_num))
            for g in range(gpu_num):
                if g == worst:
                    continue
                for m in list(placement[worst]):
                    if m.model_size <= free_mem[g]:
                        placement[worst].remove(m)
                        placement[g].append(m)
                        if max(kvpr(placement[x]) for x in range(gpu_num)) < old_max:
                            free_mem[worst] += m.model_size
                            free_mem[g] -= m.model_size
                            improved = True
                            break
                        placement[g].remove(m)
                        placement[worst].append(m)
                if improved:
                    break
            if improved:
                continue
            # Swap pass: exchange a model on the worst GPU with one on
            # another GPU if it lowers the overall max KVPR.
            for g in range(gpu_num):
                if g == worst:
                    continue
                for m in list(placement[worst]):
                    for m2 in list(placement[g]):
                        if (m.model_size - m2.model_size) <= free_mem[g] and \
                           (m2.model_size - m.model_size) <= free_mem[worst]:
                            placement[worst].remove(m)
                            placement[g].remove(m2)
                            placement[worst].append(m2)
                            placement[g].append(m)
                            if max(kvpr(placement[x]) for x in range(gpu_num)) < old_max:
                                free_mem[worst] += m.model_size - m2.model_size
                                free_mem[g] += m2.model_size - m.model_size
                                improved = True
                                break
                            placement[worst].remove(m2)
                            placement[g].remove(m)
                            placement[worst].append(m)
                            placement[g].append(m2)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: m.req_rate / m.slo / m.model_size, reverse=True),
    ]

    best_placement, best_score = None, float('inf')
    for order in orders:
        for use_resulting in (False, True):
            placement = greedy(order, use_resulting)
            if placement is None:
                continue
            placement = refine(placement)
            score = max(kvpr(placement[g]) for g in range(gpu_num))
            if score < best_score:
                best_score, best_placement = score, placement

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
