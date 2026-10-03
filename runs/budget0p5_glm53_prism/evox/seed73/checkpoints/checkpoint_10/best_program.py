GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy KVPR-minimizing placement tried over several model orderings;
    keep the placement with the lowest resulting max KVPR, then refine
    with a local search that moves models off the worst GPU whenever it
    lowers the overall maximum KVPR.
    """

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in order:
            best_idx, best_ratio = None, float('inf')
            for g in range(gpu_num):
                if model.model_size <= free_mem[g] and free_mem[g] > 0:
                    ratio = load[g] / free_mem[g]
                    if ratio < best_ratio:
                        best_ratio, best_idx = ratio, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            load[best_idx] += model.req_rate / model.slo
            free_mem[best_idx] -= model.model_size
        return placement

    def max_kvpr(placement):
        worst = 0.0
        for g, ms in placement.items():
            free = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            if free <= 0:
                return float('inf')
            worst = max(worst, sum(m.req_rate / m.slo for m in ms) / free)
        return worst

    def refine(placement):
        """Move a model from the worst GPU elsewhere if it lowers max KVPR."""
        free_mem = [GPU_MEM_SIZE - sum(m.model_size for m in placement[g])
                    for g in range(gpu_num)]
        improved = True
        while improved:
            improved = False
            worst = max(placement, key=lambda g: max_kvpr({g: placement[g]}))
            old_max = max_kvpr(placement)
            for g in range(gpu_num):
                if g == worst:
                    continue
                for m in list(placement[worst]):
                    if m.model_size <= free_mem[g]:
                        placement[worst].remove(m)
                        placement[g].append(m)
                        if max_kvpr(placement) < old_max:
                            free_mem[worst] += m.model_size
                            free_mem[g] -= m.model_size
                            improved = True
                            break
                        placement[g].remove(m)
                        placement[worst].append(m)
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
        placement = greedy(order)
        if placement is None:
            continue
        placement = refine(placement)
        score = max_kvpr(placement)
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
