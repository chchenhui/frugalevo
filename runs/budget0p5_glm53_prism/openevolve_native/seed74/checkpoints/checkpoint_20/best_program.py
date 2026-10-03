GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-seed greedy + local search.

    Try several greedy orderings of the models. For each ordering, place
    every model on the GPU that yields the smallest resulting KVPR while
    fitting in memory. Then run a local search (single moves + swaps off
    the worst GPU) to lower the max KVPR. Return the best placement found.
    """
    def kvpr(assign):
        loads = []
        for g in range(gpu_num):
            w = sum(m.req_rate / m.slo for m in assign[g])
            mem = GPU_MEM_SIZE - sum(m.model_size for m in assign[g])
            loads.append(w / mem if mem > 0 else float('inf'))
        return loads

    def greedy(order, criterion='ratio'):
        placement = {g: [] for g in range(gpu_num)}
        weighted = [0.0] * gpu_num
        free = [GPU_MEM_SIZE] * gpu_num
        for model in order:
            w = model.req_rate / model.slo
            best_idx, best_score = None, float('inf')
            for g in range(gpu_num):
                rem = free[g] - model.model_size
                if rem > 0:
                    if criterion == 'ratio':
                        score = (weighted[g] + w) / rem
                    else:  # minimize KVPR increase
                        cur = weighted[g] / (rem + model.model_size)
                        new = (weighted[g] + w) / rem
                        score = new - cur
                    if score < best_score:
                        best_score, best_idx = score, g
            if best_idx is None:
                return None
            placement[best_idx].append(model)
            weighted[best_idx] += w
            free[best_idx] -= model.model_size
        return placement, free

    def local_search(placement, free):
        improved = True
        while improved:
            improved = False
            loads = kvpr(placement)
            old_max = max(loads)
            worst = max(range(gpu_num), key=lambda g: loads[g])
            # single moves from worst GPU
            for m in list(placement[worst]):
                for g in range(gpu_num):
                    if g != worst and free[g] >= m.model_size:
                        placement[worst].remove(m)
                        placement[g].append(m)
                        if max(kvpr(placement)) < old_max:
                            free[worst] += m.model_size
                            free[g] -= m.model_size
                            improved = True
                            break
                        placement[g].remove(m)
                        placement[worst].append(m)
                if improved:
                    break
            if improved:
                continue
            # swaps between worst GPU and others
            for m in list(placement[worst]):
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    for m2 in list(placement[g]):
                        if free[worst] + m.model_size - m2.model_size >= 0 and \
                           free[g] + m2.model_size - m.model_size >= 0:
                            placement[worst].remove(m)
                            placement[g].remove(m2)
                            placement[worst].append(m2)
                            placement[g].append(m)
                            if max(kvpr(placement)) < old_max:
                                free[worst] += m.model_size - m2.model_size
                                free[g] += m2.model_size - m.model_size
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
        return placement

    seeds = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size),
        sorted(models, key=lambda m: m.model_size),
        list(models),
    ]
    # deterministic pseudo-random shuffles for extra diversity
    import random
    rng = random.Random(42)
    for _ in range(8):
        order = list(models)
        rng.shuffle(order)
        seeds.append(order)

    best_placement, best_max = None, float('inf')
    for order in seeds:
        for criterion in ('ratio', 'delta'):
            res = greedy(order, criterion)
            if res is None:
                continue
            placement, free = res
            placement = local_search(placement, free)
            cur = max(kvpr(placement))
            if cur < best_max:
                best_max, best_placement = cur, placement
    if best_placement is None:
        raise ValueError("Cannot fit models on GPUs.")
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
