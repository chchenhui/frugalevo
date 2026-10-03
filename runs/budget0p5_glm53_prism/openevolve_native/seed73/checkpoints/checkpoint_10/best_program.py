GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Greedy placement: for each model, choose the GPU where the resulting
    # KVPR (after adding the model) is smallest, among GPUs that fit it.
    # Try several model orderings (deterministic heuristics plus randomized
    # perturbations within a time budget), refine each with a
    # move/swap local search, and keep the best feasible placement.

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num  # sum of req_rate/slo per GPU
        for m in order:
            best, best_kvpr = None, float('inf')
            for g in range(gpu_num):
                rem = free_mem[g] - m.model_size
                if rem <= 0:
                    continue
                kvpr = (load[g] + m.req_rate / m.slo) / rem
                if kvpr < best_kvpr:
                    best_kvpr, best = kvpr, g
            if best is None:
                return None
            placement[best].append(m)
            load[best] += m.req_rate / m.slo
            free_mem[best] -= m.model_size
        return placement

    def max_kvpr(placement):
        worst = 0.0
        for g, ms in placement.items():
            if not ms:
                continue
            load = sum(m.req_rate / m.slo for m in ms)
            free = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            if free <= 0:
                return float('inf')
            worst = max(worst, load / free)
        return worst

    def local_search(placement):
        # Repeatedly move a single model to another GPU, or swap a pair of
        # models between two GPUs, whenever it lowers the max KVPR.
        improved = True
        while improved:
            improved = False
            worst = max_kvpr(placement)
            # Try single-model moves first
            for src in range(gpu_num):
                for m in list(placement[src]):
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if GPU_MEM_SIZE - sum(x.model_size for x in placement[dst]) - m.model_size <= 0:
                            continue
                        placement[src].remove(m)
                        placement[dst].append(m)
                        if max_kvpr(placement) < worst:
                            improved = True
                            break
                        placement[dst].remove(m)
                        placement[src].append(m)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # Try pairwise swaps between GPUs
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for ma in list(placement[a]):
                        for mb in list(placement[b]):
                            placement[a].remove(ma)
                            placement[b].remove(mb)
                            placement[a].append(mb)
                            placement[b].append(ma)
                            if max_kvpr(placement) < worst:
                                improved = True
                                break
                            placement[a].remove(mb)
                            placement[b].remove(ma)
                            placement[a].append(ma)
                            placement[b].append(mb)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return placement

    import time, random

    orders = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: m.req_rate / m.slo),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9), reverse=True),
        list(models),
    ]

    best_result, best_score = None, float('inf')
    for order in orders:
        result = greedy(order)
        if result is None:
            continue
        result = local_search(result)
        score = max_kvpr(result)
        if score < best_score:
            best_score, best_result = score, result

    # Randomized multi-start: shuffle a load-weighted ordering many times
    # within a time budget; each greedy result is locally refined.
    deadline = time.time() + 1.5
    rng = random.Random(0)
    base = sorted(models, key=lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9), reverse=True)
    while time.time() < deadline and best_score > 1e-9:
        order = base[:]
        # random perturbation: shuffle a small window of the ordering
        for _ in range(max(1, len(order) // 4)):
            i = rng.randrange(len(order))
            j = rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
        result = greedy(order)
        if result is None:
            continue
        result = local_search(result)
        score = max_kvpr(result)
        if score < best_score:
            best_score, best_result = score, result

    if best_result is None:
        raise ValueError("Unable to place all models within GPU memory.")

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
