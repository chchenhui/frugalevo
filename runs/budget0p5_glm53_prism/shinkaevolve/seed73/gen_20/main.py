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

    import random

    def build_placement(order):
        # Greedy KVPR-minimizing placement followed by local search,
        # for a fixed input ordering of models.
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]

        for model in order:
            best_idx = None
            best_ratio = float('inf')

            for gpu_id in range(gpu_num):
                if model.model_size < shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                    new_w = weighted_req_rate[gpu_id] + model.req_rate / model.slo
                    new_mem = shared_kv[gpu_id] - model.model_size
                    new_ratio = new_w / new_mem
                    if new_ratio < best_ratio or (
                        new_ratio == best_ratio and best_idx is not None
                        and shared_kv[gpu_id] < shared_kv[best_idx]
                    ):
                        best_ratio = new_ratio
                        best_idx = gpu_id

            if best_idx is None:
                return None

            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += model.req_rate / model.slo
            shared_kv[best_idx] -= model.model_size

        # Local search: moves and swaps to reduce the maximum KVPR
    def kvpr_of(i):
        free = shared_kv[i]
        if free <= 0:
            return float('inf') if weighted_req_rate[i] > 0 else 0.0
        return weighted_req_rate[i] / free

    for _ in range(200):  # bounded passes to guarantee termination
        max_gpu = max(range(gpu_num), key=lambda i: (kvpr_of(i), -shared_kv[i]))
        max_kvpr = kvpr_of(max_gpu)
        improved = False

        # Try moving a model off the worst GPU
        for m in list(placement[max_gpu]):
            for g in range(gpu_num):
                if g == max_gpu or m.model_size > shared_kv[g]:
                    continue
                # simulate move
                weighted_req_rate[max_gpu] -= m.req_rate / m.slo
                shared_kv[max_gpu] += m.model_size
                weighted_req_rate[g] += m.req_rate / m.slo
                shared_kv[g] -= m.model_size
                new_max = max(kvpr_of(x) for x in range(gpu_num))
                if new_max < max_kvpr - 1e-12:
                    placement[max_gpu].remove(m)
                    placement[g].append(m)
                    improved = True
                    break
                # revert
                weighted_req_rate[max_gpu] += m.req_rate / m.slo
                shared_kv[max_gpu] -= m.model_size
                weighted_req_rate[g] -= m.req_rate / m.slo
                shared_kv[g] += m.model_size
            if improved:
                break
        if improved:
            continue

        # Try swapping a model on the worst GPU with a model elsewhere
        for m1 in list(placement[max_gpu]):
            for g in range(gpu_num):
                if g == max_gpu:
                    continue
                for m2 in list(placement[g]):
                    # check sizes fit after swap
                    if (m1.model_size - m2.model_size) > shared_kv[g]:
                        continue
                    if (m2.model_size - m1.model_size) > shared_kv[max_gpu]:
                        continue
                    # simulate swap
                    weighted_req_rate[max_gpu] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                    shared_kv[max_gpu] += m1.model_size - m2.model_size
                    weighted_req_rate[g] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
                    shared_kv[g] += m2.model_size - m1.model_size
                    new_max = max(kvpr_of(x) for x in range(gpu_num))
                    if new_max < max_kvpr - 1e-12:
                        placement[max_gpu].remove(m1)
                        placement[g].append(m1)
                        placement[g].remove(m2)
                        placement[max_gpu].append(m2)
                        improved = True
                        break
                    # revert
                    weighted_req_rate[max_gpu] -= m2.req_rate / m2.slo - m1.req_rate / m1.slo
                    shared_kv[max_gpu] -= m1.model_size - m2.model_size
                    weighted_req_rate[g] -= m1.req_rate / m1.slo - m2.req_rate / m2.slo
                    shared_kv[g] -= m2.model_size - m1.model_size
                if improved:
                    break
            if improved:
                break
            if not improved:
                break

        return placement

    def max_kvpr_of(placement):
        if placement is None:
            return float('inf')
        worst = 0.0
        for gpu_id in range(gpu_num):
            free = GPU_MEM_SIZE - sum(m.model_size for m in placement[gpu_id])
            w = sum(m.req_rate / m.slo for m in placement[gpu_id])
            if free <= 0:
                return float('inf') if w > 0 else 0.0
            worst = max(worst, w / free)
        return worst

    # Enumerate candidate orderings and keep the best placement
    candidates = []
    candidates.append(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    candidates.append(sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size))
    candidates.append(sorted(models, key=lambda m: m.model_size, reverse=True))
    candidates.append(sorted(models, key=lambda m: m.req_rate, reverse=True))
    candidates.append(sorted(models, key=lambda m: m.slo))
    rng = random.Random(42)
    for _ in range(5):
        shuffled = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
        rng.shuffle(shuffled)
        candidates.append(shuffled)

    best_placement = None
    best_score = float('inf')
    for order in candidates:
        p = build_placement(order)
        s = max_kvpr_of(p)
        if s < best_score:
            best_score = s
            best_placement = p

    if best_placement is None:
        raise ValueError("Unable to place all models on the available GPUs.")

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