GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _greedy_place(gpu_num, ordered_models):
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted = [0.0 for _ in range(gpu_num)]

    for model in ordered_models:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            remaining = shared_kv[gpu_id] - model.model_size
            if remaining > 0:
                resulting = (weighted[gpu_id] + model.req_rate / model.slo) / remaining
                if resulting < best_ratio:
                    best_ratio = resulting
                    best_idx = gpu_id
        if best_idx is None:
            return None
        placement[best_idx].append(model)
        weighted[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size
    return placement


def _max_kvpr(placement):
    worst = 0.0
    for gpu_id, models in placement.items():
        if not models:
            continue
        load = sum(m.req_rate / m.slo for m in models)
        free = GPU_MEM_SIZE - sum(m.model_size for m in models)
        if free <= 0:
            return float('inf')
        worst = max(worst, load / free)
    return worst


def _local_search(placement, gpu_num, max_iters=200):
    best = _max_kvpr(placement)
    for _ in range(max_iters):
        improved = False
        # try moving each model from its GPU to another GPU
        items = [(m, gid) for gid in placement for m in placement[gid]]
        for model, src in items:
            if model not in placement[src]:
                continue
            for dst in range(gpu_num):
                if dst == src:
                    continue
                # check memory fit on dst
                used_dst = sum(m.model_size for m in placement[dst])
                if used_dst + model.model_size >= GPU_MEM_SIZE:
                    continue
                # simulate move
                placement[src].remove(model)
                placement[dst].append(model)
                new_score = _max_kvpr(placement)
                if new_score < best - 1e-12:
                    best = new_score
                    improved = True
                else:
                    placement[dst].remove(model)
                    placement[src].append(model)
            if improved:
                break
        if not improved:
            break
    return placement


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    candidates = []
    models = list(models)

    def try_order(order):
        p = _greedy_place(gpu_num, order)
        if p is not None:
            candidates.append(p)

    # deterministic orderings
    try_order(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True))
    try_order(sorted(models, key=lambda m: m.req_rate / m.slo))
    try_order(sorted(models, key=lambda m: -(m.req_rate / m.slo) / m.model_size))
    try_order(sorted(models, key=lambda m: -m.req_rate))
    try_order(sorted(models, key=lambda m: m.req_rate))
    try_order(sorted(models, key=lambda m: -m.model_size))
    try_order(sorted(models, key=lambda m: m.model_size))
    try_order(sorted(models, key=lambda m: m.slo))
    try_order(sorted(models, key=lambda m: -m.slo))

    # randomized orderings with fixed seed
    rng = random.Random(0)
    for _ in range(20):
        order = models[:]
        rng.shuffle(order)
        try_order(order)

    if not candidates:
        raise ValueError("Unable to place models on the given GPUs.")

    # pick the best candidate
    best_placement = min(candidates, key=_max_kvpr)

    # improve with local search (move-based)
    best_placement = _local_search(best_placement, gpu_num)

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