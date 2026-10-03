GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily minimize each model's projected KVPR, then apply the best relocation or swap."""
    placement = {g: [] for g in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        demand, size = model.req_rate / model.slo, model.model_size
        gpu = min(
            (g for g in range(gpu_num) if size <= remaining[g]),
            key=lambda g: (load[g] + demand) / max(remaining[g] - size, 1e-12),
            default=None,
        )
        if gpu is None:
            raise ValueError(f"Unable to place model of size {size} GB.")
        placement[gpu].append(model)
        load[gpu] += demand
        remaining[gpu] -= size

    best_score = max(load[g] / max(remaining[g], 1e-12) for g in range(gpu_num))
    best_action = None

    for source in range(gpu_num):
        for model in placement[source]:
            demand, size = model.req_rate / model.slo, model.model_size
            for target in range(gpu_num):
                if source == target or size > remaining[target]:
                    continue
                score = max(
                    (load[g] - (demand if g == source else 0) + (demand if g == target else 0))
                    / max(remaining[g] + (size if g == source else 0) - (size if g == target else 0), 1e-12)
                    for g in range(gpu_num)
                )
                if score < best_score:
                    best_score, best_action = score, ("move", source, target, model)

    for left in range(gpu_num):
        for right in range(left + 1, gpu_num):
            for a in placement[left]:
                da, sa = a.req_rate / a.slo, a.model_size
                for b in placement[right]:
                    db, sb = b.req_rate / b.slo, b.model_size
                    if sb > remaining[left] + sa or sa > remaining[right] + sb:
                        continue
                    score = max(
                        (load[g] + (db - da if g == left else da - db if g == right else 0))
                        / max(remaining[g] + (sa - sb if g == left else sb - sa if g == right else 0), 1e-12)
                        for g in range(gpu_num)
                    )
                    if score < best_score:
                        best_score, best_action = score, ("swap", left, right, a, b)

    if best_action:
        if best_action[0] == "move":
            _, source, target, model = best_action
            placement[source].remove(model)
            placement[target].append(model)
        else:
            _, left, right, a, b = best_action
            placement[left].remove(a)
            placement[right].remove(b)
            placement[left].append(b)
            placement[right].append(a)

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
