GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use multi-start greedy packing followed by feasible simulated annealing."""
    import math
    import random

    result = {i: [] for i in range(gpu_num)}
    if not models:
        return result

    rng = random.Random(17)
    pressure = {id(m): m.req_rate / m.slo for m in models}

    def score(free, load):
        return max(
            load[i] / free[i] if free[i] > 0 else float("inf")
            for i in range(gpu_num)
        )

    best_score = float("inf")
    best_groups = None
    best_free = None
    best_load = None

    # Different orderings create substantially different feasible packings.
    orders = [
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: pressure[id(m)], reverse=True),
        sorted(models, key=lambda m: pressure[id(m)] / m.model_size, reverse=True),
    ]
    max_size = max(m.model_size for m in models)
    max_pressure = max(pressure[id(m)] for m in models)
    for _ in range(24):
        weight = rng.random()
        orders.append(sorted(
            models,
            key=lambda m: weight * m.model_size / max_size +
            (1 - weight) * pressure[id(m)] / max_pressure,
            reverse=True,
        ))

    for order in orders:
        groups = [[] for _ in range(gpu_num)]
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        valid = True
        for model in order:
            size, demand = model.model_size, pressure[id(model)]
            choices = [i for i in range(gpu_num) if size <= free[i]]
            if not choices:
                valid = False
                break
            gpu = min(
                choices,
                key=lambda i: max(
                    (load[j] + demand) / (free[j] - size)
                    if j == i and free[j] > size else
                    float("inf") if j == i else
                    load[j] / free[j] if free[j] > 0 else float("inf")
                    for j in range(gpu_num)
                ),
            )
            groups[gpu].append(model)
            free[gpu] -= size
            load[gpu] += demand
        if valid and score(free, load) < best_score:
            best_score = score(free, load)
            best_groups, best_free, best_load = groups, free, load

    if best_groups is None:
        raise ValueError("Models do not fit in available GPU memory")

    groups = [list(group) for group in best_groups]
    free, load = list(best_free), list(best_load)
    current = best_score
    temperature = max(current * 0.15, 0.01)
    steps = max(400, 80 * len(models))

    for step in range(steps):
        occupied = [i for i in range(gpu_num) if groups[i]]
        if not occupied:
            break
        src = rng.choice(occupied)
        dst = rng.randrange(gpu_num)
        if src == dst:
            continue

        left = rng.choice(groups[src])
        ls, lp = left.model_size, pressure[id(left)]

        # Alternate between relocations and swaps.
        right = rng.choice(groups[dst]) if groups[dst] and rng.random() < 0.5 else None
        rs = right.model_size if right else 0
        rp = pressure[id(right)] if right else 0

        new_src_free = free[src] + ls - rs
        new_dst_free = free[dst] + rs - ls
        if new_src_free < 0 or new_dst_free < 0:
            continue

        old_free_src, old_free_dst = free[src], free[dst]
        old_load_src, old_load_dst = load[src], load[dst]
        free[src], free[dst] = new_src_free, new_dst_free
        load[src], load[dst] = load[src] - lp + rp, load[dst] - rp + lp
        candidate = score(free, load)
        temp = temperature * (1 - step / steps) + 1e-9

        if candidate <= current or rng.random() < math.exp((current - candidate) / temp):
            groups[src].remove(left)
            groups[dst].append(left)
            if right:
                groups[dst].remove(right)
                groups[src].append(right)
            current = candidate
            if candidate < best_score:
                best_score = candidate
                best_groups = [list(group) for group in groups]
        else:
            free[src], free[dst] = old_free_src, old_free_dst
            load[src], load[dst] = old_load_src, old_load_dst

    return {i: best_groups[i] for i in range(gpu_num)}

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
