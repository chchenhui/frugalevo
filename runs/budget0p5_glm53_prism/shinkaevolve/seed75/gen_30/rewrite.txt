GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

import random


def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    def kvpr_of(load, free):
        if free <= 0:
            return float('inf')
        return load / free

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    def run_pipeline(seed, epsilon):
        rng = random.Random(seed)

        # 1) Sort models by r_j / s_j descending
        sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        # 2) Greedy placement with randomized tie-breaking within epsilon of best
        for model in sorted_models:
            w = model.req_rate / model.slo
            best_kvpr = float('inf')
            cands = []
            for g in range(gpu_num):
                rem = free_mem[g] - model.model_size
                if rem <= 0:
                    continue
                new_kvpr = kvpr_of(load[g] + w, rem)
                if new_kvpr < best_kvpr:
                    best_kvpr = new_kvpr
                cands.append((new_kvpr, g))
            if not cands:
                return None, float('inf')
            near = [g for kv, g in cands if kv <= best_kvpr * (1 + epsilon) + 1e-15]
            chosen = rng.choice(near) if seed != 0 else near[0]
            placement[chosen].append(model)
            free_mem[chosen] -= model.model_size
            load[chosen] += w

        # 3) Local search: moves then swaps from worst GPU
        def kvpr(g):
            return kvpr_of(load[g], free_mem[g])

        for _ in range(60):
            kvprs = [kvpr(g) for g in range(gpu_num)]
            worst = max(range(gpu_num), key=lambda g: kvprs[g])
            worst_kvpr = kvprs[worst]
            best_gain, best_move = 0.0, None
            for model in placement[worst]:
                w = model.req_rate / model.slo
                src_rem = free_mem[worst] + model.model_size
                src_kvpr = kvpr_of(load[worst] - w, src_rem)
                for g in range(gpu_num):
                    if g == worst or model.model_size > free_mem[g]:
                        continue
                    dst_kvpr = kvpr_of(load[g] + w, free_mem[g] - model.model_size)
                    others = [kvprs[k] for k in range(gpu_num) if k not in (worst, g)]
                    new_max = max(others + [src_kvpr, dst_kvpr]) if others else max(src_kvpr, dst_kvpr)
                    gain = worst_kvpr - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (model, g)
            if best_move is not None:
                model, dst = best_move
                w = model.req_rate / model.slo
                placement[worst].remove(model)
                placement[dst].append(model)
                load[worst] -= w
                free_mem[worst] += model.model_size
                load[dst] += w
                free_mem[dst] -= model.model_size
                continue

            # swap phase
            best_gain, best_swap = 0.0, None
            for model in list(placement[worst]):
                w_a = model.req_rate / model.slo
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    for other in list(placement[g]):
                        w_b = other.req_rate / other.slo
                        if model.model_size - other.model_size > free_mem[g]:
                            continue
                        if other.model_size - model.model_size > free_mem[worst]:
                            continue
                        src_rem = free_mem[worst] + model.model_size - other.model_size
                        dst_rem = free_mem[g] + other.model_size - model.model_size
                        if src_rem <= 0 or dst_rem <= 0:
                            continue
                        src_kvpr = kvpr_of(load[worst] - w_a + w_b, src_rem)
                        dst_kvpr = kvpr_of(load[g] - w_b + w_a, dst_rem)
                        others = [kvprs[k] for k in range(gpu_num) if k not in (worst, g)]
                        new_max = max(others + [src_kvpr, dst_kvpr]) if others else max(src_kvpr, dst_kvpr)
                        gain = worst_kvpr - new_max
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (model, other, g)
            if best_swap is not None:
                model, other, g = best_swap
                placement[worst].remove(model)
                placement[worst].append(other)
                placement[g].remove(other)
                placement[g].append(model)
                load[worst] += other.req_rate / other.slo - model.req_rate / model.slo
                load[g] += model.req_rate / model.slo - other.req_rate / other.slo
                free_mem[worst] += model.model_size - other.model_size
                free_mem[g] += other.model_size - model.model_size
                continue
            break

        kvprs = [kvpr(g) for g in range(gpu_num)]
        return placement, max(kvprs)

    best_placement, best_max = None, float('inf')
    for seed in range(12):
        epsilon = 0.0 if seed == 0 else 0.05
        placement, mx = run_pipeline(seed, epsilon)
        if placement is not None and mx < best_max:
            best_max, best_placement = mx, placement

    if best_placement is None:
        raise ValueError(
            f"Unable to place models on {gpu_num} GPUs of {GPU_MEM_SIZE} GB each."
        )

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