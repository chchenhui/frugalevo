GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses randomized multistart greedy construction followed by local search
    (moves and swaps), keeping the best placement found.
    """
    if not models or gpu_num <= 0:
        return {g: [] for g in range(gpu_num)}

    models = list(models)
    n = len(models)
    time_limit = 2.0
    start = time.time()

    best_placement = None
    best_score = float('inf')

    def kvpr_of(loads, free):
        # compute kvpr for all gpus
        vals = []
        for g in range(gpu_num):
            den = GPU_MEM_SIZE - free[g]
            if den <= 0:
                vals.append(float('inf'))
            else:
                vals.append(loads[g] / den)
        return vals

    attempt = 0
    while time.time() - start < time_limit:
        attempt += 1
        # randomized greedy construction
        order = models[:]
        if attempt == 1:
            order.sort(key=lambda m: m.req_rate / m.slo, reverse=True)
        else:
            random.shuffle(order)

        placement = {g: [] for g in range(gpu_num)}
        free = [float(GPU_MEM_SIZE)] * gpu_num
        loads = [0.0] * gpu_num
        feasible = True

        for m in order:
            w = m.req_rate / m.slo
            candidates = []
            for g in range(gpu_num):
                if m.model_size < free[g]:
                    new_ratio = (loads[g] + w) / (free[g] - m.model_size)
                    candidates.append((new_ratio, g))
            if not candidates:
                feasible = False
                break
            candidates.sort()
            # pick among top choices randomly (exploration)
            pick = candidates[0]
            if len(candidates) > 1 and attempt > 1 and random.random() < 0.3:
                pick = candidates[random.randrange(min(3, len(candidates)))]
            g = pick[1]
            placement[g].append(m)
            free[g] -= m.model_size
            loads[g] += w

        if not feasible:
            continue

        # local search: moves and swaps
        def full_score():
            kvs = kvpr_of(loads, free)
            return max(kvs)

        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            kvs = kvpr_of(loads, free)
            cur_max = max(kvs)
            # sort gpus by kvpr descending
            src = max(range(gpu_num), key=lambda g: kvs[g])
            # try moves from hottest gpu
            for m in placement[src]:
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or m.model_size >= free[dst]:
                        continue
                    src_den = free[src] + m.model_size
                    dst_den = free[dst] - m.model_size
                    if src_den <= 0 or dst_den <= 0:
                        continue
                    new_src = (loads[src] - w) / src_den
                    new_dst = (loads[dst] + w) / dst_den
                    others = max([kvs[i] for i in range(gpu_num) if i not in (src, dst)] or [0.0])
                    if max(new_src, new_dst, others) < cur_max - 1e-12:
                        placement[src].remove(m)
                        placement[dst].append(m)
                        free[src] += m.model_size
                        free[dst] -= m.model_size
                        loads[src] -= w
                        loads[dst] += w
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # try swaps between hottest and other gpus
            for m1 in placement[src]:
                w1 = m1.req_rate / m1.slo
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    for m2 in placement[dst]:
                        w2 = m2.req_rate / m2.slo
                        dfree = m2.model_size - m1.model_size
                        if free[dst] + dfree <= 0 or free[src] - dfuse_free if False else (free[src] - dfree) <= 0:
                            continue
                        new_src = (loads[src] - w1 + w2) / (free[src] - dfree)
                        new_dst = (loads[dst] - w2 + w1) / (free[dst] + dfree)
                        others = max([kvs[i] for i in range(gpu_num) if i not in (src, dst)] or [0.0])
                        if max(new_src, new_dst, others) < cur_max - 1e-12:
                            placement[src].remove(m1)
                            placement[dst].remove(m2)
                            placement[src].append(m2)
                            placement[dst].append(m1)
                            free[src] -= dfree
                            free[dst] += dfree
                            loads[src] += w2 - w1
                            loads[dst] += w1 - w2
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        score = full_score()
        if score < best_score:
            best_score = score
            best_placement = {g: list(placement[g]) for g in range(gpu_num)}

    if best_placement is None:
        raise ValueError("Unable to place all models within GPU memory limits.")

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