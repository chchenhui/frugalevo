GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement minimizing the maximum KVPR across all GPUs.
    Architecture: randomized-restart λ-threshold best-fit packing, each candidate
    refined by local search (moves/swaps off the hottest GPU); best kept.
    """

    def ratio(W, used_mem):
        free = GPU_MEM_SIZE - used_mem
        if free <= 1e-9:
            return float('inf')
        return W / free

    def pack_with_lam(order, lam):
        """Best-fit pack models (given order) under KVPR <= lam.
        Returns (placement, used, W) or None."""
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        W = [0.0] * gpu_num
        opened = 0
        for m in order:
            w = m.req_rate / m.slo
            s = m.model_size
            best_i, best_val = None, None
            for i in range(opened):
                if s <= GPU_MEM_SIZE - used[i] - 1e-12:
                    val = (W[i] + w) / (GPU_MEM_SIZE - used[i] - s)
                    if val <= lam + 1e-12 and (best_val is None or val < best_val):
                        best_val, best_i = val, i
            if best_i is None:
                # open new GPU
                if opened < gpu_num and s < GPU_MEM_SIZE - 1e-12:
                    val = w / (GPU_MEM_SIZE - s)
                    if val <= lam + 1e-12:
                        best_i, best_val = opened, val
                if best_i is None:
                    return None
                opened += 1
            placement[best_i].append(m)
            used[best_i] += s
            W[best_i] += w
        return placement, used, W

    def binary_search(order):
        """Find smallest feasible lam for this order; return candidate state or None."""
        lo, hi = 0.0, 1e9
        result = pack_with_lam(order, hi)
        if result is None:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            res = pack_with_lam(order, mid)
            if res is not None:
                result, hi = res, mid
            else:
                lo = mid
        return result

    def local_search(placement, used, W):
        """Greedy moves/swaps off the hottest GPU to reduce max KVPR."""
        def kvpr(i):
            return ratio(W[i], used[i])

        guard = 0
        improved = True
        while improved and guard < 3000:
            improved = False
            guard += 1
            max_i = max(range(gpu_num), key=kvpr)
            max_val = kvpr(max_i)
            if max_val == float('inf'):
                break
            # try moving each model off the hottest GPU
            for m in list(placement[max_i]):
                w, s = m.req_rate / m.slo, m.model_size
                for j in range(gpu_num):
                    if j == max_i:
                        continue
                    if s <= GPU_MEM_SIZE - used[j] - 1e-12:
                        n1 = ratio(W[max_i] - w, used[max_i] - s)
                        n2 = ratio(W[j] + w, used[j] + s)
                        others = max((kvpr(k) for k in range(gpu_num)
                                      if k not in (max_i, j)), default=0.0)
                        if max(n1, n2, others) < max_val - 1e-12:
                            placement[max_i].remove(m)
                            placement[j].append(m)
                            W[max_i] -= w; used[max_i] -= s
                            W[j] += w; used[j] += s
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # try swapping a model on the hottest GPU with one elsewhere
            for m1 in list(placement[max_i]):
                w1, s1 = m1.req_rate / m1.slo, m1.model_size
                for j in range(gpu_num):
                    if j == max_i:
                        continue
                    for m2 in list(placement[j]):
                        w2, s2 = m2.req_rate / m2.slo, m2.model_size
                        if (s1 - s2 <= GPU_MEM_SIZE - used[j] - 1e-12 and
                                s2 - s1 <= GPU_MEM_SIZE - used[max_i] - 1e-12):
                            n1 = ratio(W[max_i] - w1 + w2,
                                       used[max_i] - s1 + s2)
                            n2 = ratio(W[j] - w2 + w1, used[j] - s2 + s1)
                            others = max((kvpr(k) for k in range(gpu_num)
                                          if k not in (max_i, j)), default=0.0)
                            if max(n1, n2, others) < max_val - 1e-12:
                                placement[max_i].remove(m1)
                                placement[max_i].append(m2)
                                placement[j].remove(m2)
                                placement[j].append(m1)
                                W[max_i] += w2 - w1; used[max_i] += s2 - s1
                                W[j] += w1 - w2; used[j] += s1 - s2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return placement, used, W

    def score(used, W):
        vals = [ratio(W[i], used[i]) for i in range(gpu_num)]
        vals = [v for v in vals if v != float('inf')]
        return max(vals) if vals else float('inf')

    def fallback(order):
        """Memory-only best-fit packing (no KVPR constraint)."""
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        W = [0.0] * gpu_num
        for m in order:
            w, s = m.req_rate / m.slo, m.model_size
            best_i, best_val = None, None
            for i in range(gpu_num):
                if s <= GPU_MEM_SIZE - used[i] - 1e-12:
                    val = ratio(W[i] + w, used[i] + s)
                    if best_val is None or val < best_val:
                        best_val, best_i = val, i
            if best_i is None:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU."
                )
            placement[best_i].append(m)
            used[best_i] += s
            W[best_i] += w
        return placement, used, W

    # --- main loop: randomized restarts ---
    base_order = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    rng = random.Random(1234)

    best_state, best_score_val = None, float('inf')
    for restart in range(30):
        order = list(base_order)
        if restart > 0:
            # shuffle within neighborhoods of the density-sorted order
            for k in range(0, len(order), 3):
                seg = order[k:k + 3]
                rng.shuffle(seg)
                order[k:k + 3] = seg
        state = binary_search(order)
        if state is None:
            continue
        state = local_search(*state)
        sc = score(state[1], state[2])
        if sc < best_score_val:
            best_score_val, best_state = sc, state

    if best_state is None:
        best_state = fallback(base_order)
        best_state = local_search(*best_state)

    return best_state[0]

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
