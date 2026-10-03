GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy best-fit placement (models sorted by req_rate/slo descending)
    followed by move/swap hill climbing on (max KVPR, sum KVPR) over all
    GPU pairs, with a few randomized-restart perturbations to escape
    local optima. Keeps per-GPU W (sum r/s) and S (sum sizes) cached.
    """

    def kvpr(W, S):
        return W / (GPU_MEM_SIZE - S) if S < GPU_MEM_SIZE else 0.0

    M = GPU_MEM_SIZE
    idx = {id(m): i for i, m in enumerate(models)}
    v = [m.req_rate / m.slo for m in models]
    s = [m.model_size for m in models]

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        W = [0.0] * gpu_num
        S = [0.0] * gpu_num
        for j in order:
            r, sz = v[j], s[j]
            bg, bv = None, float('inf')
            for g in range(gpu_num):
                if S[g] + sz < M:
                    val = (W[g] + r) / (M - S[g] - sz)
                    if val < bv:
                        bv, bg = val, g
            if bg is None:
                for g in range(gpu_num):
                    if S[g] == 0 and sz == M:
                        bg = g
                        break
            if bg is None:
                return None
            placement[bg].append(models[j])
            W[bg] += r
            S[bg] += sz
        return placement, W, S

    def repair(placement, W, S):
        def score():
            k = [kvpr(W[g], S[g]) for g in range(gpu_num)]
            return (max(k), sum(k))

        for _ in range(300):
            cur = score()
            best = None  # (new_score, action)
            for a in range(gpu_num):
                for i, m1 in enumerate(placement[a]):
                    r1, s1 = v[idx[id(m1)]], s[idx[id(m1)]]
                    for b in range(gpu_num):
                        if b == a:
                            continue
                        # move a -> b
                        if S[b] + s1 < M:
                            k = [kvpr(W[h], S[h]) for h in range(gpu_num)]
                            k[a] = kvpr(W[a] - r1, S[a] - s1)
                            k[b] = kvpr(W[b] + r1, S[b] + s1)
                            sc = (max(k), sum(k))
                            if best is None or sc < best[0]:
                                best = (sc, ('move', a, i, b))
                        # swap m1 <-> m2
                        for j, m2 in enumerate(placement[b]):
                            r2, s2 = v[idx[id(m2)]], s[idx[id(m2)]]
                            if S[a] - s1 + s2 >= M or S[b] - s2 + s1 >= M:
                                continue
                            k = [kvpr(W[h], S[h]) for h in range(gpu_num)]
                            k[a] = kvpr(W[a] - r1 + r2, S[a] - s1 + s2)
                            k[b] = kvpr(W[b] - r2 + r1, S[b] - s2 + s1)
                            sc = (max(k), sum(k))
                            if best is None or sc < best[0]:
                                best = (sc, ('swap', a, i, b, j))
            if best is None or best[0] >= cur:
                break
            act = best[1]
            if act[0] == 'move':
                _, a, i, b = act
                m = placement[a].pop(i)
                W[a] -= v[idx[id(m)]]
                S[a] -= s[idx[id(m)]]
                placement[b].append(m)
                W[b] += v[idx[id(m)]]
                S[b] += s[idx[id(m)]]
            else:
                _, a, i, b, j = act
                m1 = placement[a].pop(i)
                m2 = placement[b].pop(j)
                i1, i2 = idx[id(m1)], idx[id(m2)]
                W[a] += v[i2] - v[i1]
                S[a] += s[i2] - s[i1]
                W[b] += v[i1] - v[i2]
                S[b] += s[i1] - s[i2]
                placement[a].append(m2)
                placement[b].append(m1)
        return placement, W, S

    order = sorted(range(len(models)), key=lambda j: v[j], reverse=True)
    res = greedy(order)
    if res is None:
        raise ValueError("Unable to place model")
    placement, W, S = repair(*res)
    best = placement
    best_score = max(kvpr(W[g], S[g]) for g in range(gpu_num))

    # Randomized restarts: perturb order, re-greedy, re-repair, keep best.
    import random
    for _ in range(8):
        perm = order[:]
        random.shuffle(perm)
        res = greedy(perm)
        if res is None:
            continue
        p2, W2, S2 = repair(*res)
        sc = max(kvpr(W2[g], S2[g]) for g in range(gpu_num))
        if sc < best_score:
            best_score, best = sc, p2

    return best

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
