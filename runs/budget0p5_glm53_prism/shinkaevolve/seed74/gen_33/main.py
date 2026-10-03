GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

import random

def compute_model_placement(gpu_num, models):
    """
    Iterated local search: greedy randomized multistart placements refined by
    lexicographic-leximin local search (moves + swaps between any GPU pair),
    plus perturbation-and-restart from the incumbent.
    """

    def ratio(W, used):
        free = GPU_MEM_SIZE - used
        if free <= 1e-9:
            return float('inf')
        return W / free

    def refine(placement, used, W):
        def kvs():
            return [ratio(W[i], used[i]) for i in range(gpu_num)]

        guard = 0
        improved = True
        while improved and guard < 3000:
            improved = False
            guard += 1
            cur = kvs()
            top = sorted(cur, reverse=True)
            n = len(models)

            # --- moves: any source, any destination ---
            for src in range(gpu_num):
                if improved:
                    break
                for m in list(placement[src]):
                    if improved:
                        break
                    w = m.req_rate / m.slo
                    for j in range(gpu_num):
                        if j == src:
                            continue
                        if m.model_size <= GPU_MEM_SIZE - used[j] + 1e-9:
                            nsrc = ratio(W[src] - w, used[src] - m.model_size)
                            ndst = ratio(W[j] + w, used[j] + m.model_size)
                            nl = list(cur)
                            nl[src] = nsrc
                            nl[j] = ndst
                            if sorted(nl, reverse=True) < top:
                                placement[src].remove(m)
                                placement[j].append(m)
                                W[src] -= w; used[src] -= m.model_size
                                W[j] += w; used[j] += m.model_size
                                improved = True
                                break

            if improved:
                continue

            # --- swaps: any pair of GPUs ---
            for a in range(gpu_num):
                if improved:
                    break
                for b in range(a + 1, gpu_num):
                    if improved:
                        break
                    for m in list(placement[a]):
                        if improved:
                            break
                        for m2 in list(placement[b]):
                            nu1 = used[a] - m.model_size + m2.model_size
                            nu2 = used[b] - m2.model_size + m.model_size
                            if nu1 <= GPU_MEM_SIZE + 1e-9 and nu2 <= GPU_MEM_SIZE + 1e-9:
                                nW1 = W[a] - m.req_rate / m.slo + m2.req_rate / m2.slo
                                nW2 = W[b] - m2.req_rate / m2.slo + m.req_rate / m.slo
                                nl = list(cur)
                                nl[a] = ratio(nW1, nu1)
                                nl[b] = ratio(nW2, nu2)
                                if sorted(nl, reverse=True) < top:
                                    placement[a].remove(m)
                                    placement[b].remove(m2)
                                    placement[a].append(m2)
                                    placement[b].append(m)
                                    W[a], W[b] = nW1, nW2
                                    used[a], used[b] = nu1, nu2
                                    improved = True
                                    break

        return placement, used, W

    def attempt(order):
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        W = [0.0] * gpu_num
        for m in order:
            best_val = float('inf')
            cands = []
            for i in range(gpu_num):
                if m.model_size <= GPU_MEM_SIZE - used[i] + 1e-9:
                    v = ratio(W[i] + m.req_rate / m.slo, used[i] + m.model_size)
                    if v < best_val - 1e-12:
                        best_val = v
                        cands = [i]
                    elif v < best_val + 1e-12:
                        cands.append(i)
            if not cands:
                return None
            i = random.choice(cands)
            placement[i].append(m)
            used[i] += m.model_size
            W[i] += m.req_rate / m.slo
        return refine(placement, used, W)

    def score(placement, used, W):
        return max(ratio(W[i], used[i]) for i in range(gpu_num))

    base = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    best = None
    best_used = best_W = None
    best_val = float('inf')

    # --- multistart greedy + refine ---
    for it in range(20):
        if it == 0:
            order = list(base)
        elif it % 2 == 1:
            order = list(models)
            random.shuffle(order)
        else:
            order = sorted(models,
                           key=lambda m: (m.req_rate / m.slo) * random.uniform(0.8, 1.2),
                           reverse=True)
        res = attempt(order)
        if res is None:
            continue
        p, u, w = res
        v = score(p, u, w)
        if v < best_val - 1e-15:
            best_val, best, best_used, best_W = v, p, u, w

    if best is None:
        res = attempt(list(base))
        if res is None:
            raise ValueError("No feasible placement found.")
        return res[0]

    # --- iterated local search: perturb incumbent and re-refine ---
    for it in range(20):
        p = {i: list(best[i]) for i in range(gpu_num)}
        u = list(best_used)
        w = list(best_W)
        # perturb: move 1-3 random models to random feasible GPUs
        ok = True
        for _ in range(random.randint(1, 3)):
            srcs = [i for i in range(gpu_num) if p[i]]
            if not srcs:
                break
            s = random.choice(srcs)
            m = random.choice(p[s])
            cands = [j for j in range(gpu_num)
                     if m.model_size <= GPU_MEM_SIZE - u[j] + 1e-9]
            if not cands:
                ok = False
                break
            j = random.choice(cands)
            p[s].remove(m)
            p[j].append(m)
            u[s] -= m.model_size
            u[j] += m.model_size
            wm = m.req_rate / m.slo
            w[s] -= wm
            w[j] += wm
        if not ok:
            continue
        p, u, w = refine(p, u, w)
        v = score(p, u, w)
        if v < best_val - 1e-15:
            best_val, best, best_used, best_W = v, p, u, w

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
