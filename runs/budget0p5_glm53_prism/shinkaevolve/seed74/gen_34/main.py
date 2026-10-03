GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement minimizing the maximum KVPR across all GPUs.
    Multi-start greedy seeds + move/swap local search with a full
    lexicographic (sorted KVPR vector) acceptance criterion.
    """
    if gpu_num <= 0 or not models:
        raise ValueError("gpu_num must be positive and models non-empty")

    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Models cannot fit into GPU memory")

    W = [m.req_rate / m.slo for m in models]
    S = [m.model_size for m in models]

    def greedy(order):
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            w = m.req_rate / m.slo
            best_g, best_r = None, float('inf')
            for g in range(gpu_num):
                free = rem[g] - m.model_size
                if free > 0:
                    r = (load[g] + w) / free
                    if r < best_r:
                        best_r, best_g = r, g
            if best_g is None:
                for g in range(gpu_num):
                    if m.model_size <= rem[g]:
                        best_g = g
                        break
            if best_g is None:
                return None
            placement[best_g].append(m)
            load[best_g] += w
            rem[best_g] -= m.model_size
        return placement

    def kvpr(load, free):
        if free <= 0:
            return float('inf') if load > 0 else 0.0
        return load / free

    def kvpr_vector(gload, gsize):
        return sorted((kvpr(gload[g], GPU_MEM_SIZE - gsize[g])
                       for g in range(gpu_num)), reverse=True)

    def stats(placement):
        gload = [0.0] * gpu_num
        gsize = [0.0] * gpu_num
        for g, ms in placement.items():
            for m in ms:
                gload[g] += m.req_rate / m.slo
                gsize[g] += m.model_size
        return gload, gsize

    def local_search(placement):
        best = {g: list(v) for g, v in placement.items()}
        gload, gsize = stats(best)
        best_vec = kvpr_vector(gload, gsize)

        for _ in range(3000):
            improved = False
            # single-model moves, accepted if new sorted vector is lexicographically smaller
            for mi, m in enumerate(models):
                w, s = W[mi], S[mi]
                src = next(g for g in range(gpu_num) if m in best[src] if False) if False else None
                src = next(g for g, ms in best.items() if m in ms)
                src_free = GPU_MEM_SIZE - gsize[src] + s
                src_load = gload[src] - w
                for g in range(gpu_num):
                    if g == src:
                        continue
                    if s <= GPU_MEM_SIZE - gsize[g]:
                        nl = list(gload)
                        ns = list(gsize)
                        nl[src] = src_load; ns[src] = gsize[src] - s
                        nl[g] = gload[g] + w; ns[g] = gsize[g] + s
                        vec = kvpr_vector(nl, ns)
                        if vec < best_vec:
                            best[src].remove(m)
                            best[g].append(m)
                            gload, gsize = nl, ns
                            best_vec = vec
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue
            # swap moves between GPUs
            placed = [(m, g) for g, ms in best.items() for m in ms]
            n = len(placed)
            for i in range(n):
                if improved:
                    break
                m1, g1 = placed[i]
                w1, s1 = m1.req_rate / m1.slo, m1.model_size
                for j in range(i + 1, n):
                    m2, g2 = placed[j]
                    if g1 == g2:
                        continue
                    w2, s2 = m2.req_rate / m2.slo, m2.model_size
                    if s2 - s1 <= GPU_MEM_SIZE - gsize[g1] + s1 and \
                       s1 - s2 <= GPU_MEM_SIZE - gsize[g2] + s2:
                        nl = list(gload)
                        ns = list(gsize)
                        nl[g1] += w2 - w1; ns[g1] += s2 - s1
                        nl[g2] += w1 - w2; ns[g2] += s1 - s2
                        vec = kvpr_vector(nl, ns)
                        if vec < best_vec:
                            best[g1].remove(m1); best[g1].append(m2)
                            best[g2].remove(m2); best[g2].append(m1)
                            gload, gsize = nl, ns
                            best_vec = vec
                            improved = True
                            break
            if not improved:
                break
        return best

    orderings = [
        sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: -(m.req_rate / m.slo) / max(m.model_size, 1e-12)),
        sorted(models, key=lambda m: m.model_size),
        sorted(models, key=lambda m: (m.req_rate / m.slo) / max(GPU_MEM_SIZE - m.model_size, 1e-9), reverse=True),
    ]

    best_placement, best_val = None, float('inf')
    for order in orderings:
        pl = greedy(order)
        if pl is None:
            raise ValueError("Unable to place all models in GPU memory")
        pl = local_search(pl)
        gload, gsize = stats(pl)
        val = max(kvpr(gload[g], GPU_MEM_SIZE - gsize[g]) for g in range(gpu_num))
        if val < best_val:
            best_val, best_placement = val, pl

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
