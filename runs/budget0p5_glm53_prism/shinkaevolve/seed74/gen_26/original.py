GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Multi-start greedy with move+swap local search refinement.
    """
    if gpu_num <= 0 or not models:
        raise ValueError("gpu_num must be positive and models non-empty")

    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Models cannot fit into GPU memory")

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
                # fallback: allow exact fill
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

    def kvpr_of(load, free):
        if free <= 0:
            return float('inf') if load > 0 else 0.0
        return load / free

    def max_kvpr(placement):
        worst = 0.0
        for g, ms in placement.items():
            free = GPU_MEM_SIZE - sum(m.model_size for m in ms)
            load = sum(m.req_rate / m.slo for m in ms)
            if free <= 0:
                if load > 0:
                    return float('inf')
                continue
            worst = max(worst, load / free)
        return worst

    def local_search(placement):
        best = {g: list(v) for g, v in placement.items()}
        best_val = max_kvpr(best)
        # track per-GPU load and size incrementally
        gload = {g: sum(m.req_rate / m.slo for m in best[g]) for g in range(gpu_num)}
        gsize = {g: sum(m.model_size for m in best[g]) for g in range(gpu_num)}

        for _ in range(30):
            improved = False
            # single-model moves
            for m in models:
                src = next(g for g, ms in best.items() if m in ms)
                w = m.req_rate / m.slo
                src_free_after = GPU_MEM_SIZE - (gsize[src] - m.model_size)
                src_load_after = gload[src] - w
                for g in range(gpu_num):
                    if g == src:
                        continue
                    if m.model_size <= GPU_MEM_SIZE - gsize[g]:
                        cand = kvpr_of(src_load_after, src_free_after)
                        cand = max(cand, kvpr_of(gload[g] + w, GPU_MEM_SIZE - gsize[g] - m.model_size))
                        if cand < best_val - 1e-15:
                            best[src].remove(m)
                            best[g].append(m)
                            gload[src] -= w
                            gload[g] += w
                            gsize[src] -= m.model_size
                            gsize[g] += m.model_size
                            best_val = max_kvpr(best)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # swap moves between GPUs
                placed = [(m, g) for g, ms in best.items() for m in ms]
                for i in range(len(placed)):
                    if improved:
                        break
                    m1, g1 = placed[i]
                    for j in range(i + 1, len(placed)):
                        m2, g2 = placed[j]
                        if g1 == g2:
                            continue
                        if m1.model_size - m2.model_size <= GPU_MEM_SIZE - gsize[g2] + m2.model_size and \
                           m2.model_size - m1.model_size <= GPU_MEM_SIZE - gsize[g1] + m1.model_size:
                            cand = {k: list(v) for k, v in best.items()}
                            cand[g1].remove(m1); cand[g1].append(m2)
                            cand[g2].remove(m2); cand[g2].append(m1)
                            val = max_kvpr(cand)
                            if val < best_val - 1e-15:
                                best = cand
                                gload = {g: sum(x.req_rate / x.slo for x in best[g]) for g in range(gpu_num)}
                                gsize = {g: sum(x.model_size for x in best[g]) for g in range(gpu_num)}
                                best_val = val
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
    ]

    best_placement, best_val = None, float('inf')
    for order in orderings:
        pl = greedy(order)
        if pl is None:
            raise ValueError("Unable to place all models in GPU memory")
        pl = local_search(pl)
        val = max_kvpr(pl)
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
