GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Two-phase placement minimizing the maximum KVPR across GPUs.

    Phase 1 (greedy): sort models by req_rate/slo descending; assign each to
    the GPU that minimizes the resulting *global* max KVPR (ties broken by
    the GPU's own resulting KVPR), among GPUs with strictly positive
    remaining memory after placement.

    Phase 2 (local search): repeatedly try moving a model off the current
    hottest GPU to any other GPU if it strictly lowers the global max KVPR;
    stop when no improving move exists.
    """

    def kvpr(load, free):
        return load / free if free > 0 else float('inf')

    def max_kvpr(loads, frees):
        best = 0.0
        for g in range(gpu_num):
            if frees[g] < GPU_MEM_SIZE:  # only GPUs with models count
                best = max(best, kvpr(loads[g], frees[g]))
        return best

    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num  # remaining memory per GPU
    load = [0.0] * gpu_num               # sum of req_rate/slo per GPU

    for model in sorted_models:
        w = model.req_rate / model.slo
        best_idx, best_key = None, None
        for gpu_id in range(gpu_num):
            new_free = free_mem[gpu_id] - model.model_size
            if new_free <= 0:
                continue
            new_load = load[gpu_id] + w
            # resulting KVPR on this GPU
            own = new_load / new_free
            # resulting global max KVPR (only GPUs with models matter)
            glob = own
            for g in range(gpu_num):
                if g != gpu_id and free_mem[g] < GPU_MEM_SIZE:
                    glob = max(glob, kvpr(load[g], free_mem[g]))
            key = (glob, own)
            if best_key is None or key < best_key:
                best_key, best_idx = key, gpu_id

        # Fallback: allow exact-fit placement if nothing else works
        if best_idx is None:
            for gpu_id in range(gpu_num):
                if free_mem[gpu_id] >= model.model_size:
                    best_idx = gpu_id
                    break
        if best_idx is None:
            raise ValueError(f"Cannot place model of size {model.model_size} GB")

        placement[best_idx].append(model)
        load[best_idx] += w
        free_mem[best_idx] -= model.model_size

    # Phase 2: local search — moves from any GPU, then swaps if stalled
    improved = True
    while improved:
        improved = False
        cur_max = max_kvpr(load, free_mem)
        # Pass A: try moving any model from any GPU to any other GPU
        for src in range(gpu_num):
            for m in list(placement[src]):
                w = m.req_rate / m.slo
                for dst in range(gpu_num):
                    if dst == src or free_mem[dst] - m.model_size <= 0:
                        continue
                    # simulate move
                    load[src] -= w; free_mem[src] += m.model_size
                    load[dst] += w; free_mem[dst] -= m.model_size
                    if max_kvpr(load, free_mem) < cur_max - 1e-12:
                        placement[src].remove(m)
                        placement[dst].append(m)
                        improved = True
                        break
                    # undo
                    load[src] += w; free_mem[src] -= m.model_size
                    load[dst] -= w; free_mem[dst] += m.model_size
                if improved:
                    break
            if improved:
                break
        if improved:
            continue
        # Pass B: try swapping a model from the hottest GPU with one elsewhere
        hot = max((g for g in range(gpu_num) if placement[g]),
                  key=lambda g: kvpr(load[g], free_mem[g]), default=None)
        if hot is None:
            break
        for m in list(placement[hot]):
            w1 = m.req_rate / m.slo
            for dst in range(gpu_num):
                if dst == hot:
                    continue
                for m2 in list(placement[dst]):
                    w2 = m2.req_rate / m2.slo
                    # simulate swap (check memory feasibility both ways)
                    if (free_mem[hot] + m.model_size - m2.model_size <= 0 or
                            free_mem[dst] + m2.model_size - m.model_size <= 0):
                        continue
                    load[hot] += w2 - w1
                    free_mem[hot] += m.model_size - m2.model_size
                    load[dst] += w1 - w2
                    free_mem[dst] += m2.model_size - m.model_size
                    if max_kvpr(load, free_mem) < cur_max - 1e-12:
                        placement[hot].remove(m); placement[hot].append(m2)
                        placement[dst].remove(m2); placement[dst].append(m)
                        improved = True
                        break
                    # undo
                    load[hot] -= w2 - w1
                    free_mem[hot] -= m.model_size - m2.model_size
                    load[dst] -= w1 - w2
                    free_mem[dst] -= m2.model_size - m.model_size
                if improved:
                    break
            if improved:
                break

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
