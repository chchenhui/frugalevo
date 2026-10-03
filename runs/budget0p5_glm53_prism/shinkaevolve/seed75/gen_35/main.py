GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    def kvpr_of(load, used_mem):
        denom = GPU_MEM_SIZE - used_mem
        if denom <= 0:
            return float('inf') if load > 0 else 0.0
        return load / denom

    def run_pipeline(order_key):
        sorted_models = sorted(models, key=order_key, reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        free_mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        for model in sorted_models:
            w = model.req_rate / model.slo
            best_idx, best_kvpr = None, float('inf')
            for g in range(gpu_num):
                if model.model_size <= free_mem[g]:
                    k = kvpr_of(load[g] + w,
                                GPU_MEM_SIZE - free_mem[g] + model.model_size)
                    if k < best_kvpr:
                        best_kvpr, best_idx = k, g
            if best_idx is None:
                return None, float('inf')
            placement[best_idx].append(model)
            load[best_idx] += w
            free_mem[best_idx] -= model.model_size

        def state_kvprs():
            return [kvpr_of(load[g], GPU_MEM_SIZE - free_mem[g])
                    for g in range(gpu_num)]

        def improve_target(hot, iters):
            """Run move/swap/2-for-1 ops involving `hot`; only accept ops that
            reduce the current global max KVPR. Returns True if progress made."""
            progressed = False
            for _ in range(iters):
                kvprs = state_kvprs()
                cur_max = max(kvprs)
                hot_kvpr = kvprs[hot]
                others = [kvprs[g] for g in range(gpu_num) if g != hot]
                base_others = max(others) if others else 0.0

                best_gain, best_op = 0.0, None
                hot_used = GPU_MEM_SIZE - free_mem[hot]

                # --- moves: one model from hot to g ---
                for model in placement[hot]:
                    w = model.req_rate / model.slo
                    new_used = hot_used - model.model_size
                    src_kvpr = kvpr_of(load[hot] - w, new_used) if new_used > 0 else 0.0
                    for g in range(gpu_num):
                        if g == hot or model.model_size > free_mem[g]:
                            continue
                        dst_kvpr = kvpr_of(load[g] + w,
                                           GPU_MEM_SIZE - free_mem[g] + model.model_size)
                        new_max = max(base_others, src_kvpr, dst_kvpr)
                        gain = cur_max - new_max
                        if gain > best_gain + 1e-12:
                            best_gain = gain
                            best_op = ('move', model, g, None)

                # --- 1-for-1 swaps: model on hot <-> model on g ---
                for model in placement[hot]:
                    w1 = model.req_rate / model.slo
                    s1 = model.model_size
                    for g in range(gpu_num):
                        if g == hot:
                            continue
                        for m2 in placement[g]:
                            w2 = m2.req_rate / m2.slo
                            s2 = m2.model_size
                            if s2 > free_mem[hot] + s1:
                                continue
                            src_used = hot_used - s1 + s2
                            dst_used = (GPU_MEM_SIZE - free_mem[g]) - s2 + s1
                            if src_used <= 0 or dst_used <= 0:
                                continue
                            src_kvpr = kvpr_of(load[hot] - w1 + w2, src_used)
                            dst_kvpr = kvpr_of(load[g] - w2 + w1, dst_used)
                            new_max = max(base_others, src_kvpr, dst_kvpr)
                            gain = cur_max - new_max
                            if gain > best_gain + 1e-12:
                                best_gain = gain
                                best_op = ('swap', model, g, m2)

                # --- 2-for-1: one model on hot <-> two models on g ---
                for model in placement[hot]:
                    w1 = model.req_rate / model.slo
                    s1 = model.model_size
                    for g in range(gpu_num):
                        if g == hot:
                            continue
                        glist = placement[g]
                        for i in range(len(glist)):
                            for j in range(i + 1, len(glist)):
                                m2, m3 = glist[i], glist[j]
                                s2 = m2.model_size + m3.model_size
                                w2 = m2.req_rate / m2.slo + m3.req_rate / m3.slo
                                if s2 > free_mem[hot] + s1 or s1 > free_mem[g] + s2:
                                    continue
                                src_used = hot_used - s1 + s2
                                dst_used = (GPU_MEM_SIZE - free_mem[g]) - s2 + s1
                                if src_used <= 0 or dst_used <= 0:
                                    continue
                                src_kvpr = kvpr_of(load[hot] - w1 + w2, src_used)
                                dst_kvpr = kvpr_of(load[g] - w2 + w1, dst_used)
                                new_max = max(base_others, src_kvpr, dst_kvpr)
                                gain = cur_max - new_max
                                if gain > best_gain + 1e-12:
                                    best_gain = gain
                                    best_op = ('swap2', model, g, (m2, m3))

                if best_op is None:
                    break
                progressed = True

                kind, model, g, extra = best_op
                w1 = model.req_rate / model.slo
                s1 = model.model_size
                if kind == 'move':
                    placement[hot].remove(model)
                    placement[g].append(model)
                    load[hot] -= w1
                    free_mem[hot] += s1
                    load[g] += w1
                    free_mem[g] -= s1
                elif kind == 'swap':
                    m2 = extra
                    w2 = m2.req_rate / m2.slo
                    s2 = m2.model_size
                    placement[hot].remove(model)
                    placement[g].remove(m2)
                    placement[hot].append(m2)
                    placement[g].append(model)
                    load[hot] += w2 - w1
                    free_mem[hot] += s1 - s2
                    load[g] += w1 - w2
                    free_mem[g] += s2 - s1
                else:  # swap2
                    m2, m3 = extra
                    w2 = m2.req_rate / m2.slo + m3.req_rate / m3.slo
                    s2 = m2.model_size + m3.model_size
                    placement[hot].remove(model)
                    placement[g].remove(m2)
                    placement[g].remove(m3)
                    placement[hot].append(m2)
                    placement[hot].append(m3)
                    placement[g].append(model)
                    load[hot] += w2 - w1
                    free_mem[hot] += s1 - s2
                    load[g] += w1 - w2
                    free_mem[g] += s2 - s1
            return progressed

        # Phase 1: target the worst GPU
        for _ in range(30):
            kvprs = state_kvprs()
            ranked = sorted(range(gpu_num), key=lambda gg: -kvprs[gg])
            if not improve_target(ranked[0], 60):
                break

        # Phase 2: target the runner-up GPU when phase 1 stalls
        for _ in range(20):
            kvprs = state_kvprs()
            ranked = sorted(range(gpu_num), key=lambda gg: -kvprs[gg])
            if len(ranked) < 2:
                break
            if not improve_target(ranked[1], 30):
                break

        kvprs = state_kvprs()
        return placement, max(kvprs)

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    orderings = [
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: (m.req_rate / m.slo) / max(m.model_size, 1e-9),
        lambda m: (m.req_rate / m.slo, m.model_size),
    ]

    best_placement, best_max = None, float('inf')
    for key in orderings:
        placement, mx = run_pipeline(key)
        if placement is not None and mx < best_max:
            best_max, best_placement = mx, placement

    if best_placement is None:
        raise ValueError("Unable to place all models on the given GPUs.")

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