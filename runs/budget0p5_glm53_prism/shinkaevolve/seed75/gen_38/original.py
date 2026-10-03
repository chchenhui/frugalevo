GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
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

        for _ in range(60):
            kvprs = state_kvprs()
            hot = max(range(gpu_num), key=lambda g: kvprs[g])
            hot_kvpr = kvprs[hot]
            others = [kvprs[g] for g in range(gpu_num) if g != hot]
            base_others = max(others) if others else 0.0
            cur_max = hot_kvpr

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
                        best_op = ('move', (model,), g, ())

            # --- 1-for-1 swaps ---
            for model in placement[hot]:
                w1 = model.req_rate / model.slo
                s1 = model.model_size
                for g in range(gpu_num):
                    if g == hot:
                        continue
                    for m2 in placement[g]:
                        w2 = m2.req_rate / m2.slo
                        s2 = m2.model_size
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
                            best_op = ('swap', (model,), g, (m2,))

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
                                best_op = ('swap', (model,), g, (m2, m3))

            # --- 2-for-2: two models on hot <-> two models on g ---
            hotlist = placement[hot]
            for i in range(len(hotlist)):
                for j in range(i + 1, len(hotlist)):
                    m1, m1b = hotlist[i], hotlist[j]
                    s1 = m1.model_size + m1b.model_size
                    w1 = m1.req_rate / m1.slo + m1b.req_rate / m1b.slo
                    for g in range(gpu_num):
                        if g == hot:
                            continue
                        glist = placement[g]
                        for a in range(len(glist)):
                            for b in range(a + 1, len(glist)):
                                m2, m2b = glist[a], glist[b]
                                s2 = m2.model_size + m2b.model_size
                                w2 = m2.req_rate / m2.slo + m2b.req_rate / m2b.slo
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
                                    best_op = ('swap', (m1, m1b), g, (m2, m2b))

            if best_op is None:
                break

            kind, src_models, g, dst_models = best_op
            w_src = sum(m.req_rate / m.slo for m in src_models)
            s_src = sum(m.model_size for m in src_models)
            w_dst = sum(m.req_rate / m.slo for m in dst_models)
            s_dst = sum(m.model_size for m in dst_models)
            for m in src_models:
                placement[hot].remove(m)
            for m in dst_models:
                placement[g].remove(m)
            for m in dst_models:
                placement[hot].append(m)
            for m in src_models:
                placement[g].append(m)
            load[hot] += w_dst - w_src
            free_mem[hot] += s_src - s_dst
            load[g] += w_src - w_dst
            free_mem[g] += s_dst - s_src

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