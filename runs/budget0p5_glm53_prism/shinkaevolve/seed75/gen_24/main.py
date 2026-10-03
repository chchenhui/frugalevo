GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses best-fit greedy construction followed by local-search refinement.
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num  # sum of req_rate/slo per GPU

    def kvpr(g):
        denom = GPU_MEM_SIZE - free_mem[g]
        if denom <= 0:
            return float('inf') if load[g] > 0 else 0.0
        return load[g] / denom

    def max_kvpr(exclude=None):
        vals = [kvpr(g) for g in range(gpu_num) if g != exclude]
        return max(vals) if vals else 0.0

    # Sort models by pressure contribution (r/s) descending
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # Best-fit greedy: choose GPU minimizing resulting KVPR
    for model in sorted_models:
        best_g, best_val = None, float('inf')
        for g in range(gpu_num):
            if model.model_size > free_mem[g]:
                continue
            denom = GPU_MEM_SIZE - (free_mem[g] - model.model_size)
            if denom <= 0:
                continue
            val = (load[g] + model.req_rate / model.slo) / denom
            if val < best_val:
                best_val, best_g = val, g
        if best_g is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Free memory per GPU: {free_mem}"
            )
        placement[best_g].append(model)
        free_mem[best_g] -= model.model_size
        load[best_g] += model.req_rate / model.slo

    # Local search refinement: move, swap (1-for-1), and 2-for-1 exchanges
    # off the worst GPU, accepted only on strict improvement of global max KVPR.
    improved = True
    max_iters = 200
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        worst = max(range(gpu_num), key=lambda g: kvpr(g))
        cur_max = max_kvpr()

        def try_accept(new_max):
            return new_max < cur_max - 1e-12

        done = False
        # 1) simple moves
        for m_idx in range(len(placement[worst])):
            if done:
                break
            model = placement[worst][m_idx]
            for g in range(gpu_num):
                if g == worst or model.model_size > free_mem[g]:
                    continue
                placement[worst].pop(m_idx)
                free_mem[worst] += model.model_size
                load[worst] -= model.req_rate / model.slo
                free_mem[g] -= model.model_size
                load[g] += model.req_rate / model.slo
                if try_accept(max_kvpr()):
                    placement[g].append(model)
                    improved = True
                    done = True
                    break
                free_mem[g] += model.model_size
                load[g] -= model.req_rate / model.slo
                free_mem[worst] -= model.model_size
                load[worst] += model.req_rate / model.slo
                placement[worst].insert(m_idx, model)
        if done:
            continue

        # 2) 1-for-1 swaps: model on worst <-> one model from GPU g
        for m_idx in range(len(placement[worst])):
            if done:
                break
            model = placement[worst][m_idx]
            for g in range(gpu_num):
                if g == worst:
                    continue
                for o_idx in range(len(placement[g])):
                    if done:
                        break
                    other = placement[g][o_idx]
                    # simulate swap
                    placement[worst].pop(m_idx)
                    placement[g].pop(o_idx)
                    free_mem[worst] += model.model_size
                    load[worst] -= model.req_rate / model.slo
                    free_mem[g] += other.model_size
                    load[g] -= other.req_rate / other.slo
                    if other.model_size <= free_mem[worst] and model.model_size <= free_mem[g]:
                        free_mem[worst] -= other.model_size
                        load[worst] += other.req_rate / other.slo
                        free_mem[g] -= model.model_size
                        load[g] += model.req_rate / model.slo
                        if GPU_MEM_SIZE - free_mem[worst] > 0 and GPU_MEM_SIZE - free_mem[g] > 0 \
                                and try_accept(max_kvpr()):
                            placement[worst].append(other)
                            placement[g].append(model)
                            improved = True
                            done = True
                            break
                        free_mem[worst] += other.model_size
                        load[worst] -= other.req_rate / other.slo
                        free_mem[g] += model.model_size
                        load[g] -= model.req_rate / model.slo
                    # revert swap
                    free_mem[worst] -= model.model_size
                    load[worst] += model.req_rate / model.slo
                    free_mem[g] += other.model_size
                    load[g] += other.req_rate / other.slo
                    placement[worst].insert(m_idx, model)
                    placement[g].insert(o_idx, other)
                if done:
                    break
        if done:
            continue

        # 3) 2-for-1 exchanges: one model on worst <-> two models from GPU g
        for m_idx in range(len(placement[worst])):
            if done:
                break
            model = placement[worst][m_idx]
            for g in range(gpu_num):
                if g == worst or len(placement[g]) < 2:
                    continue
                n = len(placement[g])
                for a in range(n):
                    if done:
                        break
                    for b in range(a + 1, n):
                        if done:
                            break
                        ma, mb = placement[g][a], placement[g][b]
                        pair_size = ma.model_size + mb.model_size
                        pair_load = ma.req_rate / ma.slo + mb.req_rate / mb.slo
                        # simulate: remove model from worst, remove pair from g
                        placement[worst].pop(m_idx)
                        del placement[g][b]
                        del placement[g][a]
                        free_mem[worst] += model.model_size
                        load[worst] -= model.req_rate / model.slo
                        free_mem[g] += pair_size
                        load[g] -= pair_load
                        if pair_size <= free_mem[worst] and model.model_size <= free_mem[g]:
                            free_mem[worst] -= pair_size
                            load[worst] += pair_load
                            free_mem[g] -= model.model_size
                            load[g] += model.req_rate / model.slo
                            if GPU_MEM_SIZE - free_mem[worst] > 0 and GPU_MEM_SIZE - free_mem[g] > 0 \
                                    and try_accept(max_kvpr()):
                                placement[worst].extend([ma, mb])
                                placement[g].append(model)
                                improved = True
                                done = True
                                break
                            free_mem[worst] += pair_size
                            load[worst] -= pair_load
                            free_mem[g] += model.model_size
                            load[g] -= model.req_rate / model.slo
                        # revert
                        free_mem[worst] -= model.model_size
                        load[worst] += model.req_rate / model.slo
                        free_mem[g] -= pair_size
                        load[g] += pair_load
                        placement[worst].insert(m_idx, model)
                        placement[g].insert(a, ma)
                        placement[g].insert(b, mb)
                if done:
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