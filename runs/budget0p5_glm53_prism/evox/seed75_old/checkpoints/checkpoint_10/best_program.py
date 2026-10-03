GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Greedy placement (sort by req_rate/slo desc, place on GPU minimizing
    resulting KVPR) followed by a local search that moves models from the
    hottest GPU to cooler GPUs whenever it lowers the max KVPR.
    """

    sorted_models = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    free_mem = [GPU_MEM_SIZE] * gpu_num
    pressure = [0.0] * gpu_num

    for model in sorted_models:
        best_idx, best_kvpr = None, float('inf')
        for gpu_id in range(gpu_num):
            rem = free_mem[gpu_id] - model.model_size
            if rem > 0:
                kvpr = (pressure[gpu_id] + model.req_rate / model.slo) / rem
                if kvpr < best_kvpr:
                    best_kvpr, best_idx = kvpr, gpu_id

        if best_idx is None:
            raise ValueError(f"Cannot fit model of size {model.model_size} GB")

        placement[best_idx].append(model)
        pressure[best_idx] += model.req_rate / model.slo
        free_mem[best_idx] -= model.model_size

    # Local search: moves and swaps off the hottest GPU to reduce max KVPR
    def kvpr(gpu_id):
        return pressure[gpu_id] / free_mem[gpu_id] if free_mem[gpu_id] > 0 else float('inf')

    improved = True
    while improved:
        improved = False
        hot = max(range(gpu_num), key=kvpr)
        current_max = kvpr(hot)
        # try moving each model on the hot GPU to another GPU
        for model in list(placement[hot]):
            for dst in range(gpu_num):
                if dst == hot or model.model_size > free_mem[dst]:
                    continue
                rem_dst = free_mem[dst] - model.model_size
                new_dst = (pressure[dst] + model.req_rate / model.slo) / rem_dst
                new_hot = (pressure[hot] - model.req_rate / model.slo) / (free_mem[hot] + model.model_size)
                if max(new_dst, new_hot) < current_max:
                    placement[hot].remove(model)
                    placement[dst].append(model)
                    pressure[hot] -= model.req_rate / model.slo
                    free_mem[hot] += model.model_size
                    pressure[dst] += model.req_rate / model.slo
                    free_mem[dst] -= model.model_size
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # no move helped: try swapping a hot-GPU model with a cooler-GPU model
        for m_hot in list(placement[hot]):
            for dst in range(gpu_num):
                if dst == hot:
                    continue
                for m_dst in list(placement[dst]):
                    d_hot = m_hot.model_size - m_dst.model_size
                    if free_mem[dst] + m_dst.model_size < m_hot.model_size or free_mem[hot] + m_hot.model_size < m_dst.model_size:
                        continue
                    p_hot = pressure[hot] - m_hot.req_rate / m_hot.slo + m_dst.req_rate / m_dst.slo
                    p_dst = pressure[dst] - m_dst.req_rate / m_dst.slo + m_hot.req_rate / m_hot.slo
                    new_hot = p_hot / (free_mem[hot] + d_hot)
                    new_dst = p_dst / (free_mem[dst] - d_hot)
                    if max(new_hot, new_dst) < current_max:
                        placement[hot].remove(m_hot)
                        placement[dst].remove(m_dst)
                        placement[hot].append(m_dst)
                        placement[dst].append(m_hot)
                        pressure[hot] = p_hot
                        pressure[dst] = p_dst
                        free_mem[hot] += d_hot
                        free_mem[dst] -= d_hot
                        improved = True
                        break
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
