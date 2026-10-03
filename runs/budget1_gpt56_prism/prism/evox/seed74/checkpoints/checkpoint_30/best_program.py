GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Pack large models first, then repeatedly apply the best KVPR-reducing move or swap."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(
        models, key=lambda m: (m.model_size, m.req_rate / m.slo), reverse=True
    ):
        best = min(
            (i for i in range(gpu_num) if model.model_size <= free[i]),
            key=lambda i: load[i] / free[i] if free[i] else float("inf"),
            default=None,
        )
        if best is None:
            raise ValueError("Models do not fit in available GPU memory")
        placement[best].append(model)
        free[best] -= model.model_size
        load[best] += model.req_rate / model.slo

    for _ in range(len(models)):
        ratios = [load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num)]
        best_score = max(ratios)
        best_action = None

        for src in range(gpu_num):
            for model in placement[src]:
                size = model.model_size
                pressure = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if src == dst or size > free[dst]:
                        continue
                    src_ratio = (load[src] - pressure) / (free[src] + size)
                    remaining = free[dst] - size
                    dst_ratio = (load[dst] + pressure) / remaining if remaining else float("inf")
                    score = max(
                        src_ratio if i == src else dst_ratio if i == dst else ratios[i]
                        for i in range(gpu_num)
                    )
                    if score < best_score:
                        best_score = score
                        best_action = ("move", src, dst, model, size, pressure)

        for src in range(gpu_num):
            for dst in range(src + 1, gpu_num):
                for left in placement[src]:
                    left_size = left.model_size
                    left_load = left.req_rate / left.slo
                    for right in placement[dst]:
                        right_size = right.model_size
                        right_load = right.req_rate / right.slo
                        src_free = free[src] + left_size - right_size
                        dst_free = free[dst] + right_size - left_size
                        if src_free < 0 or dst_free < 0:
                            continue
                        src_ratio = (load[src] - left_load + right_load) / src_free
                        dst_ratio = (load[dst] - right_load + left_load) / dst_free
                        score = max(
                            src_ratio if i == src else dst_ratio if i == dst else ratios[i]
                            for i in range(gpu_num)
                        )
                        if score < best_score:
                            best_score = score
                            best_action = ("swap", src, dst, left, right, left_size,
                                           right_size, left_load, right_load)

        if best_action is None:
            break
        if best_action[0] == "move":
            _, src, dst, model, size, pressure = best_action
            placement[src].remove(model)
            placement[dst].append(model)
            free[src] += size
            free[dst] -= size
            load[src] -= pressure
            load[dst] += pressure
        else:
            _, src, dst, left, right, left_size, right_size, left_load, right_load = best_action
            placement[src].remove(left)
            placement[dst].remove(right)
            placement[src].append(right)
            placement[dst].append(left)
            free[src] += left_size - right_size
            free[dst] += right_size - left_size
            load[src] += right_load - left_load
            load[dst] += left_load - right_load

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
