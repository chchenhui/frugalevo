GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily minimize resulting maximum KVPR, then apply improving moves and swaps."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def maximum():
        return max(load[i] / free[i] if free[i] else float("inf")
                   for i in range(gpu_num))

    # Pressure-first placement protects the most latency-sensitive models;
    # size is used as a tie-breaker to preserve future feasibility.
    for model in sorted(models, key=lambda m: (-(m.req_rate / m.slo), -m.model_size)):
        size, weight = model.model_size, model.req_rate / model.slo
        best, value = None, float("inf")

        for dst in range(gpu_num):
            if size > free[dst]:
                continue
            free[dst] -= size
            load[dst] += weight
            score = maximum()
            free[dst] += size
            load[dst] -= weight
            if score < value:
                best, value = dst, score

        if best is None:
            raise ValueError("Models do not fit in the available GPU memory")

        placement[best].append(model)
        free[best] -= size
        load[best] += weight

    # Best-improvement local search fixes greedy decisions while remaining small.
    # One additional pass often exposes improvements enabled by prior moves/swaps.
    for _ in range(min(5, len(models))):
        best, value = None, maximum()

        for src in range(gpu_num):
            for model in placement[src]:
                size, weight = model.model_size, model.req_rate / model.slo
                for dst in range(gpu_num):
                    if src == dst or size > free[dst]:
                        continue
                    free[src] += size
                    load[src] -= weight
                    free[dst] -= size
                    load[dst] += weight
                    score = maximum()
                    free[src] -= size
                    load[src] += weight
                    free[dst] += size
                    load[dst] -= weight
                    if score < value:
                        best, value = ("move", src, dst, model), score

        for left_gpu in range(gpu_num):
            for right_gpu in range(left_gpu + 1, gpu_num):
                for left in placement[left_gpu]:
                    for right in placement[right_gpu]:
                        ls, lw = left.model_size, left.req_rate / left.slo
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if rs > free[left_gpu] + ls or ls > free[right_gpu] + rs:
                            continue
                        free[left_gpu] += ls - rs
                        load[left_gpu] += rw - lw
                        free[right_gpu] += rs - ls
                        load[right_gpu] += lw - rw
                        score = maximum()
                        free[left_gpu] -= ls - rs
                        load[left_gpu] -= rw - lw
                        free[right_gpu] -= rs - ls
                        load[right_gpu] -= lw - rw
                        if score < value:
                            best, value = ("swap", left_gpu, right_gpu, left, right), score

        if best is None:
            break
        if best[0] == "move":
            _, src, dst, model = best
            placement[src].remove(model)
            placement[dst].append(model)
            free[src] += model.model_size
            load[src] -= model.req_rate / model.slo
            free[dst] -= model.model_size
            load[dst] += model.req_rate / model.slo
        else:
            _, a, b, left, right = best
            placement[a].remove(left)
            placement[b].remove(right)
            placement[a].append(right)
            placement[b].append(left)
            free[a] += left.model_size - right.model_size
            load[a] += right.req_rate / right.slo - left.req_rate / left.slo
            free[b] += right.model_size - left.model_size
            load[b] += left.req_rate / left.slo - right.req_rate / right.slo

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
