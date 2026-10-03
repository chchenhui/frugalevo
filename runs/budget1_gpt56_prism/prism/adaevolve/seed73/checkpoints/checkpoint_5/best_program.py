GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place high-pressure models greedily by resulting max KVPR, then improve with moves and swaps."""

    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def pressure(i):
        return load[i] / remaining[i] if remaining[i] else float("inf")

    def maximum_pressure():
        return max(pressure(i) for i in range(gpu_num))

    for model in sorted(models, key=lambda m: (-(m.req_rate / m.slo), -m.model_size)):
        size = model.model_size
        weight = model.req_rate / model.slo
        best_gpu = None
        best_value = float("inf")

        for gpu in range(gpu_num):
            if size > remaining[gpu]:
                continue
            old_free, old_load = remaining[gpu], load[gpu]
            remaining[gpu] -= size
            load[gpu] += weight
            value = maximum_pressure()
            remaining[gpu], load[gpu] = old_free, old_load
            if value < best_value:
                best_value, best_gpu = value, gpu

        if best_gpu is None:
            raise ValueError("Models do not fit in the available GPU memory")

        placement[best_gpu].append(model)
        remaining[best_gpu] -= size
        load[best_gpu] += weight

    for _ in range(min(5, len(models))):
        current = maximum_pressure()
        best = None
        best_value = current

        for src in range(gpu_num):
            for model in placement[src]:
                size = model.model_size
                weight = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if src == dst or size > remaining[dst]:
                        continue
                    remaining[src] += size
                    load[src] -= weight
                    remaining[dst] -= size
                    load[dst] += weight
                    value = maximum_pressure()
                    remaining[src] -= size
                    load[src] += weight
                    remaining[dst] += size
                    load[dst] -= weight
                    if value < best_value:
                        best_value = value
                        best = ("move", src, dst, model)

        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for left in placement[a]:
                    for right in placement[b]:
                        ls, lw = left.model_size, left.req_rate / left.slo
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if rs > remaining[a] + ls or ls > remaining[b] + rs:
                            continue
                        remaining[a] += ls - rs
                        load[a] += rw - lw
                        remaining[b] += rs - ls
                        load[b] += lw - rw
                        value = maximum_pressure()
                        remaining[a] -= ls - rs
                        load[a] -= rw - lw
                        remaining[b] -= rs - ls
                        load[b] -= lw - rw
                        if value < best_value:
                            best_value = value
                            best = ("swap", a, b, left, right)

        if best is None:
            break

        if best[0] == "move":
            _, src, dst, model = best
            placement[src].remove(model)
            placement[dst].append(model)
            remaining[src] += model.model_size
            load[src] -= model.req_rate / model.slo
            remaining[dst] -= model.model_size
            load[dst] += model.req_rate / model.slo
        else:
            _, a, b, left, right = best
            placement[a].remove(left)
            placement[b].remove(right)
            placement[a].append(right)
            placement[b].append(left)
            remaining[a] += left.model_size - right.model_size
            load[a] += right.req_rate / right.slo - left.req_rate / left.slo
            remaining[b] += right.model_size - left.model_size
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
