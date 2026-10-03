GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily place models, then repeatedly apply the best KVPR-improving move or swap."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        gpu = min(
            (i for i in range(gpu_num) if model.model_size <= remaining[i]),
            key=lambda i: load[i] / remaining[i],
            default=None,
        )
        if gpu is None:
            raise ValueError(f"Unable to place model of size {model.model_size} GB")
        placement[gpu].append(model)
        load[gpu] += model.req_rate / model.slo
        remaining[gpu] -= model.model_size

    # Best-improvement search is monotonic: every accepted operation lowers the
    # actual maximum final pressure.  Swaps escape cases where no direct move fits.
    for _ in range(len(models)):
        pressure = [load[i] / remaining[i] if remaining[i] else float("inf")
                    for i in range(gpu_num)]
        best_value, best = max(pressure), None

        for source in range(gpu_num):
            for model in placement[source]:
                weight, size = model.req_rate / model.slo, model.model_size
                for dest in range(gpu_num):
                    if source == dest or remaining[dest] <= size:
                        continue
                    value = max(
                        (load[source] - weight) / (remaining[source] + size) if i == source else
                        (load[dest] + weight) / (remaining[dest] - size) if i == dest else
                        pressure[i]
                        for i in range(gpu_num)
                    )
                    if value < best_value:
                        best_value, best = value, ("move", source, dest, model)

        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for x in placement[a]:
                    wx, sx = x.req_rate / x.slo, x.model_size
                    for y in placement[b]:
                        wy, sy = y.req_rate / y.slo, y.model_size
                        ra, rb = remaining[a] + sx - sy, remaining[b] + sy - sx
                        if ra <= 0 or rb <= 0:
                            continue
                        value = max(
                            (load[a] - wx + wy) / ra if i == a else
                            (load[b] - wy + wx) / rb if i == b else pressure[i]
                            for i in range(gpu_num)
                        )
                        if value < best_value:
                            best_value, best = value, ("swap", a, b, x, y)

        if best is None:
            break
        if best[0] == "move":
            _, source, dest, model = best
            weight, size = model.req_rate / model.slo, model.model_size
            placement[source].remove(model)
            placement[dest].append(model)
            load[source] -= weight
            load[dest] += weight
            remaining[source] += size
            remaining[dest] -= size
        else:
            _, a, b, x, y = best
            wx, sx = x.req_rate / x.slo, x.model_size
            wy, sy = y.req_rate / y.slo, y.model_size
            placement[a].remove(x)
            placement[b].remove(y)
            placement[a].append(y)
            placement[b].append(x)
            load[a] += wy - wx
            load[b] += wx - wy
            remaining[a] += sx - sy
            remaining[b] += sy - sx

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
