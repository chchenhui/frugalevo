GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily balance KVPR, then apply improving feasible moves and swaps."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    pressure = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        candidates = [i for i in range(gpu_num) if remaining[i] >= model.model_size]
        if not candidates:
            raise ValueError(f"Unable to place model; remaining memory: {remaining}")
        gpu = min(candidates, key=lambda i: pressure[i] / remaining[i])
        placement[gpu].append(model)
        remaining[gpu] -= model.model_size
        pressure[gpu] += model.req_rate / model.slo

    def ratios():
        return [pressure[i] / remaining[i] if remaining[i] else float("inf")
                for i in range(gpu_num)]

    for _ in range(len(models)):
        old = ratios()
        best_score, best = (max(old), sum(old)), None

        for src in range(gpu_num):
            for model in placement[src]:
                weight, size = model.req_rate / model.slo, model.model_size
                for dst in range(gpu_num):
                    if src == dst or remaining[dst] < size:
                        continue
                    a = (pressure[src] - weight) / (remaining[src] + size)
                    free = remaining[dst] - size
                    b = (pressure[dst] + weight) / free if free else float("inf")
                    values = [a if i == src else b if i == dst else old[i]
                              for i in range(gpu_num)]
                    score = (max(values), sum(values))
                    if score < best_score:
                        best_score, best = score, ("move", src, dst, model)

        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for x in placement[a]:
                    for y in placement[b]:
                        if remaining[a] + x.model_size < y.model_size:
                            continue
                        if remaining[b] + y.model_size < x.model_size:
                            continue
                        wx, wy = x.req_rate / x.slo, y.req_rate / y.slo
                        ra = remaining[a] + x.model_size - y.model_size
                        rb = remaining[b] + y.model_size - x.model_size
                        va = (pressure[a] - wx + wy) / ra if ra else float("inf")
                        vb = (pressure[b] - wy + wx) / rb if rb else float("inf")
                        values = [va if i == a else vb if i == b else old[i]
                                  for i in range(gpu_num)]
                        score = (max(values), sum(values))
                        if score < best_score:
                            best_score, best = score, ("swap", a, b, x, y)

        if best is None:
            break
        if best[0] == "move":
            _, src, dst, model = best
            placement[src].remove(model)
            placement[dst].append(model)
            remaining[src] += model.model_size
            remaining[dst] -= model.model_size
            weight = model.req_rate / model.slo
            pressure[src] -= weight
            pressure[dst] += weight
        else:
            _, a, b, x, y = best
            placement[a].remove(x)
            placement[b].remove(y)
            placement[a].append(y)
            placement[b].append(x)
            remaining[a] += x.model_size - y.model_size
            remaining[b] += y.model_size - x.model_size
            pressure[a] += y.req_rate / y.slo - x.req_rate / x.slo
            pressure[b] += x.req_rate / x.slo - y.req_rate / y.slo

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
