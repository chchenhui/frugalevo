GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Try pressure-aware greedy orders, then lexicographically improve GPU KVPRs by moves and swaps."""
    def weight(m):
        return m.req_rate / m.slo

    def build(order):
        placed = {g: [] for g in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            w = weight(m)
            choices = [
                ((load[g] + w) / (free[g] - m.model_size), -(free[g] - m.model_size), g)
                for g in range(gpu_num) if free[g] > m.model_size
            ]
            if not choices:
                raise ValueError(f"Unable to place model of size {m.model_size} GB")
            g = min(choices)[2]
            placed[g].append(m)
            free[g] -= m.model_size
            load[g] += w
        return placed, free, load

    orders = (
        sorted(models, key=lambda m: weight(m) / (GPU_MEM_SIZE - m.model_size), reverse=True),
        sorted(models, key=lambda m: weight(m), reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: weight(m) * m.model_size, reverse=True),
    )
    answer = None

    for order in orders:
        placed, free, load = build(order)
        for _ in range(20):
            old = [load[g] / free[g] for g in range(gpu_num)]
            best_key = tuple(sorted(old, reverse=True))
            best = None

            for a in range(gpu_num):
                for x in placed[a]:
                    wx = weight(x)
                    for b in range(gpu_num):
                        if a == b or free[b] <= x.model_size:
                            continue
                        values = [
                            (load[a] - wx) / (free[a] + x.model_size),
                            (load[b] + wx) / (free[b] - x.model_size),
                        ] + [old[g] for g in range(gpu_num) if g != a and g != b]
                        key = tuple(sorted(values, reverse=True))
                        if key < best_key:
                            best_key, best = key, ("move", a, b, x)

            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for x in placed[a]:
                        wx = weight(x)
                        for y in placed[b]:
                            fa = free[a] + x.model_size - y.model_size
                            fb = free[b] + y.model_size - x.model_size
                            if fa <= 0 or fb <= 0:
                                continue
                            values = [
                                (load[a] - wx + weight(y)) / fa,
                                (load[b] - weight(y) + wx) / fb,
                            ] + [old[g] for g in range(gpu_num) if g != a and g != b]
                            key = tuple(sorted(values, reverse=True))
                            if key < best_key:
                                best_key, best = key, ("swap", a, b, x, y)

            if best is None:
                break
            if best[0] == "move":
                _, a, b, x = best
                placed[a].remove(x)
                placed[b].append(x)
                free[a] += x.model_size
                free[b] -= x.model_size
                load[a] -= weight(x)
                load[b] += weight(x)
            else:
                _, a, b, x, y = best
                placed[a].remove(x)
                placed[b].remove(y)
                placed[a].append(y)
                placed[b].append(x)
                free[a] += x.model_size - y.model_size
                free[b] += y.model_size - x.model_size
                load[a] += weight(y) - weight(x)
                load[b] += weight(x) - weight(y)

        score = max(load[g] / free[g] for g in range(gpu_num))
        if answer is None or score < answer[0]:
            answer = score, placed

    return answer[1]

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
