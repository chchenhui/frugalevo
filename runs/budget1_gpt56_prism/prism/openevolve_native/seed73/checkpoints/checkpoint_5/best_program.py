GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use several pressure-aware greedy orders, then lexicographically improve KVPRs by moves and swaps."""
    if not models:
        return {i: [] for i in range(gpu_num)}

    def weight(m):
        return m.req_rate / m.slo

    def pressure(i):
        return load[i] / free[i] if free[i] > 0 else float("inf")

    def objective():
        return tuple(sorted((pressure(i) for i in range(gpu_num)), reverse=True))

    def greedy(order):
        p = {i: [] for i in range(gpu_num)}
        f, l = [GPU_MEM_SIZE] * gpu_num, [0.0] * gpu_num
        for m in order:
            w = weight(m)
            choices = [i for i in range(gpu_num) if m.model_size <= f[i]]
            if not choices:
                return None
            i = min(choices, key=lambda j: (l[j] + w) / (f[j] - m.model_size)
                    if f[j] > m.model_size else float("inf"))
            p[i].append(m)
            f[i] -= m.model_size
            l[i] += w
        return p, f, l

    orders = (
        sorted(models, key=weight, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: weight(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: (m.model_size, weight(m)), reverse=True),
    )
    candidates = [x for x in map(greedy, orders) if x is not None]
    if not candidates:
        raise ValueError("Unable to place all models on the available GPUs.")

    placement, free, load = min(
        candidates,
        key=lambda x: max(l / f if f > 0 else float("inf") for l, f in zip(x[2], x[1]))
    )

    # Lexicographic comparison also resolves ties between multiple worst GPUs.
    for _ in range(30):
        current = objective()
        best_score, best = current, None

        for a in range(gpu_num):
            for m in placement[a]:
                s, w = m.model_size, weight(m)
                for b in range(gpu_num):
                    if a == b or s > free[b]:
                        continue
                    old = load[a], free[a], load[b], free[b]
                    load[a], free[a] = load[a] - w, free[a] + s
                    load[b], free[b] = load[b] + w, free[b] - s
                    score = objective()
                    load[a], free[a], load[b], free[b] = old
                    if score < best_score:
                        best_score, best = score, ("move", a, b, m)

        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for x in placement[a]:
                    for y in placement[b]:
                        sx, sy = x.model_size, y.model_size
                        if sy > free[a] + sx or sx > free[b] + sy:
                            continue
                        wx, wy = weight(x), weight(y)
                        old = load[a], free[a], load[b], free[b]
                        load[a], free[a] = load[a] - wx + wy, free[a] + sx - sy
                        load[b], free[b] = load[b] - wy + wx, free[b] + sy - sx
                        score = objective()
                        load[a], free[a], load[b], free[b] = old
                        if score < best_score:
                            best_score, best = score, ("swap", a, b, x, y)

        if best is None:
            break
        if best[0] == "move":
            _, a, b, m = best
            placement[a].remove(m)
            placement[b].append(m)
            free[a] += m.model_size
            free[b] -= m.model_size
            load[a] -= weight(m)
            load[b] += weight(m)
        else:
            _, a, b, x, y = best
            placement[a].remove(x)
            placement[b].remove(y)
            placement[a].append(y)
            placement[b].append(x)
            free[a] += x.model_size - y.model_size
            free[b] += y.model_size - x.model_size
            load[a] += weight(y) - weight(x)
            load[b] += weight(x) - weight(y)

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
