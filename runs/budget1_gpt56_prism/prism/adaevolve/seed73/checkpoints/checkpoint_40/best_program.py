GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Minimize maximum KVPR by binary-searching pressure thresholds and memoized bin packing."""
    from functools import lru_cache

    n = len(models)
    if n == 0:
        return {g: [] for g in range(gpu_num)}

    size = [m.model_size for m in models]
    weight = [m.req_rate / m.slo for m in models]
    eps = 1e-9

    if gpu_num <= 0 or any(s >= GPU_MEM_SIZE - eps for s in size):
        raise ValueError("A model leaves no KV-cache memory")

    # A fast feasible incumbent supplies a valid upper bound for threshold search.
    def greedy(order):
        groups = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num
        for i in order:
            choices = [g for g in range(gpu_num)
                       if used[g] + size[i] < GPU_MEM_SIZE - eps]
            if not choices:
                return None
            g = min(choices, key=lambda g: max(
                (load[j] + (weight[i] if j == g else 0.0)) /
                (GPU_MEM_SIZE - used[j] - (size[i] if j == g else 0.0))
                for j in range(gpu_num)))
            groups[g].append(i)
            used[g] += size[i]
            load[g] += weight[i]
        return groups, used, load

    orders = [
        sorted(range(n), key=lambda i: (-size[i], -weight[i])),
        sorted(range(n), key=lambda i: (-weight[i], -size[i])),
        sorted(range(n), key=lambda i: (-(weight[i] / (GPU_MEM_SIZE - size[i])), -size[i])),
    ]
    candidates = [x for x in (greedy(order) for order in orders) if x is not None]
    if not candidates:
        raise ValueError("Models do not fit in the available GPU memory")

    incumbent = min(candidates, key=lambda x: max(
        x[2][g] / (GPU_MEM_SIZE - x[1][g]) for g in range(gpu_num)))
    high = max(incumbent[2][g] / (GPU_MEM_SIZE - incumbent[1][g])
               for g in range(gpu_num))

    total_free = gpu_num * GPU_MEM_SIZE - sum(size)
    if total_free <= eps:
        raise ValueError("Models leave no KV-cache memory")
    low = max(sum(weight) / total_free,
              max(weight[i] / (GPU_MEM_SIZE - size[i]) for i in range(n)))

    # At threshold T, load/(80-used)<=T is exactly:
    # sum(weight_i + T*size_i) <= 80*T for every GPU.
    def pack(t, need_assignment=False):
        cap = GPU_MEM_SIZE * t
        item = [weight[i] + t * size[i] for i in range(n)]
        order = sorted(range(n), key=lambda i: (-item[i], -size[i]))
        nodes = [0]
        aborted = [False]

        @lru_cache(maxsize=None)
        def solve(k, state):
            nodes[0] += 1
            if nodes[0] > 50000:
                aborted[0] = True
                return False
            if k == n:
                return True
            i = order[k]
            seen = set()
            for p, (remaining, memory) in enumerate(state):
                key = (round(remaining, 8), round(memory, 8))
                if key in seen:
                    continue
                seen.add(key)
                if remaining + eps < item[i] or memory <= size[i] + eps:
                    continue
                nxt = list(state)
                nxt[p] = (remaining - item[i], memory - size[i])
                nxt = tuple(sorted((round(a, 10), round(b, 10)) for a, b in nxt))
                if solve(k + 1, nxt):
                    return True
            return False

        start = tuple([(cap, GPU_MEM_SIZE)] * gpu_num)
        if not solve(0, start) or aborted[0]:
            return None

        if not need_assignment:
            return True

        # Replay successful cached decisions while retaining real GPU labels.
        remaining = [cap] * gpu_num
        memory = [GPU_MEM_SIZE] * gpu_num
        groups = [[] for _ in range(gpu_num)]
        for k, i in enumerate(order):
            choices = sorted(range(gpu_num), key=lambda g: remaining[g], reverse=True)
            placed = False
            seen = set()
            for g in choices:
                key = (round(remaining[g], 8), round(memory[g], 8))
                if key in seen:
                    continue
                seen.add(key)
                if remaining[g] + eps < item[i] or memory[g] <= size[i] + eps:
                    continue
                trial = [(remaining[j], memory[j]) for j in range(gpu_num)]
                trial[g] = (remaining[g] - item[i], memory[g] - size[i])
                state = tuple(sorted((round(a, 10), round(b, 10)) for a, b in trial))
                if solve(k + 1, state):
                    remaining[g] -= item[i]
                    memory[g] -= size[i]
                    groups[g].append(i)
                    placed = True
                    break
            if not placed:
                return None
        return groups

    # Feasibility is monotone, so this directly searches the minimax KVPR.
    best_groups = incumbent[0]
    for _ in range(28):
        mid = (low + high) / 2.0
        if pack(mid):
            high = mid
        else:
            low = mid

    found = pack(high + 1e-8, True)
    if found is not None:
        best_groups = found
    return {g: [models[i] for i in best_groups[g]] for g in range(gpu_num)}

    """Use DIRECT over deterministic priority/GPU assignments, then improve by moves and swaps."""

    n = len(models)
    size = [m.model_size for m in models]
    weight = [m.req_rate / m.slo for m in models]
    eps = 1e-10

    if any(s >= GPU_MEM_SIZE - eps for s in size):
        raise ValueError("A model leaves no KV-cache memory")

    def pressure(free, load):
        return max(load[g] / free[g] for g in range(gpu_num))

    def decode(x=None):
        # x contains one ordering coordinate and one preferred-GPU coordinate
        # per model.  Canonicalizing labels removes equivalent GPU permutations.
        if x is None:
            order = sorted(range(n), key=lambda i: (-size[i], -weight[i]))
            preferred = [0] * n
        else:
            order = sorted(range(n), key=lambda i: (x[i], -size[i]))
            raw = [min(gpu_num - 1, int(x[n + i] * gpu_num)) for i in range(n)]
            rename, preferred = {}, []
            for g in raw:
                if g not in rename:
                    rename[g] = len(rename)
                preferred.append(rename[g])

        groups = {g: [] for g in range(gpu_num)}
        free, load = [GPU_MEM_SIZE] * gpu_num, [0.0] * gpu_num
        for i in order:
            choices = [g for g in range(gpu_num) if free[g] - size[i] > eps]
            if not choices:
                return None
            p = preferred[i]
            # A preferred label affects tie breaking, while the actual decision
            # minimizes the exact post-placement maximum KVPR.
            g = min(choices, key=lambda g: (
                max((load[j] + (weight[i] if j == g else 0.0)) /
                    (free[j] - (size[i] if j == g else 0.0))
                    for j in range(gpu_num)),
                g != p, g))
            groups[g].append(i)
            free[g] -= size[i]
            load[g] += weight[i]
        return groups, free, load

    def improve(state):
        groups, free, load = state
        for _ in range(min(5, n)):
            best, bestv = None, pressure(free, load)
            for a in range(gpu_num):
                for i in groups[a]:
                    for b in range(gpu_num):
                        if a == b or free[b] - size[i] <= eps:
                            continue
                        free[a] += size[i]; load[a] -= weight[i]
                        free[b] -= size[i]; load[b] += weight[i]
                        v = pressure(free, load)
                        free[a] -= size[i]; load[a] += weight[i]
                        free[b] += size[i]; load[b] -= weight[i]
                        if v < bestv - 1e-12:
                            best, bestv = ("move", a, b, i), v
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for i in groups[a]:
                        for j in groups[b]:
                            if free[a] + size[i] - size[j] <= eps or free[b] + size[j] - size[i] <= eps:
                                continue
                            free[a] += size[i] - size[j]; load[a] += weight[j] - weight[i]
                            free[b] += size[j] - size[i]; load[b] += weight[i] - weight[j]
                            v = pressure(free, load)
                            free[a] -= size[i] - size[j]; load[a] -= weight[j] - weight[i]
                            free[b] -= size[j] - size[i]; load[b] -= weight[i] - weight[j]
                            if v < bestv - 1e-12:
                                best, bestv = ("swap", a, b, i, j), v
            if best is None:
                break
            if best[0] == "move":
                _, a, b, i = best
                groups[a].remove(i); groups[b].append(i)
                free[a] += size[i]; load[a] -= weight[i]
                free[b] -= size[i]; load[b] += weight[i]
            else:
                _, a, b, i, j = best
                groups[a].remove(i); groups[b].remove(j)
                groups[a].append(j); groups[b].append(i)
                free[a] += size[i] - size[j]; load[a] += weight[j] - weight[i]
                free[b] += size[j] - size[i]; load[b] += weight[i] - weight[j]
        return groups, free, load

    best = decode()
    if best is None:
        raise ValueError("Models do not fit in the available GPU memory")
    best = improve(best)
    bestv = pressure(best[1], best[2])

    try:
        from scipy.optimize import direct

        def objective(x):
            state = decode(x)
            if state is None:
                return 1e12
            # This is exactly max load/(80-used), equivalently testing all
            # load + T*used <= 80*T feasibility thresholds.
            return pressure(state[1], state[2])

        result = direct(objective, [(0.0, 1.0)] * (2 * n),
                        maxfun=min(1200, 120 + 60 * n),
                        locally_biased=False)
        candidate = decode(result.x)
        if candidate is not None:
            candidate = improve(candidate)
            value = pressure(candidate[1], candidate[2])
            if value < bestv:
                best, bestv = candidate, value
    except Exception:
        pass

    return {g: [models[i] for i in best[0][g]] for g in range(gpu_num)}

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
