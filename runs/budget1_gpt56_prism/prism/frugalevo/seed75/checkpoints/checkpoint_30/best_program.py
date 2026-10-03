GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use greedy/VND placement plus pattern set-partitioning threshold solves."""
    import math

    EPS = 1e-9
    n = len(models)
    sizes = [float(m.model_size) for m in models]
    values = [float(m.req_rate) / float(m.slo) for m in models]

    if gpu_num <= 0:
        if n:
            raise ValueError("No GPUs available")
        return {}

    def pressures(weights, remaining):
        ans = []
        for w, r in zip(weights, remaining):
            ans.append(w / r if r > EPS else (float("inf") if w > EPS else 0.0))
        return ans

    def signature(weights, remaining):
        return tuple(sorted(pressures(weights, remaining), reverse=True))

    def peak(weights, remaining):
        return max(pressures(weights, remaining), default=0.0)

    def build(order):
        place = [[] for _ in range(gpu_num)]
        rem = [float(GPU_MEM_SIZE)] * gpu_num
        weight = [0.0] * gpu_num
        for i in order:
            choices = []
            for g in range(gpu_num):
                if rem[g] - sizes[i] >= EPS:
                    nr = rem[g] - sizes[i]
                    nw = weight[g] + values[i]
                    choices.append((nw / nr, nw, -nr, g))
            if not choices:
                raise ValueError("No strictly memory-feasible placement")
            g = min(choices)[3]
            place[g].append(i)
            rem[g] -= sizes[i]
            weight[g] += values[i]
        return place, rem, weight

    def improve(place, rem, weight, limit=40):
        for _ in range(limit):
            old = signature(weight, rem)
            best_sig = old
            action = None

            for a in range(gpu_num):
                for i in place[a]:
                    for b in range(gpu_num):
                        if a == b or rem[b] - sizes[i] < EPS:
                            continue
                        rw, rr = weight[:], rem[:]
                        rw[a] -= values[i]
                        rw[b] += values[i]
                        rr[a] += sizes[i]
                        rr[b] -= sizes[i]
                        s = signature(rw, rr)
                        if s < best_sig:
                            best_sig, action = s, ("move", a, b, i)

            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for i in place[a]:
                        for j in place[b]:
                            ra = rem[a] + sizes[i] - sizes[j]
                            rb = rem[b] + sizes[j] - sizes[i]
                            if ra < EPS or rb < EPS:
                                continue
                            rw, rr = weight[:], rem[:]
                            rw[a] += values[j] - values[i]
                            rw[b] += values[i] - values[j]
                            rr[a], rr[b] = ra, rb
                            s = signature(rw, rr)
                            if s < best_sig:
                                best_sig, action = s, ("swap", a, b, i, j)

            checked = 0
            stop = False
            for a in range(gpu_num):
                if stop:
                    break
                for b in range(gpu_num):
                    if b == a or stop:
                        continue
                    for c in range(b + 1, gpu_num):
                        if c == a or stop:
                            continue
                        for i in place[a]:
                            for j in place[b]:
                                for k in place[c]:
                                    checked += 1
                                    if checked > 20000:
                                        stop = True
                                        break
                                    # a receives k, b receives i, c receives j
                                    ra = rem[a] + sizes[i] - sizes[k]
                                    rb = rem[b] + sizes[j] - sizes[i]
                                    rc = rem[c] + sizes[k] - sizes[j]
                                    if ra < EPS or rb < EPS or rc < EPS:
                                        continue
                                    rw, rr = weight[:], rem[:]
                                    rw[a] += values[k] - values[i]
                                    rw[b] += values[i] - values[j]
                                    rw[c] += values[j] - values[k]
                                    rr[a], rr[b], rr[c] = ra, rb, rc
                                    s = signature(rw, rr)
                                    if s < best_sig:
                                        best_sig = s
                                        action = ("cycle", a, b, c, i, j, k)
                                if stop:
                                    break
                            if stop:
                                break

            if action is None:
                break
            if action[0] == "move":
                _, a, b, i = action
                place[a].remove(i)
                place[b].append(i)
                rem[a] += sizes[i]
                rem[b] -= sizes[i]
                weight[a] -= values[i]
                weight[b] += values[i]
            elif action[0] == "swap":
                _, a, b, i, j = action
                place[a].remove(i)
                place[b].remove(j)
                place[a].append(j)
                place[b].append(i)
                rem[a] += sizes[i] - sizes[j]
                rem[b] += sizes[j] - sizes[i]
                weight[a] += values[j] - values[i]
                weight[b] += values[i] - values[j]
            else:
                _, a, b, c, i, j, k = action
                place[a].remove(i)
                place[b].remove(j)
                place[c].remove(k)
                place[a].append(k)
                place[b].append(i)
                place[c].append(j)
                rem[a] += sizes[i] - sizes[k]
                rem[b] += sizes[j] - sizes[i]
                rem[c] += sizes[k] - sizes[j]
                weight[a] += values[k] - values[i]
                weight[b] += values[i] - values[j]
                weight[c] += values[j] - values[k]
        return place, rem, weight

    orders = [
        sorted(range(n), key=lambda i: sizes[i], reverse=True),
        sorted(range(n), key=lambda i: values[i], reverse=True),
        sorted(range(n), key=lambda i: values[i] * sizes[i], reverse=True),
        sorted(range(n), key=lambda i: values[i] / max(EPS, GPU_MEM_SIZE - sizes[i]),
               reverse=True),
    ]

    best = None
    best_peak = float("inf")
    for order in orders:
        try:
            candidate = improve(*build(order))
            p = peak(candidate[2], candidate[1])
            if p < best_peak:
                best, best_peak = candidate, p
        except ValueError:
            pass

    if best is None:
        raise ValueError("No strictly memory-feasible placement exists")

    # Enumerate complete feasible GPU configurations and solve a threshold
    # set-partitioning master.  This captures compatible dense packings that
    # assignment variables and local exchanges can miss.
    if n and n <= 22:
        try:
            import numpy as np
            from scipy.optimize import milp, Bounds, LinearConstraint
            from scipy.sparse import lil_matrix

            patterns = []
            subset_size = [0.0] * (1 << n)
            subset_value = [0.0] * (1 << n)
            for mask in range(1, 1 << n):
                bit = mask & -mask
                i = bit.bit_length() - 1
                parent = mask ^ bit
                subset_size[mask] = subset_size[parent] + sizes[i]
                subset_value[mask] = subset_value[parent] + values[i]
                if subset_size[mask] < GPU_MEM_SIZE - EPS:
                    patterns.append((mask, subset_size[mask], subset_value[mask]))

            # One empty pattern is sufficient for unused GPUs.
            patterns.append((0, 0.0, 0.0))
            total_free = gpu_num * GPU_MEM_SIZE - sum(sizes)
            lower = sum(values) / total_free if total_free > EPS else 0.0
            upper = best_peak

            for _ in range(10):
                target = (lower + upper) * 0.5
                usable = [
                    p for p in patterns
                    if p[2] <= target * (GPU_MEM_SIZE - p[1]) + 1e-10
                ]
                cols = len(usable)
                if cols < gpu_num:
                    break

                mat = lil_matrix((n + 1, cols), dtype=float)
                lo = np.zeros(n + 1, dtype=float)
                hi = np.zeros(n + 1, dtype=float)
                lo[:n] = hi[:n] = 1.0
                lo[n] = hi[n] = float(gpu_num)
                for col, (mask, _, _) in enumerate(usable):
                    mat[n, col] = 1.0
                    while mask:
                        bit = mask & -mask
                        mat[bit.bit_length() - 1, col] = 1.0
                        mask ^= bit

                result = milp(
                    c=np.zeros(cols, dtype=float),
                    integrality=np.ones(cols, dtype=float),
                    bounds=Bounds(0.0, 1.0),
                    constraints=LinearConstraint(mat.tocsr(), lo, hi),
                    options={"time_limit": 0.15},
                )
                if result.status == 2:
                    lower = target
                    continue
                if result.status != 0 or result.x is None:
                    break

                chosen = [
                    usable[j][0] for j, x in enumerate(result.x)
                    if x > 0.5
                ]
                if len(chosen) != gpu_num:
                    break

                place = [[] for _ in range(gpu_num)]
                rem = [float(GPU_MEM_SIZE)] * gpu_num
                weight = [0.0] * gpu_num
                for g, mask in enumerate(chosen):
                    while mask:
                        bit = mask & -mask
                        i = bit.bit_length() - 1
                        place[g].append(i)
                        rem[g] -= sizes[i]
                        weight[g] += values[i]
                        mask ^= bit

                candidate = improve(place, rem, weight)
                candidate_peak = peak(candidate[2], candidate[1])
                if candidate_peak <= upper + 1e-7:
                    best, best_peak = candidate, candidate_peak
                    upper = candidate_peak
                else:
                    break
        except Exception:
            pass

    return {g: [models[i] for i in best[0][g]] for g in range(gpu_num)}

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    all_kvpr = []
    for gpu_num, gpu_models in generate_test_gpu_models():
        result = compute_model_placement(gpu_num, gpu_models)
        all_kvpr.append(safe_float(calculate_kvcache_pressure(result)))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr
    print(f"Max KVPR: {avg_kvpr:.3f}")