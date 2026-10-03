GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Minimize peak KVPR using several feasible seeds, VND, and small MILPs."""
    import math

    EPS = 1e-9
    n = len(models)
    sizes = [m.model_size for m in models]
    values = [m.req_rate / m.slo for m in models]

    def peak(weights, remaining):
        result = 0.0
        for w, r in zip(weights, remaining):
            if r <= 0:
                return float("inf") if w > 0 else result
            result = max(result, w / r)
        return result

    def build_seed(order):
        placement = [[] for _ in range(gpu_num)]
        remaining = [float(GPU_MEM_SIZE)] * gpu_num
        weights = [0.0] * gpu_num

        for i in order:
            candidates = []
            for g in range(gpu_num):
                new_remaining = remaining[g] - sizes[i]
                if new_remaining >= EPS:
                    candidates.append(
                        ((weights[g] + values[i]) / new_remaining, weights[g], g)
                    )
            if not candidates:
                raise ValueError(
                    "Unable to place all models with positive remaining GPU memory"
                )
            _, _, g = min(candidates)
            placement[g].append(i)
            remaining[g] -= sizes[i]
            weights[g] += values[i]
        return placement, remaining, weights

    def improve(placement, remaining, weights, limit=60):
        """Interleave relocations, swaps, and exchanges using lexicographic VND."""
        def signature(test_weights, test_remaining):
            pressures = []
            for weight, free in zip(test_weights, test_remaining):
                if free <= 0:
                    pressures.append(float("inf") if weight > 0 else 0.0)
                else:
                    pressures.append(weight / free)
            return tuple(sorted(pressures, reverse=True))

        for _ in range(limit):
            current = signature(weights, remaining)
            best_value = current
            best_action = None

            # Relocations.
            for a in range(gpu_num):
                for i in placement[a]:
                    for b in range(gpu_num):
                        if a == b or remaining[b] - sizes[i] < EPS:
                            continue
                        rw = weights[:]
                        rr = remaining[:]
                        rw[a] -= values[i]
                        rw[b] += values[i]
                        rr[a] += sizes[i]
                        rr[b] -= sizes[i]
                        p = signature(rw, rr)
                        if p < best_value:
                            best_value = p
                            best_action = ("move", a, b, i)

            # One-for-one swaps.
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for i in placement[a]:
                        for j in placement[b]:
                            ra = remaining[a] + sizes[i] - sizes[j]
                            rb = remaining[b] + sizes[j] - sizes[i]
                            if ra < EPS or rb < EPS:
                                continue
                            rw = weights[:]
                            rr = remaining[:]
                            rw[a] += values[j] - values[i]
                            rw[b] += values[i] - values[j]
                            rr[a], rr[b] = ra, rb
                            p = signature(rw, rr)
                            if p < best_value:
                                best_value = p
                                best_action = ("swap", a, b, i, j)

            # Directed exchange: two models from a for one model from b.
            for a in range(gpu_num):
                if len(placement[a]) < 2:
                    continue
                for b in range(gpu_num):
                    if a == b:
                        continue
                    source = placement[a]
                    for x in range(len(source) - 1):
                        i = source[x]
                        for y in range(x + 1, len(source)):
                            j = source[y]
                            pair_size = sizes[i] + sizes[j]
                            pair_value = values[i] + values[j]
                            for k in placement[b]:
                                ra = remaining[a] + pair_size - sizes[k]
                                rb = remaining[b] + sizes[k] - pair_size
                                if ra < EPS or rb < EPS:
                                    continue
                                rw = weights[:]
                                rr = remaining[:]
                                rw[a] += values[k] - pair_value
                                rw[b] += pair_value - values[k]
                                rr[a], rr[b] = ra, rb
                                p = signature(rw, rr)
                                if p < best_value:
                                    best_value = p
                                    best_action = ("two_one", a, b, i, j, k)

            if best_action is None:
                break

            kind = best_action[0]
            if kind == "move":
                _, a, b, i = best_action
                placement[a].remove(i)
                placement[b].append(i)
                remaining[a] += sizes[i]
                remaining[b] -= sizes[i]
                weights[a] -= values[i]
                weights[b] += values[i]
            elif kind == "swap":
                _, a, b, i, j = best_action
                placement[a].remove(i)
                placement[b].remove(j)
                placement[a].append(j)
                placement[b].append(i)
                remaining[a] += sizes[i] - sizes[j]
                remaining[b] += sizes[j] - sizes[i]
                weights[a] += values[j] - values[i]
                weights[b] += values[i] - values[j]
            else:
                _, a, b, i, j, k = best_action
                placement[a].remove(i)
                placement[a].remove(j)
                placement[b].remove(k)
                placement[a].append(k)
                placement[b].append(i)
                placement[b].append(j)
                remaining[a] += sizes[i] + sizes[j] - sizes[k]
                remaining[b] += sizes[k] - sizes[i] - sizes[j]
                weights[a] += values[k] - values[i] - values[j]
                weights[b] += values[i] + values[j] - values[k]

        return placement, remaining, weights

    # Different constructions give the local search materially different basins.
    orders = [
        sorted(range(n), key=lambda i: sizes[i], reverse=True),
        sorted(range(n), key=lambda i: values[i], reverse=True),
        sorted(
            range(n),
            key=lambda i: values[i] / max(GPU_MEM_SIZE - sizes[i], EPS),
            reverse=True,
        ),
        sorted(
            range(n),
            key=lambda i: (values[i] + 1e-6) * sizes[i],
            reverse=True,
        ),
    ]

    best = None
    best_peak = float("inf")
    for order in orders:
        try:
            candidate = improve(*build_seed(order))
        except ValueError:
            continue
        candidate_peak = peak(candidate[2], candidate[1])
        if candidate_peak < best_peak:
            best_peak = candidate_peak
            best = candidate

    if best is None:
        raise ValueError("No strictly memory-feasible placement exists")

    # A bounded global threshold MILP is worthwhile only for compact instances.
    if n and n * gpu_num <= 180:
        try:
            import numpy as np
            from scipy.optimize import milp, Bounds, LinearConstraint
            from scipy.sparse import lil_matrix

            total_size = sum(sizes)
            total_value = sum(values)
            denominator = gpu_num * GPU_MEM_SIZE - total_size
            lower = total_value / denominator if denominator > EPS else 0.0
            upper = best_peak
            variable_count = n * gpu_num

            for _ in range(12):
                target = (lower + upper) / 2.0
                rows = n + 2 * gpu_num
                matrix = lil_matrix((rows, variable_count), dtype=float)
                lo = np.full(rows, -np.inf)
                hi = np.full(rows, np.inf)

                for i in range(n):
                    for g in range(gpu_num):
                        matrix[i, i * gpu_num + g] = 1.0
                    lo[i] = hi[i] = 1.0

                for g in range(gpu_num):
                    ratio_row = n + g
                    memory_row = n + gpu_num + g
                    for i in range(n):
                        col = i * gpu_num + g
                        matrix[ratio_row, col] = values[i] + target * sizes[i]
                        matrix[memory_row, col] = sizes[i]
                    hi[ratio_row] = target * GPU_MEM_SIZE
                    hi[memory_row] = GPU_MEM_SIZE - EPS

                result = milp(
                    c=np.zeros(variable_count),
                    integrality=np.ones(variable_count),
                    bounds=Bounds(0.0, 1.0),
                    constraints=LinearConstraint(matrix.tocsr(), lo, hi),
                    options={"time_limit": 0.15},
                )

                if result.status == 2:
                    lower = target
                    continue
                if result.status != 0 or result.x is None:
                    break

                assigned = [int(np.argmax(result.x[i * gpu_num:(i + 1) * gpu_num]))
                            for i in range(n)]
                placement = [[] for _ in range(gpu_num)]
                remaining = [float(GPU_MEM_SIZE)] * gpu_num
                weights = [0.0] * gpu_num
                for i, g in enumerate(assigned):
                    placement[g].append(i)
                    remaining[g] -= sizes[i]
                    weights[g] += values[i]

                candidate_peak = peak(weights, remaining)
                if min(remaining) >= EPS and candidate_peak <= target + 1e-7:
                    best = improve(placement, remaining, weights)
                    best_peak = peak(best[2], best[1])
                    upper = min(upper, best_peak)
                else:
                    break
        except Exception:
            pass

    return {
        gpu_id: [models[i] for i in best[0][gpu_id]]
        for gpu_id in range(gpu_num)
    }

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for gpu_num, gpu_models in test_cases:
        results = compute_model_placement(gpu_num, gpu_models)
        all_kvpr.append(safe_float(calculate_kvcache_pressure(results)))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr
    print(f"Max KVPR: {avg_kvpr:.3f}")