GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use greedy/local-search placement, then improve its maximum KVPR with fixed-threshold MILP feasibility."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def pressure(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    def objective(changes=()):
        loads, memory = load[:], remaining[:]
        for g, dw, dm in changes:
            loads[g] += dw
            memory[g] += dm
        return tuple(sorted((pressure(loads[g], memory[g]) for g in range(gpu_num)), reverse=True))

    ordered = sorted(
        models,
        key=lambda m: pressure(m.req_rate / m.slo, GPU_MEM_SIZE - m.model_size),
        reverse=True,
    )
    for model in ordered:
        size, weight = model.model_size, model.req_rate / model.slo
        choices = [g for g in range(gpu_num) if remaining[g] >= size]
        if not choices:
            raise ValueError("Models cannot fit in the available GPU memory")
        best = min(choices, key=lambda g: pressure(load[g] + weight, remaining[g] - size))
        placement[best].append(model)
        load[best] += weight
        remaining[best] -= size

    for _ in range(min(16, len(models))):
        best_value, best_action = objective(), None
        for source in range(gpu_num):
            for model in placement[source]:
                size, weight = model.model_size, model.req_rate / model.slo
                for target in range(gpu_num):
                    if target == source or remaining[target] < size:
                        continue
                    value = objective(((source, -weight, size), (target, weight, -size)))
                    if value < best_value:
                        best_value, best_action = value, ("move", source, target, model)

        for source in range(gpu_num):
            for target in range(source + 1, gpu_num):
                for left in placement[source]:
                    ls, lw = left.model_size, left.req_rate / left.slo
                    for right in placement[target]:
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if remaining[source] + ls < rs or remaining[target] + rs < ls:
                            continue
                        value = objective(((source, rw - lw, ls - rs),
                                           (target, lw - rw, rs - ls)))
                        if value < best_value:
                            best_value, best_action = value, ("swap", source, target, left, right)

        if best_action is None:
            break
        if best_action[0] == "move":
            _, source, target, model = best_action
            size, weight = model.model_size, model.req_rate / model.slo
            placement[source].remove(model)
            placement[target].append(model)
            load[source] -= weight
            load[target] += weight
            remaining[source] += size
            remaining[target] -= size
        else:
            _, source, target, left, right = best_action
            ls, lw = left.model_size, left.req_rate / left.slo
            rs, rw = right.model_size, right.req_rate / right.slo
            placement[source].remove(left)
            placement[target].remove(right)
            placement[source].append(right)
            placement[target].append(left)
            load[source] += rw - lw
            load[target] += lw - rw
            remaining[source] += ls - rs
            remaining[target] += rs - ls

    # For a fixed pressure bound T, the nonlinear KVPR condition becomes:
    # sum((weight + T * size) * x) <= 80 * T for every GPU.
    # The greedy result is always retained as a safe incumbent/fallback.
    n = len(models)
    upper = max((pressure(load[g], remaining[g]) for g in range(gpu_num)), default=0.0)
    if n and gpu_num and upper > 0 and upper < float("inf") and n * gpu_num <= 180:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix

            sizes = np.array([m.model_size for m in models], dtype=float)
            weights = np.array([m.req_rate / m.slo for m in models], dtype=float)
            lower = max(
                max((w / (GPU_MEM_SIZE - s) if s < GPU_MEM_SIZE else upper)
                    for s, w in zip(sizes, weights)),
                weights.sum() / max(gpu_num * GPU_MEM_SIZE - sizes.sum(), 1e-9),
            )
            best_x, lo, hi = None, min(lower, upper), upper

            def feasible(limit):
                # Assignment rows, memory rows, and fixed-pressure rows.
                rows = n + 2 * gpu_num
                matrix = lil_matrix((rows, n * gpu_num))
                lb = np.full(rows, -np.inf)
                ub = np.full(rows, np.inf)
                for i in range(n):
                    for g in range(gpu_num):
                        matrix[i, i * gpu_num + g] = 1
                    lb[i] = ub[i] = 1
                for g in range(gpu_num):
                    for i in range(n):
                        j = i * gpu_num + g
                        matrix[n + g, j] = sizes[i]
                        matrix[n + gpu_num + g, j] = weights[i] + limit * sizes[i]
                    ub[n + g] = GPU_MEM_SIZE
                    ub[n + gpu_num + g] = GPU_MEM_SIZE * limit
                result = milp(
                    c=np.zeros(n * gpu_num),
                    integrality=np.ones(n * gpu_num),
                    bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix.tocsc(), lb, ub),
                    options={"time_limit": 0.15},
                )
                return result.x if result.success else None

            # Binary search is monotone: a feasible pressure bound remains feasible
            # for every larger bound.  The final solve result is integral.
            for _ in range(11):
                mid = (lo + hi) / 2
                candidate = feasible(mid)
                if candidate is None:
                    lo = mid
                else:
                    hi, best_x = mid, candidate

            if best_x is not None:
                answer = {g: [] for g in range(gpu_num)}
                for i, g in enumerate(np.argmax(best_x.reshape(n, gpu_num), axis=1)):
                    answer[int(g)].append(models[i])
                if all(sum(m.model_size for m in answer[g]) <= GPU_MEM_SIZE + 1e-7
                       for g in answer):
                    placement = answer
        except Exception:
            pass

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
