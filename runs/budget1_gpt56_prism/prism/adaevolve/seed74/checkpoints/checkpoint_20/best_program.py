GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Exactly solve feasible subset set-partitioning with MILP for small inputs, else use local search."""
    # A configuration is a feasible set of models placed on one GPU.  Selecting
    # configurations is a set-partitioning problem: every model must be covered
    # once, no more than gpu_num configurations may be selected, and z bounds
    # every selected configuration's exact KVPR.
    n = len(models)
    if n <= 15:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import csc_matrix

            size = [m.model_size for m in models]
            rate = [m.req_rate / m.slo for m in models]
            total_size = [0.0] * (1 << n)
            total_rate = [0.0] * (1 << n)
            masks, pressures = [0], [0.0]  # Empty configuration represents unused GPUs.

            for mask in range(1, 1 << n):
                bit = mask & -mask
                i = bit.bit_length() - 1
                previous = mask ^ bit
                total_size[mask] = total_size[previous] + size[i]
                total_rate[mask] = total_rate[previous] + rate[i]
                if total_size[mask] < GPU_MEM_SIZE:
                    masks.append(mask)
                    pressures.append(total_rate[mask] / (GPU_MEM_SIZE - total_size[mask]))

            # A model that cannot form even a singleton has no feasible placement.
            if all(any(mask & (1 << i) for mask in masks) for i in range(n)):
                count = len(masks)
                rows, cols, data = [], [], []

                for j, mask in enumerate(masks):
                    for i in range(n):
                        if mask & (1 << i):
                            rows.append(i)
                            cols.append(j)
                            data.append(1.0)
                    rows.append(n)
                    cols.append(j)
                    data.append(1.0)
                    # pressure[j] * x[j] <= z
                    rows.extend((n + 1 + j, n + 1 + j))
                    cols.extend((j, count))
                    data.extend((pressures[j], -1.0))

                constraints = LinearConstraint(
                    csc_matrix((data, (rows, cols)), shape=(n + 1 + count, count + 1)),
                    np.r_[np.ones(n), 0.0, np.full(count, -np.inf)],
                    np.r_[np.ones(n), float(gpu_num), np.zeros(count)],
                )
                objective = np.zeros(count + 1)
                objective[-1] = 1.0
                result = milp(
                    objective,
                    integrality=np.r_[np.ones(count), 0],
                    bounds=Bounds(np.zeros(count + 1), np.r_[np.ones(count), np.inf]),
                    constraints=constraints,
                    options={"disp": False},
                )

                if result.success:
                    chosen = [j for j in range(count) if masks[j] and result.x[j] > 0.5]
                    covered = sum(masks[j] for j in chosen)
                    if len(chosen) <= gpu_num and covered == (1 << n) - 1:
                        answer = {i: [] for i in range(gpu_num)}
                        for gpu, j in enumerate(chosen):
                            for i, model in enumerate(models):
                                if masks[j] & (1 << i):
                                    answer[gpu].append(model)
                        return answer
        except Exception:
            # Preserve the fast heuristic when scipy is unavailable or an exact
            # solve cannot be completed.
            pass

    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        weight, size = model.req_rate / model.slo, model.model_size
        gpu = min(
            (i for i in range(gpu_num) if size < remaining[i]),
            key=lambda i: (load[i] + weight) / (remaining[i] - size),
            default=None,
        )
        if gpu is None:
            raise ValueError(f"Unable to place model of size {size} GB")
        placement[gpu].append(model)
        load[gpu] += weight
        remaining[gpu] -= size

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
