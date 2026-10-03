GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Exactly solve feasible subset set-partitioning with MILP for small inputs, else use local search."""
    # A configuration is a feasible set of models placed on one GPU.  Selecting
    # configurations is a set-partitioning problem: every model must be covered
    # once, no more than gpu_num configurations may be selected, and z bounds
    # every selected configuration's exact KVPR.
    n = len(models)

    # Exact minimax solver for small instances.  For a threshold T, a subset is
    # valid precisely when size < 80 and rate + T * size <= 80 * T.  The DP
    # partitions the remaining models into the fewest valid unlabeled GPU groups.
    # Choosing the first unassigned model removes GPU and subset-order symmetry.
    if 0 < n <= 18 and gpu_num > 0:
        try:
            from functools import lru_cache

            sizes = [float(m.model_size) for m in models]
            rates = [float(m.req_rate) / m.slo for m in models]
            total_masks = 1 << n
            used_size = [0.0] * total_masks
            used_rate = [0.0] * total_masks
            subsets = []

            for mask in range(1, total_masks):
                bit = mask & -mask
                item = bit.bit_length() - 1
                previous = mask ^ bit
                used_size[mask] = used_size[previous] + sizes[item]
                used_rate[mask] = used_rate[previous] + rates[item]
                if used_size[mask] < GPU_MEM_SIZE:
                    subsets.append(mask)

            # Do not spend exponential DP time on unusually unconstrained cases.
            if subsets and len(subsets) <= 180000:
                pressure = {
                    mask: used_rate[mask] / (GPU_MEM_SIZE - used_size[mask])
                    for mask in subsets
                }
                thresholds = sorted(set(pressure.values()))
                by_item = [[] for _ in range(n)]
                for mask in subsets:
                    for i in range(n):
                        if mask & (1 << i):
                            by_item[i].append(mask)

                full = total_masks - 1

                def partition(limit, keep_choice=False):
                    options = []
                    for i in range(n):
                        choices = [
                            mask for mask in by_item[i]
                            if pressure[mask] <= limit + 1e-12
                        ]
                        # Large groups first commonly prove feasibility quickly.
                        choices.sort(key=int.bit_count, reverse=True)
                        options.append(choices)

                    choice = {}

                    @lru_cache(maxsize=250000)
                    def needed(remaining):
                        if not remaining:
                            return 0
                        first = (remaining & -remaining).bit_length() - 1
                        best = gpu_num + 1
                        selected = None
                        for group in options[first]:
                            if group & remaining != group:
                                continue
                            value = 1 + needed(remaining ^ group)
                            if value < best:
                                best, selected = value, group
                                if best == 1:
                                    break
                        if keep_choice and selected is not None:
                            choice[remaining] = selected
                        return best

                    count = needed(full)
                    return count, choice

                low, high = 0, len(thresholds) - 1
                answer_limit = None
                while low <= high:
                    middle = (low + high) // 2
                    count, _ = partition(thresholds[middle])
                    if count <= gpu_num:
                        answer_limit = thresholds[middle]
                        high = middle - 1
                    else:
                        low = middle + 1

                if answer_limit is not None:
                    _, choices = partition(answer_limit, True)
                    answer = {gpu: [] for gpu in range(gpu_num)}
                    remaining = full
                    gpu = 0
                    while remaining:
                        group = choices[remaining]
                        for i, model in enumerate(models):
                            if group & (1 << i):
                                answer[gpu].append(model)
                        remaining ^= group
                        gpu += 1
                    return answer
        except (MemoryError, RecursionError):
            # Large cache growth falls through to the robust heuristic below.
            pass

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

    # Beam search keeps several globally promising partial packings rather than
    # committing to one greedy insertion trajectory.  GPU-label permutations are
    # removed by deduplicating sorted (used_memory, load) signatures.
    import heapq

    if gpu_num <= 0:
        raise ValueError("At least one GPU is required")

    size = [m.model_size for m in models]
    rate = [m.req_rate / m.slo for m in models]
    order = sorted(
        range(n),
        key=lambda i: (size[i], rate[i] / max(GPU_MEM_SIZE - size[i], 1e-9)),
        reverse=True,
    )
    width = 160
    states = [(tuple([0.0] * gpu_num), tuple([0.0] * gpu_num),
               tuple(() for _ in range(gpu_num)))]

    for item in order:
        candidates = {}
        for used, loads, groups in states:
            for g in range(gpu_num):
                if used[g] + size[item] >= GPU_MEM_SIZE:
                    continue
                new_used = list(used)
                new_loads = list(loads)
                new_groups = list(groups)
                new_used[g] += size[item]
                new_loads[g] += rate[item]
                new_groups[g] = new_groups[g] + (item,)
                free = [GPU_MEM_SIZE - x for x in new_used]
                # Both terms are valid lower bounds on the final bottleneck.
                rank = max(
                    max(new_loads[j] / free[j] for j in range(gpu_num)),
                    sum(rate) / sum(free),
                )
                state = (tuple(new_used), tuple(new_loads), tuple(new_groups))
                signature = tuple(sorted(zip(state[0], state[1])))
                if signature not in candidates or rank < candidates[signature][0]:
                    candidates[signature] = (rank, state)

        # Keep the smallest ranks with an explicit bounded max-heap.
        heap = []
        for rank, state in candidates.values():
            entry = (-rank, state)
            heapq.heappush(heap, entry)
            if len(heap) > width:
                heapq.heappop(heap)
        states = [entry[1] for entry in heap]

        if not states:
            break

    if states:
        used, load, groups = min(
            states,
            key=lambda s: max(
                s[1][g] / (GPU_MEM_SIZE - s[0][g]) for g in range(gpu_num)
            ),
        )
        placement = {
            g: [models[i] for i in groups[g]]
            for g in range(gpu_num)
        }
        remaining = [GPU_MEM_SIZE - x for x in used]
        load = list(load)
    else:
        # Guaranteed simple fallback when a narrow beam loses a feasible prefix.
        placement = {i: [] for i in range(gpu_num)}
        remaining = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
            weight, model_size = model.req_rate / model.slo, model.model_size
            gpu = min(
                (i for i in range(gpu_num) if model_size < remaining[i]),
                key=lambda i: (load[i] + weight) / (remaining[i] - model_size),
                default=None,
            )
            if gpu is None:
                raise ValueError(f"Unable to place model of size {model_size} GB")
            placement[gpu].append(model)
            load[gpu] += weight
            remaining[gpu] -= model_size

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

    # For medium instances, use integer differential evolution to explore
    # coordinated reassignments that moves and swaps cannot reach.
    n = len(models)
    if 1 < gpu_num and 15 < n <= 28:
        try:
            import numpy as np
            from scipy.optimize import differential_evolution

            sizes = np.array([m.model_size for m in models], dtype=float)
            rates = np.array([m.req_rate / m.slo for m in models], dtype=float)
            index = {id(m): i for i, m in enumerate(models)}
            base = np.zeros(n, dtype=int)
            for g, group in placement.items():
                for m in group:
                    base[index[id(m)]] = g

            def pack(order):
                """Create a feasible demand-aware assignment for DE initialization."""
                rem = [float(GPU_MEM_SIZE)] * gpu_num
                loads = [0.0] * gpu_num
                result = np.empty(n, dtype=int)
                for i in order:
                    choices = [g for g in range(gpu_num) if rem[g] > sizes[i]]
                    if not choices:
                        return base.copy()
                    g = min(choices, key=lambda x: (loads[x] + rates[i]) / (rem[x] - sizes[i]))
                    result[i] = g
                    rem[g] -= sizes[i]
                    loads[g] += rates[i]
                return result

            demand_order = sorted(range(n), key=lambda i: rates[i], reverse=True)
            size_order = sorted(range(n), key=lambda i: sizes[i], reverse=True)
            density_order = sorted(range(n), key=lambda i: rates[i] / sizes[i], reverse=True)
            seeds = [base, pack(demand_order), pack(size_order), pack(density_order)]
            rng = np.random.default_rng(0)
            for _ in range(8):
                # Different orderings provide feasible, diverse starting points.
                seeds.append(pack(sorted(range(n), key=lambda i: rates[i] * rng.uniform(.7, 1.3), reverse=True)))

            def objective(x):
                """Return exact peak KVPR, with a finite penalty for invalid memory."""
                assignment = np.asarray(x, dtype=int)
                used = np.bincount(assignment, weights=sizes, minlength=gpu_num)
                overflow = np.maximum(used - (GPU_MEM_SIZE - 1e-9), 0.0).sum()
                if overflow:
                    return 1e9 + overflow * 1e7
                loads = np.bincount(assignment, weights=rates, minlength=gpu_num)
                return float(np.max(loads / (GPU_MEM_SIZE - used)))

            result = differential_evolution(
                objective,
                [(0, gpu_num - 1)] * n,
                integrality=[True] * n,
                init=np.asarray(seeds, dtype=float),
                seed=0,
                maxiter=25,
                polish=False,
                updating="immediate",
                workers=1,
                tol=.01,
            )
            assignment = np.asarray(result.x, dtype=int)
            candidate = objective(assignment)
            current = max(load[i] / remaining[i] for i in range(gpu_num))
            if candidate < current:
                placement = {g: [] for g in range(gpu_num)}
                for i, g in enumerate(assignment):
                    placement[int(g)].append(models[i])
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
