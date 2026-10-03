GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use multi-start greedy placement and local search, then bounded exact minimax refinement."""
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
        sorted(models, key=lambda m: weight(m) / (GPU_MEM_SIZE - m.model_size) ** 2, reverse=True),
        sorted(models, key=lambda m: weight(m) * m.model_size / (GPU_MEM_SIZE - m.model_size), reverse=True),
        sorted(models, key=lambda m: weight(m), reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: weight(m) * m.model_size, reverse=True),
    )
    answer = None

    for order in orders:
        placed, free, load = build(order)
        for _ in range(60):
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

    # Try direct best-fit packings for progressively tighter KVPR targets.
    # At target T, a GPU is valid exactly when
    # sum(weight + T * model_size) <= T * GPU_MEM_SIZE.
    # This can find coordinated re-packings that single moves and swaps miss.
    best_score, best_placed = answer
    total_weight = sum(weight(m) for m in models)
    total_size = sum(m.model_size for m in models)
    lower = max(
        max(weight(m) / (GPU_MEM_SIZE - m.model_size) for m in models),
        total_weight / (gpu_num * GPU_MEM_SIZE - total_size),
    )
    upper = best_score

    for _ in range(14):
        target = (lower + upper) / 2
        capacity = target * GPU_MEM_SIZE
        remaining = [capacity] * gpu_num
        free = [GPU_MEM_SIZE] * gpu_num
        candidate = {g: [] for g in range(gpu_num)}
        valid = True

        for m in sorted(
            models,
            key=lambda m: weight(m) + target * m.model_size,
            reverse=True,
        ):
            need = weight(m) + target * m.model_size
            choices = [
                g for g in range(gpu_num)
                if remaining[g] + 1e-12 >= need and free[g] > m.model_size
            ]
            if not choices:
                valid = False
                break
            g = min(
                choices,
                key=lambda g: (remaining[g] - need, free[g] - m.model_size),
            )
            candidate[g].append(m)
            remaining[g] -= need
            free[g] -= m.model_size

        if valid:
            score = max(
                sum(weight(m) for m in candidate[g]) / free[g]
                for g in range(gpu_num)
            )
            if score < best_score:
                best_score, best_placed = score, candidate
            upper = target
        else:
            lower = target

    answer = best_score, best_placed

    # For a proposed bound T, every GPU must satisfy:
    #   sum(weight + T * model_size) <= T * GPU_MEM_SIZE.
    # This is a bin-packing feasibility problem.  On small instances, a
    # bounded branch-and-bound search can improve placements that require
    # multiple coordinated moves and therefore cannot be reached by swaps.
    if len(models) <= 16:
        best_score, best_placed = answer
        total_weight = sum(weight(m) for m in models)
        total_size = sum(m.model_size for m in models)
        low = max(
            [weight(m) / (GPU_MEM_SIZE - m.model_size) for m in models]
            + [total_weight / (gpu_num * GPU_MEM_SIZE - total_size)]
        )
        high = best_score

        def feasible(target, limit=12000):
            items = sorted(
                models,
                key=lambda m: weight(m) + target * m.model_size,
                reverse=True,
            )
            capacity = target * GPU_MEM_SIZE
            room = [GPU_MEM_SIZE] * gpu_num
            remaining = [capacity] * gpu_num
            result = {g: [] for g in range(gpu_num)}
            nodes = [0]

            def search(i):
                if i == len(items):
                    return True
                nodes[0] += 1
                if nodes[0] > limit:
                    return None

                m = items[i]
                need = weight(m) + target * m.model_size
                seen = set()
                choices = sorted(range(gpu_num), key=lambda g: remaining[g] - need)
                for g in choices:
                    state = (remaining[g], room[g])
                    if state in seen:
                        continue
                    seen.add(state)
                    if remaining[g] + 1e-12 < need or room[g] <= m.model_size:
                        continue
                    remaining[g] -= need
                    room[g] -= m.model_size
                    result[g].append(m)
                    found = search(i + 1)
                    if found:
                        return True
                    result[g].pop()
                    room[g] += m.model_size
                    remaining[g] += need
                    if found is None:
                        return None
                return False

            found = search(0)
            return found, result

        # Only treat an infeasible bound as a lower bound when the search
        # completed; a node-limit interruption never discards the incumbent.
        for _ in range(12):
            target = (low + high) / 2
            found, candidate = feasible(target)
            if found is None:
                break
            if found:
                free = [
                    GPU_MEM_SIZE - sum(m.model_size for m in candidate[g])
                    for g in range(gpu_num)
                ]
                loads = [sum(weight(m) for m in candidate[g]) for g in range(gpu_num)]
                score = max(loads[g] / free[g] for g in range(gpu_num))
                if score < best_score:
                    best_score, best_placed = score, candidate
                high = target
            else:
                low = target

        answer = best_score, best_placed

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
