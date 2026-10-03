GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a feasible placement minimizing the maximum GPU KV cache pressure.

    Uses several deterministic greedy constructions and improves every feasible
    construction with strict local search (moves, swaps, and short chains).
    """

    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    models = list(models)
    n = len(models)

    sizes = [m.model_size for m in models]
    weights = [m.req_rate / m.slo for m in models]

    for size in sizes:
        if size > GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place model of size {size} GB on a {GPU_MEM_SIZE} GB GPU."
            )

    def objective(used, load):
        best = 0.0
        for g in range(gpu_num):
            remaining = GPU_MEM_SIZE - used[g]
            if remaining < 0:
                return float("inf")
            if remaining == 0:
                if load[g] > 0:
                    return float("inf")
                continue
            value = load[g] / remaining
            if value > best:
                best = value
        return best

    def greedy(order):
        assigned = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        load = [0.0] * gpu_num

        for i in order:
            best_gpu = None
            best_key = None

            for g in range(gpu_num):
                if used[g] + sizes[i] > GPU_MEM_SIZE + 1e-12:
                    continue

                new_used = used[g] + sizes[i]
                new_load = load[g] + weights[i]
                remaining = GPU_MEM_SIZE - new_used
                new_pressure = (
                    float("inf") if remaining <= 0 and new_load > 0
                    else (new_load / remaining if remaining > 0 else 0.0)
                )

                maximum = new_pressure
                for h in range(gpu_num):
                    if h != g:
                        rem = GPU_MEM_SIZE - used[h]
                        pressure = load[h] / rem if rem > 0 else float("inf")
                        if pressure > maximum:
                            maximum = pressure

                # Prefer lower resulting global pressure, then lower local
                # pressure, then more remaining capacity for deterministic ties.
                key = (maximum, new_pressure, -remaining, g)
                if best_key is None or key < best_key:
                    best_key = key
                    best_gpu = g

            if best_gpu is None:
                return None

            assigned[best_gpu].append(i)
            used[best_gpu] += sizes[i]
            load[best_gpu] += weights[i]

        return assigned, used, load

    def improve(state):
        assigned, used, load = state
        current = objective(used, load)
        eps = 1e-12
        rounds = 0

        while rounds < 120:
            rounds += 1
            best_value = current
            best_action = None

            # Single-model moves.
            for src in range(gpu_num):
                for i in list(assigned[src]):
                    for dst in range(gpu_num):
                        if src == dst or used[dst] + sizes[i] > GPU_MEM_SIZE + eps:
                            continue

                        used[src] -= sizes[i]
                        load[src] -= weights[i]
                        used[dst] += sizes[i]
                        load[dst] += weights[i]
                        value = objective(used, load)
                        used[dst] -= sizes[i]
                        load[dst] -= weights[i]
                        used[src] += sizes[i]
                        load[src] += weights[i]

                        if value + eps < best_value:
                            best_value = value
                            best_action = ("move", src, dst, i)

            # Pairwise swaps.
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for i in list(assigned[a]):
                        for j in list(assigned[b]):
                            new_a = used[a] - sizes[i] + sizes[j]
                            new_b = used[b] - sizes[j] + sizes[i]
                            if new_a > GPU_MEM_SIZE + eps or new_b > GPU_MEM_SIZE + eps:
                                continue

                            used[a], used[b] = new_a, new_b
                            load[a] += weights[j] - weights[i]
                            load[b] += weights[i] - weights[j]
                            value = objective(used, load)
                            load[a] += weights[i] - weights[j]
                            load[b] += weights[j] - weights[i]
                            used[a] = used[a] - sizes[j] + sizes[i]
                            used[b] = used[b] - sizes[i] + sizes[j]

                            if value + eps < best_value:
                                best_value = value
                                best_action = ("swap", a, b, i, j)

            # Two-step chain: move i from a to b and j from b to c.
            # This can unlock improvements unavailable to a direct move.
            for a in range(gpu_num):
                for b in range(gpu_num):
                    if a == b:
                        continue
                    for i in list(assigned[a]):
                        for j in list(assigned[b]):
                            if i == j:
                                continue
                            if used[b] - sizes[j] + sizes[i] > GPU_MEM_SIZE + eps:
                                continue
                            for c in range(gpu_num):
                                if c == a or c == b:
                                    continue
                                if used[c] + sizes[j] > GPU_MEM_SIZE + eps:
                                    continue

                                used[a] -= sizes[i]
                                load[a] -= weights[i]
                                used[b] += sizes[i] - sizes[j]
                                load[b] += weights[i] - weights[j]
                                used[c] += sizes[j]
                                load[c] += weights[j]

                                value = objective(used, load)

                                load[c] -= weights[j]
                                used[c] -= sizes[j]
                                load[b] -= weights[i] - weights[j]
                                used[b] -= sizes[i] - sizes[j]
                                load[a] += weights[i]
                                used[a] += sizes[i]

                                if value + eps < best_value:
                                    best_value = value
                                    best_action = ("chain", a, b, c, i, j)

            if best_action is None:
                break

            kind = best_action[0]
            if kind == "move":
                _, src, dst, i = best_action
                assigned[src].remove(i)
                assigned[dst].append(i)
                used[src] -= sizes[i]
                load[src] -= weights[i]
                used[dst] += sizes[i]
                load[dst] += weights[i]

            elif kind == "swap":
                _, a, b, i, j = best_action
                assigned[a].remove(i)
                assigned[b].remove(j)
                assigned[a].append(j)
                assigned[b].append(i)
                used[a] += sizes[j] - sizes[i]
                load[a] += weights[j] - weights[i]
                used[b] += sizes[i] - sizes[j]
                load[b] += weights[i] - weights[j]

            else:
                _, a, b, c, i, j = best_action
                assigned[a].remove(i)
                assigned[b].remove(j)
                assigned[b].append(i)
                assigned[c].append(j)
                used[a] -= sizes[i]
                load[a] -= weights[i]
                used[b] += sizes[i] - sizes[j]
                load[b] += weights[i] - weights[j]
                used[c] += sizes[j]
                load[c] += weights[j]

            current = best_value

        return assigned, used, load

    indices = list(range(n))
    orders = [
        sorted(indices, key=lambda i: (-weights[i], -sizes[i], i)),
        sorted(indices, key=lambda i: (-sizes[i], -weights[i], i)),
        sorted(indices, key=lambda i: (-(weights[i] / max(sizes[i], 1e-12)), -sizes[i], i)),
        sorted(indices, key=lambda i: (-(weights[i] * sizes[i]), -weights[i], i)),
        sorted(indices, key=lambda i: (-(weights[i] + sizes[i]), -sizes[i], i)),
    ]

    best_state = None
    best_score = float("inf")

    for order in orders:
        state = greedy(order)
        if state is None:
            continue

        state = improve(state)
        score = objective(state[1], state[2])

        if score < best_score:
            best_score = score
            best_state = state

    if best_state is None:
        raise ValueError(
            "Unable to place all models on the available GPUs. "
            "Total capacity or per-GPU capacity is insufficient."
        )

    assigned = best_state[0]
    return {
        gpu_id: [models[i] for i in assigned[gpu_id]]
        for gpu_id in range(gpu_num)
    }

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
