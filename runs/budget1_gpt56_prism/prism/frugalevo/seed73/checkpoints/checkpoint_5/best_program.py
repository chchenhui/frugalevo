GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Deterministic multi-seed construction plus bounded compound local search."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    n = len(models)
    size = [m.model_size for m in models]
    pressure = [m.req_rate / m.slo for m in models]

    if any(s > GPU_MEM_SIZE for s in size):
        raise ValueError("A model is larger than a GPU")

    def ratio(w, u):
        remain = GPU_MEM_SIZE - u
        return w / remain if remain > 0 else float("inf")

    def objective(used, weight):
        # The full descending vector makes plateau moves useful and deterministic.
        return tuple(sorted(
            (ratio(weight[g], used[g]) for g in range(gpu_num)),
            reverse=True,
        ))

    def construct(order):
        bins = [[] for _ in range(gpu_num)]
        used = [0.0] * gpu_num
        weight = [0.0] * gpu_num
        for k in order:
            best_g = -1
            best_key = None
            for g in range(gpu_num):
                if used[g] + size[k] > GPU_MEM_SIZE:
                    continue
                nu = used[:]
                nw = weight[:]
                nu[g] += size[k]
                nw[g] += pressure[k]
                rs = [ratio(nw[j], nu[j]) for j in range(gpu_num)]
                # Compare the complete resulting KVPR profile.  This matches
                # the local-search objective and lets construction reduce the
                # next bottleneck when the maximum is tied.
                key = (tuple(sorted(rs, reverse=True)), -used[g])
                if best_key is None or key < best_key:
                    best_key, best_g = key, g
            if best_g < 0:
                return None
            bins[best_g].append(k)
            used[best_g] += size[k]
            weight[best_g] += pressure[k]
        return bins, used, weight

    # Different constructions are useful because memory packing can block an
    # otherwise pressure-balanced greedy placement.
    orders = [
        sorted(range(n), key=lambda k: (size[k], pressure[k]), reverse=True),
        sorted(range(n), key=lambda k: (pressure[k], size[k]), reverse=True),
        sorted(range(n), key=lambda k: (size[k] * pressure[k], size[k]), reverse=True),
        sorted(range(n), key=lambda k: (
            pressure[k] / size[k] if size[k] else float("inf"), pressure[k]
        ), reverse=True),
    ]

    seeds = [x for x in (construct(order) for order in orders) if x is not None]
    if not seeds:
        raise ValueError("Unable to place all models on the available GPUs")

    def improve(state):
        bins, used, weight = state
        current = objective(used, weight)

        def accept_if_better():
            nonlocal current
            trial = objective(used, weight)
            if trial < current:
                current = trial
                return True
            return False

        # First improvement is deliberate: it bounds work while lexicographic
        # acceptance prevents getting stuck on an equal maximum plateau.
        for _ in range(40):
            changed = False

            for a in range(gpu_num):
                for x in list(bins[a]):
                    for b in range(gpu_num):
                        if a == b or used[b] + size[x] > GPU_MEM_SIZE:
                            continue
                        bins[a].remove(x)
                        bins[b].append(x)
                        used[a] -= size[x]
                        used[b] += size[x]
                        weight[a] -= pressure[x]
                        weight[b] += pressure[x]
                        if accept_if_better():
                            changed = True
                            break
                        bins[b].remove(x)
                        bins[a].append(x)
                        used[b] -= size[x]
                        used[a] += size[x]
                        weight[b] -= pressure[x]
                        weight[a] += pressure[x]
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue

            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for x in list(bins[a]):
                        for y in list(bins[b]):
                            if (used[a] - size[x] + size[y] > GPU_MEM_SIZE or
                                    used[b] - size[y] + size[x] > GPU_MEM_SIZE):
                                continue
                            bins[a].remove(x)
                            bins[b].remove(y)
                            bins[a].append(y)
                            bins[b].append(x)
                            used[a] += size[y] - size[x]
                            used[b] += size[x] - size[y]
                            weight[a] += pressure[y] - pressure[x]
                            weight[b] += pressure[x] - pressure[y]
                            if accept_if_better():
                                changed = True
                                break
                            bins[a].remove(y)
                            bins[b].remove(x)
                            bins[a].append(x)
                            bins[b].append(y)
                            used[a] += size[x] - size[y]
                            used[b] += size[y] - size[x]
                            weight[a] += pressure[x] - pressure[y]
                            weight[b] += pressure[y] - pressure[x]
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
            if changed:
                continue

            # A 2-for-1 exchange can cross a capacity/pressure barrier that
            # neither a relocation nor a pairwise swap can cross.
            checks = 0
            top_sources = sorted(
                range(gpu_num),
                key=lambda g: ratio(weight[g], used[g]),
                reverse=True,
            )[:2]
            for a in top_sources:
                for b in range(gpu_num):
                    if a == b:
                        continue
                    aa = list(bins[a])
                    bb = list(bins[b])
                    for ix in range(len(aa)):
                        for iz in range(ix + 1, len(aa)):
                            x, z = aa[ix], aa[iz]
                            for y in bb:
                                checks += 1
                                if checks > 20000:
                                    break
                                new_a = used[a] - size[x] - size[z] + size[y]
                                new_b = used[b] - size[y] + size[x] + size[z]
                                if new_a > GPU_MEM_SIZE or new_b > GPU_MEM_SIZE:
                                    continue
                                bins[a].remove(x)
                                bins[a].remove(z)
                                bins[b].remove(y)
                                bins[a].append(y)
                                bins[b].append(x)
                                bins[b].append(z)
                                used[a], used[b] = new_a, new_b
                                weight[a] += pressure[y] - pressure[x] - pressure[z]
                                weight[b] += pressure[x] + pressure[z] - pressure[y]
                                if accept_if_better():
                                    changed = True
                                    break
                                bins[a].remove(y)
                                bins[b].remove(x)
                                bins[b].remove(z)
                                bins[a].append(x)
                                bins[a].append(z)
                                bins[b].append(y)
                                used[a] = used[a] + size[x] + size[z] - size[y]
                                used[b] = used[b] - size[x] - size[z] + size[y]
                                weight[a] += pressure[x] + pressure[z] - pressure[y]
                                weight[b] += pressure[y] - pressure[x] - pressure[z]
                            if changed or checks > 20000:
                                break
                        if changed or checks > 20000:
                            break
                    if changed or checks > 20000:
                        break
                if changed or checks > 20000:
                    break

            if not changed:
                break
        return bins, used, weight

    best = None
    best_obj = None
    for seed in seeds:
        candidate = improve(seed)
        value = objective(candidate[1], candidate[2])
        if best_obj is None or value < best_obj:
            best, best_obj = candidate, value

    return {g: [models[k] for k in best[0][g]] for g in range(gpu_num)}

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):
        results = compute_model_placement(gpu_num, gpu_models)
        all_kvpr.append(safe_float(calculate_kvcache_pressure(results)))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr
    print(f"Max KVPR: {avg_kvpr:.3f}")