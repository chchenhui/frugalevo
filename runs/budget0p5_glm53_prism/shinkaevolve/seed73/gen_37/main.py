GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes resulting KVPR while fitting in memory
    for model in sorted_models:
        best_idx = None
        best_ratio = float('inf')

        for gpu_id in range(gpu_num):
            remaining_mem = shared_kv[gpu_id] - model.model_size
            if remaining_mem > 0:
                # KVPR of the GPU after placing this model on it
                new_ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / remaining_mem
                if new_ratio < best_ratio:
                    best_ratio = new_ratio
                    best_idx = gpu_id

        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # 4) Local search refinement: safe single-model moves (index-based, no list.remove)
    def gpu_kvpr(gpu_id):
        mem_left = GPU_MEM_SIZE - sum(m.model_size for m in placement[gpu_id])
        if mem_left <= 0:
            return float('inf')
        return weighted_req_rate[gpu_id] / mem_left

    def max_kvpr():
        return max(gpu_kvpr(g) for g in range(gpu_num))

    import copy
    import random

    def snapshot():
        return ([list(placement[g]) for g in range(gpu_num)],
                list(weighted_req_rate), list(shared_kv))

    def restore(snap):
        nonlocal placement, weighted_req_rate, shared_kv
        pl, wr, sk = snap
        placement = {g: list(pl[g]) for g in range(gpu_num)}
        weighted_req_rate = list(wr)
        shared_kv = list(sk)

    def local_search():
        # Local search with moves and pairwise swaps; accept equal scores
        # occasionally to allow escaping flat basins.
        nonlocal placement, weighted_req_rate, shared_kv
        improved = True
        tie_budget = 3  # allow a few equal-score acceptances per pass
        while improved:
            improved = False
            best_max = max_kvpr()
            for src in range(gpu_num):
                if improved:
                    break
                for mi in range(len(placement[src])):
                    if improved:
                        break
                    model = placement[src][mi]
                    w = model.req_rate / model.slo
                    for dst in range(gpu_num):
                        if dst == src:
                            continue
                        if model.model_size > shared_kv[dst]:
                            continue
                        # Apply move tentatively
                        placement[src].pop(mi)
                        placement[dst].append(model)
                        weighted_req_rate[src] -= w
                        weighted_req_rate[dst] += w
                        shared_kv[src] += model.model_size
                        shared_kv[dst] -= model.model_size
                        new_max = max_kvpr()
                        if new_max < best_max - 1e-12 or (new_max <= best_max + 1e-12 and tie_budget > 0):
                            if new_max <= best_max + 1e-12:
                                tie_budget -= 1
                            improved = True
                            break
                        else:
                            # Revert
                            placement[dst].pop()
                            placement[src].insert(mi, model)
                            weighted_req_rate[dst] -= w
                            weighted_req_rate[src] += w
                            shared_kv[dst] += model.model_size
                            shared_kv[src] -= model.model_size
            if improved:
                continue
            # Try pairwise swaps between GPUs
            for src in range(gpu_num):
                if improved:
                    break
                for dst in range(src + 1, gpu_num):
                    if improved:
                        break
                    for si in range(len(placement[src])):
                        if improved:
                            break
                        for di in range(len(placement[dst])):
                            ms = placement[src][si]
                            md = placement[dst][di]
                            ws = ms.req_rate / ms.slo
                            wd = md.req_rate / md.slo
                            # Feasibility of the swap
                            if (shared_kv[src] + ms.model_size - md.model_size) < 0:
                                continue
                            if (shared_kv[dst] + md.model_size - ms.model_size) < 0:
                                continue
                            # Apply swap
                            placement[src][si] = md
                            placement[dst][di] = ms
                            weighted_req_rate[src] += wd - ws
                            weighted_req_rate[dst] += ws - wd
                            shared_kv[src] += ms.model_size - md.model_size
                            shared_kv[dst] += md.model_size - ms.model_size
                            if max_kvpr() < best_max - 1e-12:
                                improved = True
                                break
                            else:
                                # Revert
                                placement[src][si] = ms
                                placement[dst][di] = md
                                weighted_req_rate[src] -= wd - ws
                                weighted_req_rate[dst] -= ws - wd
                                shared_kv[src] -= ms.model_size - md.model_size
                                shared_kv[dst] -= md.model_size - ms.model_size

        if improved:
            continue
        # 3-cycle rotation among the hottest GPUs:
        # move A: hot -> g1, B: g1 -> g2, C: g2 -> hot
        order = sorted(range(gpu_num), key=lambda g: -gpu_kvpr(g))
        if len(order) >= 3:
            hot, g1, g2 = order[0], order[1], order[2]
            best_max = max_kvpr()
            for hi in range(len(placement[hot])):
                if improved:
                    break
                A = placement[hot][hi]
                wa = A.req_rate / A.slo
                if A.model_size > shared_kv[g1]:
                    continue
                # Tentatively move A: hot -> g1
                placement[hot].pop(hi)
                placement[g1].append(A)
                weighted_req_rate[hot] -= wa
                weighted_req_rate[g1] += wa
                shared_kv[hot] += A.model_size
                shared_kv[g1] -= A.model_size
                for i1 in range(len(placement[g1])):
                    if improved:
                        break
                    B = placement[g1][i1]
                    wb = B.req_rate / B.slo
                    if B.model_size > shared_kv[g2]:
                        continue
                    # Tentatively move B: g1 -> g2
                    placement[g1].pop(i1)
                    placement[g2].append(B)
                    weighted_req_rate[g1] -= wb
                    weighted_req_rate[g2] += wb
                    shared_kv[g1] += B.model_size
                    shared_kv[g2] -= B.model_size
                    for i2 in range(len(placement[g2])):
                        C = placement[g2][i2]
                        wc = C.req_rate / C.slo
                        if C.model_size > shared_kv[hot]:
                            continue
                        # Apply C: g2 -> hot
                        placement[g2].pop(i2)
                        placement[hot].append(C)
                        weighted_req_rate[g2] -= wc
                        weighted_req_rate[hot] += wc
                        shared_kv[g2] += C.model_size
                        shared_kv[hot] -= C.model_size
                        if max_kvpr() < best_max - 1e-12:
                            improved = True
                            break
                        # Revert C
                        placement[hot].pop()
                        placement[g2].insert(i2, C)
                        weighted_req_rate[hot] -= wc
                        weighted_req_rate[g2] += wc
                        shared_kv[hot] += C.model_size
                        shared_kv[g2] -= C.model_size
                    if improved:
                        break
                    # Revert B
                    placement[g2].pop()
                    placement[g1].insert(i1, B)
                    weighted_req_rate[g2] -= wb
                    weighted_req_rate[g1] += wb
                    shared_kv[g2] += B.model_size
                    shared_kv[g1] -= B.model_size
                if improved:
                    break
                # Revert A
                placement[g1].pop()
                placement[hot].insert(hi, A)
                weighted_req_rate[g1] -= wa
                weighted_req_rate[hot] += wa
                shared_kv[g1] += A.model_size
                shared_kv[hot] -= A.model_size

    local_search()
    best_snap = snapshot()
    best_score = max_kvpr()

    # ILS: perturb then re-optimize, keep the best solution
    rng = random.Random(0)
    for _ in range(8):
        # Perturbation: try to move up to 2 random models to random feasible GPUs
        moves_done = 0
        all_items = [(g, mi) for g in range(gpu_num) for mi in range(len(placement[g]))]
        rng.shuffle(all_items)
        for g, mi in all_items:
            if moves_done >= 2:
                break
            if mi >= len(placement[g]):
                continue
            model = placement[g][mi]
            dst = rng.randrange(gpu_num)
            if dst == g or model.model_size > shared_kv[dst]:
                continue
            w = model.req_rate / model.slo
            placement[g].pop(mi)
            placement[dst].append(model)
            weighted_req_rate[g] -= w
            weighted_req_rate[dst] += w
            shared_kv[g] += model.model_size
            shared_kv[dst] -= model.model_size
            moves_done += 1
        if moves_done == 0:
            break
        local_search()
        cur_score = max_kvpr()
        if cur_score < best_score - 1e-12:
            best_score = cur_score
            best_snap = snapshot()
        else:
            restore(best_snap)

    restore(best_snap)
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