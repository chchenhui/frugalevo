GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    """

    # 1) Sort models by r_j / s_j in descending order
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}  # lists of (idx, model)
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted_req_rate = [0.0 for _ in range(gpu_num)]

    # 3) Greedy assignment
    for idx, model in enumerate(sorted_models):
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if model.model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                current_ratio = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU."
            )
        placement[best_idx].append((idx, model))
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] -= model.model_size

    # 4) Local search refinement: moves, swaps, and 3-cycle rotations
    def gpu_kvpr(gpu_id):
        mem_left = GPU_MEM_SIZE - sum(m.model_size for _, m in placement[gpu_id])
        if mem_left <= 0:
            return float('inf')
        return weighted_req_rate[gpu_id] / mem_left

    def max_kvpr():
        return max(gpu_kvpr(g) for g in range(gpu_num))

    EPS = 1e-12

    def apply_move(src, pos, dst):
        idx, model = placement[src][pos]
        placement[src].pop(pos)
        placement[dst].append((idx, model))
        r = model.req_rate / model.slo
        weighted_req_rate[src] -= r
        weighted_req_rate[dst] += r
        shared_kv[src] += model.model_size
        shared_kv[dst] -= model.model_size

    def undo_move(src, pos, dst):
        idx, model = placement[dst].pop()
        placement[src].insert(pos, (idx, model))
        r = model.req_rate / model.slo
        weighted_req_rate[src] += r
        weighted_req_rate[dst] -= r
        shared_kv[src] -= model.model_size
        shared_kv[dst] += model.model_size

    def try_move(best_max):
        # Try single-model moves off the hottest GPUs
        hot_gpus = [g for g in range(gpu_num) if gpu_kvpr(g) >= best_max - EPS]
        for src in hot_gpus:
            for pos in range(len(placement[src])):
                _, model = placement[src][pos]
                for dst in range(gpu_num):
                    if dst == src or model.model_size > shared_kv[dst]:
                        continue
                    apply_move(src, pos, dst)
                    if max_kvpr() < best_max - EPS:
                        return True
                    undo_move(src, pos, dst)
        return False

    def try_swap(best_max):
        # Try pairwise swaps between GPUs
        for src in range(gpu_num):
            if gpu_kvpr(src) < best_max - EPS:
                continue
            for dst in range(gpu_num):
                if dst == src:
                    continue
                for p1 in range(len(placement[src])):
                    for p2 in range(len(placement[dst])):
                        i1, m1 = placement[src][p1]
                        i2, m2 = placement[dst][p2]
                        delta_mem = m2.model_size - m1.model_size
                        if shared_kv[dst] + delta_mem < 0 or shared_kv[src] - delta_mem < 0:
                            continue
                        # Apply swap
                        placement[src][p1] = (i2, m2)
                        placement[dst][p2] = (i1, m1)
                        weighted_req_rate[src] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                        weighted_req_rate[dst] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
                        shared_kv[src] += delta_mem
                        shared_kv[dst] -= delta_mem
                        if max_kvpr() < best_max - EPS:
                            return True
                        # Revert swap
                        placement[src][p1] = (i1, m1)
                        placement[dst][p2] = (i2, m2)
                        weighted_req_rate[src] += m1.req_rate / m1.slo - m2.req_rate / m2.slo
                        weighted_req_rate[dst] += m2.req_rate / m2.slo - m1.req_rate / m1.slo
                        shared_kv[src] -= delta_mem
                        shared_kv[dst] += delta_mem
        return False

    def try_rotate(best_max):
        # 3-cycle rotation among the hottest GPU and other GPUs.
        # Move A (from hot) -> g2, B (from g2) -> g3, C (from g3) -> hot.
        hot = max(range(gpu_num), key=lambda g: gpu_kvpr(g))
        others = [g for g in range(gpu_num) if g != hot]
        for p_hot in range(len(placement[hot])):
            _, m_hot = placement[hot][p_hot]
            for g2 in others:
                if m_hot.model_size > shared_kv[g2]:
                    continue
                for p2 in range(len(placement[g2])):
                    _, m2 = placement[g2][p2]
                    for g3 in others:
                        if g3 == g2:
                            continue
                        # candidate cycle: hot->g2, g2->g3, g3->hot
                        for p3 in range(len(placement[g3])):
                            _, m3 = placement[g3][p3]
                            # feasibility for g2 and g3 after their changes
                            if shared_kv[g2] + m2.model_size - m_hot.model_size < 0:
                                continue
                            if shared_kv[g3] + m3.model_size - m2.model_size < 0:
                                continue
                            if shared_kv[hot] + m_hot.model_size - m3.model_size < 0:
                                continue
                            # apply cycle
                            apply_move(hot, p_hot, g2)          # now m_hot on g2
                            # locate m2 in g2 (position p2 in original g2 list)
                            # after pop/insert, indexes shift: find m2's tuple
                            pos_m2 = None
                            for q, (ii, mm) in enumerate(placement[g2]):
                                if ii == _ and mm is m2 if False else (mm is m2 and placement[g2][q][0] == placement_src_id(placement, g2, q)) :
                                    pos_m2 = q
                                    break
                            # simpler: search by object identity of model
                            pos_m2 = None
                            for q in range(len(placement[g2])):
                                if placement[g2][q][1] is m2:
                                    pos_m2 = q
                                    break
                            if pos_m2 is None:
                                undo_move(hot, p_hot, g2)
                                continue
                            apply_move(g2, pos_m2, g3)
                            pos_m3 = None
                            for q in range(len(placement[g3])):
                                if placement[g3][q][1] is m3:
                                    pos_m3 = q
                                    break
                            if pos_m3 is None:
                                # revert g2 move then hot move
                                # undo apply_move(g2, pos_m2, g3): m2 now on g3
                                undo_move(g2, len(placement[g2]), g3) if False else None
                                # manual revert: move m2 back from g3 to g2
                                for q in range(len(placement[g3])):
                                    if placement[g3][q][1] is m2:
                                        apply_move(g3, q, g2)
                                        break
                                undo_move(hot, p_hot, g2)
                                continue
                            apply_move(g3, pos_m3, hot)
                            if max_kvpr() < best_max - EPS:
                                return True
                            # revert: move m3 back hot->g3, m2 g3->g2, m_hot g2->hot
                            for q in range(len(placement[hot])):
                                if placement[hot][q][1] is m3:
                                    apply_move(hot, q, g3)
                                    break
                            for q in range(len(placement[g3])):
                                if placement[g3][q][1] is m2:
                                    apply_move(g3, q, g2)
                                    break
                            for q in range(len(placement[g2])):
                                if placement[g2][q][1] is m_hot:
                                    # this is the appended entry (last), find it
                                    pass
                            # m_hot was appended to g2 by apply_move -> it's last element
                            placement_len = len(placement[g2])
                            for q in range(placement_len - 1, -1, -1):
                                if placement[g2][q][1] is m_hot:
                                    apply_move(g2, q, hot)
                                    break
        return False

    def placement_src_id(placement, g2, q):
        return placement[g2][q][0]

    for _round in range(30):
        best_max = max_kvpr()
        if try_move(best_max):
            continue
        if try_swap(best_max):
            continue
        if try_rotate(best_max):
            continue
        break

    # Convert back to expected output format
    return {gpu_id: [m for _, m in placement[gpu_id]] for gpu_id in range(gpu_num)}

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
