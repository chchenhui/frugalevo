GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def _kvpr(w, mem):
    """KVPR = weighted req rate / free memory."""
    return w / (GPU_MEM_SIZE - mem) if mem < GPU_MEM_SIZE else float('inf')


def _greedy(gpu_num, models, key_fn):
    """One greedy construction: place models in key_fn order, each on the GPU
    minimizing the resulting KVPR (tie-break: more free memory)."""
    placement = {g: [] for g in range(gpu_num)}
    shared_kv = [0.0] * gpu_num
    weighted_req_rate = [0.0] * gpu_num
    for model in sorted(models, key=key_fn):
        best_idx = None
        best_key = (float('inf'), float('-inf'))
        for gpu_id in range(gpu_num):
            free = shared_kv[gpu_id] + model.model_size
            if free < GPU_MEM_SIZE:
                ratio = (weighted_req_rate[gpu_id] + model.req_rate / model.slo) / (GPU_MEM_SIZE - free)
                k = (ratio, -free)
                if k < best_key:
                    best_key = k
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(f"Cannot place model of size {model.model_size} GB")
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model.req_rate / model.slo
        shared_kv[best_idx] += model.model_size
    return placement, shared_kv, weighted_req_rate


_SEED_KEYS = [
    lambda m: (-(m.req_rate / m.slo),),                      # load desc (incumbent)
    lambda m: (-m.model_size,),                              # size desc
    lambda m: (-(m.req_rate / m.slo) / m.model_size,),       # load per GB desc
    lambda m: (-(m.req_rate / m.slo) * m.model_size,),       # load*size desc
    lambda m: (m.model_size, -(m.req_rate / m.slo)),         # size asc then load desc
    lambda m: (m.req_rate / m.slo,) + (m.model_size,),       # load asc (anti-greedy)
    lambda m: (-m.req_rate / m.slo, -m.model_size),          # load desc then size desc
    lambda m: (m.slo,),                                      # slo asc (tight SLO first)
    lambda m: (-m.slo, -m.model_size),                       # slo desc then size desc
    lambda m: (-m.model_size, m.req_rate / m.slo),           # size desc then load asc
]


def _max_kvpr(weighted_req_rate, shared_kv, gpu_num):
    """Global max KVPR helper."""
    return max(_kvpr(weighted_req_rate[g], shared_kv[g]) for g in range(gpu_num))


def _snapshot(placement):
    """Copy of the placement so kicks can never alias the recorded best."""
    return {g: list(ms) for g, ms in placement.items()}


def compute_model_placement(gpu_num, models):
    """
    Multistart portfolio: 10 deterministic greedy seeds, each refined by the
    move/swap local search, then a bounded perturbation phase (3 kicks per
    seed): force-move a model off the bottleneck GPU, re-run the local
    search, and restore the pre-kick snapshot unless it strictly improved.
    Snapshots guarantee the returned placement is exactly the best state
    evaluated and never worse than the single-seed baseline.
    """
    best_placement = None
    best_max = float('inf')
    for key_fn in _SEED_KEYS:
        placement, shared_kv, weighted_req_rate = _greedy(gpu_num, models, key_fn)
        _local_search(gpu_num, placement, shared_kv, weighted_req_rate)
        cur = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
        if cur < best_max:
            best_max = cur
            best_placement = _snapshot(placement)
        # Bounded deterministic kick phase.
        for kick in range(3):
            pre_placement = _snapshot(placement)
            pre_w = list(weighted_req_rate)
            pre_m = list(shared_kv)
            bottleneck = max(range(gpu_num),
                             key=lambda g: _kvpr(weighted_req_rate[g], shared_kv[g]))
            if not placement[bottleneck]:
                break
            model = placement[bottleneck][kick % len(placement[bottleneck])]
            s = model.model_size
            dests = [g for g in range(gpu_num)
                     if g != bottleneck and shared_kv[g] + s < GPU_MEM_SIZE]
            if not dests:
                break
            g2 = max(dests, key=lambda g: shared_kv[g])
            placement[bottleneck].remove(model)
            placement[g2].append(model)
            r = model.req_rate / model.slo
            weighted_req_rate[bottleneck] -= r
            weighted_req_rate[g2] += r
            shared_kv[bottleneck] -= s
            shared_kv[g2] += s
            _local_search(gpu_num, placement, shared_kv, weighted_req_rate)
            new = _max_kvpr(weighted_req_rate, shared_kv, gpu_num)
            if new < cur - 1e-12:
                cur = new
                if new < best_max:
                    best_max = new
                    best_placement = _snapshot(placement)
            else:
                # restore pre-kick state wholesale (local search may have
                # moved the kicked model, so manual undo is unsafe)
                placement = pre_placement
                weighted_req_rate = pre_w
                shared_kv = pre_m
    return best_placement


def _local_search(gpu_num, placement, shared_kv, weighted_req_rate):
    """Hill climber: move/swap models while strictly reducing max KVPR."""

    improved = True
    while improved:
        improved = False
        cur = max(_kvpr(weighted_req_rate[g], shared_kv[g]) for g in range(gpu_num))
        for g1 in range(gpu_num):
            for model in list(placement[g1]):
                # try moving model to another GPU
                for g2 in range(gpu_num):
                    if g2 == g1:
                        continue
                    if shared_kv[g2] + model.model_size < GPU_MEM_SIZE:
                        w1 = weighted_req_rate[g1] - model.req_rate / model.slo
                        w2 = weighted_req_rate[g2] + model.req_rate / model.slo
                        m1 = shared_kv[g1] - model.model_size
                        m2 = shared_kv[g2] + model.model_size
                        new_max = max(_kvpr(w1, m1), _kvpr(w2, m2),
                                      max(_kvpr(weighted_req_rate[g], shared_kv[g])
                                          for g in range(gpu_num) if g not in (g1, g2)))
                        if new_max < cur - 1e-12:
                            placement[g1].remove(model)
                            placement[g2].append(model)
                            weighted_req_rate[g1], weighted_req_rate[g2] = w1, w2
                            shared_kv[g1], shared_kv[g2] = m1, m2
                            improved = True
                            cur = new_max
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # try swaps between GPU pairs
            for g1 in range(gpu_num):
                for g2 in range(g1 + 1, gpu_num):
                    for a in placement[g1]:
                        for b in placement[g2]:
                            if (shared_kv[g1] - a.model_size + b.model_size < GPU_MEM_SIZE and
                                    shared_kv[g2] - b.model_size + a.model_size < GPU_MEM_SIZE):
                                ra, rb = a.req_rate / a.slo, b.req_rate / b.slo
                                w1 = weighted_req_rate[g1] - ra + rb
                                w2 = weighted_req_rate[g2] - rb + ra
                                m1 = shared_kv[g1] - a.model_size + b.model_size
                                m2 = shared_kv[g2] - b.model_size + a.model_size
                                new_max = max(_kvpr(w1, m1), _kvpr(w2, m2),
                                              max(_kvpr(weighted_req_rate[g], shared_kv[g])
                                                  for g in range(gpu_num) if g not in (g1, g2)))
                                if new_max < cur - 1e-12:
                                    placement[g1].remove(a)
                                    placement[g2].remove(b)
                                    placement[g1].append(b)
                                    placement[g2].append(a)
                                    weighted_req_rate[g1], weighted_req_rate[g2] = w1, w2
                                    shared_kv[g1], shared_kv[g2] = m1, m2
                                    improved = True
                                    cur = new_max
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

    return placement  # final state of hill climber (unused by caller)

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
