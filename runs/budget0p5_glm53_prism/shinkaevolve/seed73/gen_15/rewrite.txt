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

    class GPUState:
        __slots__ = ("models", "remaining_mem", "load")
        def __init__(self):
            self.models = []
            self.remaining_mem = GPU_MEM_SIZE
            self.load = 0.0  # sum of req_rate / slo

        def kvpr(self):
            if self.remaining_mem <= 0:
                return float('inf')
            return self.load / self.remaining_mem

    def build_placement(sorted_models, use_post_kvpr):
        gpus = [GPUState() for _ in range(gpu_num)]
        for model in sorted_models:
            req_per_slo = model.req_rate / model.slo
            best_idx, best_val = None, None
            for i, g in enumerate(gpus):
                if model.model_size >= g.remaining_mem:
                    continue
                if use_post_kvpr:
                    val = (g.load + req_per_slo) / (g.remaining_mem - model.model_size)
                else:
                    val = g.kvpr()
                if best_val is None or val < best_val:
                    best_val, best_idx = val, i
            if best_idx is None:
                return None
            g = gpus[best_idx]
            g.models.append(model)
            g.remaining_mem -= model.model_size
            g.load += req_per_slo
        return gpus

    def max_kvpr(gpus):
        return max(g.kvpr() for g in gpus)

    def refine(gpus, rounds=200):
        """Best-improvement local search: move models off the hottest GPU,
        choosing the move that minimizes the new global max KVPR."""
        if gpus is None:
            return None
        for _ in range(rounds):
            kvs = [g.kvpr() for g in gpus]
            hot = max(range(len(gpus)), key=lambda i: kvs[i])
            hot_kvpr = kvs[hot]
            second_kvpr = max((kvs[g] for g in range(len(gpus)) if g != hot), default=0.0)

            best_move = None
            best_new_max = hot_kvpr - 1e-12
            for model in gpus[hot].models:
                rps = model.req_rate / model.slo
                src_after = (gpus[hot].load - rps) / (gpus[hot].remaining_mem + model.model_size)
                for j, g in enumerate(gpus):
                    if j == hot:
                        continue
                    if model.model_size >= g.remaining_mem:
                        continue
                    dst_after = (g.load + rps) / (g.remaining_mem - model.model_size)
                    new_max = max(second_kvpr, src_after, dst_after)
                    if new_max < best_new_max:
                        best_new_max = new_max
                        best_move = (model, j)

            if best_move is None:
                break
            model, j = best_move
            rps = model.req_rate / model.slo
            gpus[hot].models.remove(model)
            gpus[hot].remaining_mem += model.model_size
            gpus[hot].load -= rps
            g = gpus[j]
            g.models.append(model)
            g.remaining_mem -= model.model_size
            g.load += rps
        return gpus

    def swap_refine(gpus, rounds=50):
        """Try swapping models between the hottest GPU and others."""
        if gpus is None:
            return None
        for _ in range(rounds):
            kvs = [g.kvpr() for g in gpus]
            hot = max(range(len(gpus)), key=lambda i: kvs[i])
            hot_kvpr = kvs[hot]
            second_kvpr = max((kvs[g] for g in range(len(gpus)) if g != hot), default=0.0)

            best_swap = None
            best_new_max = hot_kvpr - 1e-12
            for m1 in gpus[hot].models:
                r1 = m1.req_rate / m1.slo
                for j, g in enumerate(gpus):
                    if j == hot:
                        continue
                    for m2 in g.models:
                        r2 = m2.req_rate / m2.slo
                        # memory feasibility after swap
                        if gpus[hot].remaining_mem + m1.model_size - m2.model_size <= 0:
                            continue
                        if g.remaining_mem + m2.model_size - m1.model_size <= 0:
                            continue
                        hot_after = (gpus[hot].load - r1 + r2) / (
                            gpus[hot].remaining_mem + m1.model_size - m2.model_size)
                        other_after = (g.load - r2 + r1) / (
                            g.remaining_mem + m2.model_size - m1.model_size)
                        new_max = max(second_kvpr, hot_after, other_after)
                        if new_max < best_new_max:
                            best_new_max = new_max
                            best_swap = (m1, m2, j)

            if best_swap is None:
                break
            m1, m2, j = best_swap
            g = gpus[j]
            gpus[hot].models.remove(m1)
            gpus[hot].models.append(m2)
            g.models.remove(m2)
            g.models.append(m1)
            gpus[hot].load += m2.req_rate / m2.slo - m1.req_rate / m1.slo
            gpus[hot].remaining_mem += m1.model_size - m2.model_size
            g.load += m1.req_rate / m1.slo - m2.req_rate / m2.slo
            g.remaining_mem += m2.model_size - m1.model_size
        return gpus

    # multiple candidate strategies
    key_rs = lambda m: m.req_rate / m.slo
    key_size = lambda m: m.model_size
    key_density = lambda m: key_rs(m) / m.model_size

    candidates_specs = [
        (sorted(models, key=key_rs, reverse=True), True),
        (sorted(models, key=key_rs, reverse=True), False),
        (sorted(models, key=key_size, reverse=True), True),
        (sorted(models, key=key_density, reverse=True), True),
        (sorted(models, key=key_size, reverse=False), True),
        (sorted(models, key=key_density, reverse=False), True),
    ]

    best_gpus = None
    best_score = float('inf')
    for sorted_models, use_post in candidates_specs:
        gpus = build_placement(sorted_models, use_post)
        if gpus is None:
            continue
        gpus = refine(gpus)
        gpus = swap_refine(gpus)
        gpus = refine(gpus)
        if gpus is None:
            continue
        score = max_kvpr(gpus)
        if score < best_score:
            best_score = score
            best_gpus = gpus

    if best_gpus is None:
        raise ValueError("Unable to place all models within GPU memory.")

    return {gpu_id: best_gpus[gpu_id].models for gpu_id in range(gpu_num)}

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