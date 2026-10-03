GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Try pressure-first and size-first packing, repair each, and return the lowest max-KVPR placement."""
    def build(order):
        placement = {i: [] for i in range(gpu_num)}
        free, load = [GPU_MEM_SIZE] * gpu_num, [0.0] * gpu_num
        for model in order:
            best = min(
                (i for i in range(gpu_num) if model.model_size <= free[i]),
                key=lambda i: load[i] / free[i] if free[i] else float("inf"),
                default=None,
            )
            if best is None:
                return None
            placement[best].append(model)
            free[best] -= model.model_size
            load[best] += model.req_rate / model.slo

        for _ in range(2):
            ratios = [load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num)]
            current, move = max(ratios), None
            for src in range(gpu_num):
                for model in placement[src]:
                    p, size = model.req_rate / model.slo, model.model_size
                    for dst in range(gpu_num):
                        if src == dst or size > free[dst]:
                            continue
                        score = max(
                            (load[i] - p) / (free[i] + size) if i == src
                            else (load[i] + p) / (free[i] - size) if i == dst
                            else ratios[i] for i in range(gpu_num)
                        )
                        if score < current:
                            current, move = score, (src, dst, model, p, size)
            if move is None:
                break
            src, dst, model, p, size = move
            placement[src].remove(model)
            placement[dst].append(model)
            free[src] += size
            free[dst] -= size
            load[src] -= p
            load[dst] += p
        return placement, max(load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num))

    candidates = [
        build(sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)),
        build(sorted(models, key=lambda m: (m.model_size, m.req_rate / m.slo), reverse=True)),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise ValueError("Models do not fit in available GPU memory")
    return min(candidates, key=lambda candidate: candidate[1])[0]

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
