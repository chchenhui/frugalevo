GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Try size-, pressure-, and density-first packings, then locally minimize max KVPR."""
    best_result = None
    best_score = float("inf")

    orders = (
        lambda m: (m.model_size, m.req_rate / m.slo),
        lambda m: (m.req_rate / m.slo, m.model_size),
        lambda m: (m.req_rate / m.slo / m.model_size, m.model_size),
    )

    for order in orders:
        placement = {i: [] for i in range(gpu_num)}
        free = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num

        for model in sorted(models, key=order, reverse=True):
            pressure = model.req_rate / model.slo
            dst = min(
                (i for i in range(gpu_num) if model.model_size <= free[i]),
                key=lambda i: load[i] / free[i] if free[i] else float("inf"),
                default=None,
            )
            if dst is None:
                placement = None
                break
            placement[dst].append(model)
            free[dst] -= model.model_size
            load[dst] += pressure

        if placement is None:
            continue

        for _ in range(len(models)):
            ratios = [load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num)]
            score, action = max(ratios), None

            for src in range(gpu_num):
                for model in placement[src]:
                    size = model.model_size
                    pressure = model.req_rate / model.slo
                    for dst in range(gpu_num):
                        if src == dst or size > free[dst]:
                            continue
                        remain = free[dst] - size
                        candidate = max(
                            (load[src] - pressure) / (free[src] + size) if i == src
                            else (load[dst] + pressure) / remain if i == dst and remain
                            else float("inf") if i == dst else ratios[i]
                            for i in range(gpu_num)
                        )
                        if candidate < score:
                            score, action = candidate, ("m", src, dst, model, size, pressure)

            for src in range(gpu_num):
                for dst in range(src + 1, gpu_num):
                    for left in placement[src]:
                        for right in placement[dst]:
                            ls, rs = left.model_size, right.model_size
                            lp, rp = left.req_rate / left.slo, right.req_rate / right.slo
                            sf, df = free[src] + ls - rs, free[dst] + rs - ls
                            if sf <= 0 or df <= 0:
                                continue
                            candidate = max(
                                (load[src] - lp + rp) / sf if i == src
                                else (load[dst] - rp + lp) / df if i == dst
                                else ratios[i]
                                for i in range(gpu_num)
                            )
                            if candidate < score:
                                score, action = candidate, ("s", src, dst, left, right, ls, rs, lp, rp)

            if action is None:
                break
            if action[0] == "m":
                _, src, dst, model, size, pressure = action
                placement[src].remove(model)
                placement[dst].append(model)
                free[src] += size
                free[dst] -= size
                load[src] -= pressure
                load[dst] += pressure
            else:
                _, src, dst, left, right, ls, rs, lp, rp = action
                placement[src].remove(left)
                placement[dst].remove(right)
                placement[src].append(right)
                placement[dst].append(left)
                free[src] += ls - rs
                free[dst] += rs - ls
                load[src] += rp - lp
                load[dst] += lp - rp

        score = max(load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num))
        if score < best_score:
            best_score, best_result = score, placement

    if best_result is None:
        raise ValueError("Models do not fit in available GPU memory")
    return best_result

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
