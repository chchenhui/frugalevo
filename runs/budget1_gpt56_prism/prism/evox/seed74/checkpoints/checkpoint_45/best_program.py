GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place pressure-heavy models first, then exhaustively apply improving moves and feasible swaps."""
    placement = {i: [] for i in range(gpu_num)}
    free = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    for model in sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True):
        size, pressure = model.model_size, model.req_rate / model.slo
        best = min(
            (i for i in range(gpu_num) if size <= free[i]),
            key=lambda i: load[i] / free[i] if free[i] else float("inf"),
            default=None,
        )
        if best is None:
            raise ValueError("Models do not fit in available GPU memory")
        placement[best].append(model)
        free[best] -= size
        load[best] += pressure

    def improve_moves():
        """Apply best feasible single-model moves that strictly lower maximum KVPR."""
        for _ in range(len(models)):
            ratios = [load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num)]
            score, move = max(ratios), None
            for src in range(gpu_num):
                for model in placement[src]:
                    size, pressure = model.model_size, model.req_rate / model.slo
                    for dst in range(gpu_num):
                        if src == dst or size > free[dst]:
                            continue
                        dst_free = free[dst] - size
                        candidate = max(
                            (load[src] - pressure) / (free[src] + size) if i == src
                            else (load[dst] + pressure) / dst_free if i == dst and dst_free
                            else float("inf") if i == dst
                            else ratios[i]
                            for i in range(gpu_num)
                        )
                        if candidate < score:
                            score, move = candidate, (src, dst, model, size, pressure)
            if move is None:
                return
            src, dst, model, size, pressure = move
            placement[src].remove(model)
            placement[dst].append(model)
            free[src] += size
            free[dst] -= size
            load[src] -= pressure
            load[dst] += pressure

    improve_moves()

    # Swaps can unlock improvements when neither useful relocation fits alone.
    for _ in range(len(models)):
        ratios = [load[i] / free[i] if free[i] else float("inf") for i in range(gpu_num)]
        score, swap = max(ratios), None
        for src in range(gpu_num):
            for dst in range(src + 1, gpu_num):
                for left in placement[src]:
                    ls, lp = left.model_size, left.req_rate / left.slo
                    for right in placement[dst]:
                        rs, rp = right.model_size, right.req_rate / right.slo
                        sf, df = free[src] + ls - rs, free[dst] + rs - ls
                        if sf < 0 or df < 0:
                            continue
                        candidate = max(
                            (load[src] - lp + rp) / sf if i == src and sf
                            else (load[dst] - rp + lp) / df if i == dst and df
                            else float("inf") if i == src or i == dst
                            else ratios[i]
                            for i in range(gpu_num)
                        )
                        if candidate < score:
                            score, swap = candidate, (src, dst, left, right, sf, df, lp, rp)
        if swap is None:
            break
        src, dst, left, right, sf, df, lp, rp = swap
        placement[src].remove(left)
        placement[dst].remove(right)
        placement[src].append(right)
        placement[dst].append(left)
        free[src], free[dst] = sf, df
        load[src] += rp - lp
        load[dst] += lp - rp

    improve_moves()
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
