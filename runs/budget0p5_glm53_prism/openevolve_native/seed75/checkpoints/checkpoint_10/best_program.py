GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement with a strong local search. Each start uses
    a different model ordering; models are greedily assigned to the GPU
    yielding the lowest resulting KVPR. The placement is then refined by
    trying single-model moves between any GPU pair and swaps involving the
    worst GPU, always accepting changes that reduce the max KVPR. The best
    placement over all starts is returned.
    """

    def build(order):
        placement = {g: [] for g in range(gpu_num)}
        mem = [GPU_MEM_SIZE] * gpu_num
        load = [0.0] * gpu_num
        for m in order:
            best_idx, best = None, float('inf')
            for g in range(gpu_num):
                rem = mem[g] - m.model_size
                if rem > 1e-9:
                    r = (load[g] + m.req_rate / m.slo) / rem
                    if r < best:
                        best, best_idx = r, g
            if best_idx is None:
                raise ValueError("Model does not fit on any GPU.")
            placement[best_idx].append(m)
            mem[best_idx] -= m.model_size
            load[best_idx] += m.req_rate / m.slo
        return placement, mem, load

    def kvpr(mem, load, g):
        return load[g] / mem[g] if mem[g] > 1e-9 else float('inf')

    def refine(placement, mem, load):
        def kmax():
            return max(kvpr(mem, load, g) for g in range(gpu_num))

        def move(m, a, b):
            placement[a].remove(m); placement[b].append(m)
            mem[a] += m.model_size; mem[b] -= m.model_size
            load[a] -= m.req_rate / m.slo; load[b] += m.req_rate / m.slo

        for _ in range(100):
            improved = False
            # try moving any model to any other GPU
            for a in range(gpu_num):
                for m in list(placement[a]):
                    for b in range(gpu_num):
                        if b == a or mem[b] - m.model_size <= 1e-9:
                            continue
                        old = kmax()
                        move(m, a, b)
                        if kmax() < old - 1e-12:
                            improved = True
                            break
                        move(m, b, a)
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            # try swapping any model pair between any two GPUs
            for a in range(gpu_num):
                for b in range(a + 1, gpu_num):
                    for m in list(placement[a]):
                        for o in list(placement[b]):
                            if (mem[a] + m.model_size - o.model_size <= 1e-9
                                    or mem[b] + o.model_size - m.model_size <= 1e-9):
                                continue
                            old = kmax()
                            move(m, a, b); move(o, b, a)
                            if kmax() < old - 1e-12:
                                improved = True
                                break
                            move(o, a, b); move(m, b, a)
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return placement

    import random
    r = lambda m: m.req_rate / m.slo
    orders = [
        sorted(models, key=r, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) * m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) + m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) - m.model_size, reverse=True),
        list(models),
        list(reversed(models)),
    ]
    # randomized restarts to escape structured local optima
    for _ in range(5):
        o = list(models)
        random.shuffle(o)
        orders.append(o)
    best_placement, best_max = None, float('inf')
    for order in orders:
        placement, mem, load = build(order)
        placement = refine(placement, mem, load)
        cur = max(kvpr(mem, load, g) for g in range(gpu_num))
        if cur < best_max:
            best_max, best_placement = cur, placement
    return best_placement

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
