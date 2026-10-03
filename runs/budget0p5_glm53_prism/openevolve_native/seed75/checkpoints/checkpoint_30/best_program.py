GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Multi-start greedy placement with a worst-GPU-targeted local search.
    Models are greedily placed (over several orderings) onto the GPU that
    yields the lowest resulting KVPR. Refinement repeatedly identifies the
    GPU with maximum KVPR and tries moving one of its models to another GPU
    (or swapping with another GPU's model) whenever that strictly lowers
    the global max KVPR. Best placement across starts is returned.
    """
    import random

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

    def kmax(mem, load):
        return max(kvpr(mem, load, g) for g in range(gpu_num))

    def refine(placement, mem, load):
        """Local search targeting the worst (and second-worst) GPUs."""
        def move(m, a, b):
            placement[a].remove(m); placement[b].append(m)
            mem[a] += m.model_size; mem[b] -= m.model_size
            load[a] -= m.req_rate / m.slo; load[b] += m.req_rate / m.slo

        while True:
            cur = kmax(mem, load)
            # worst GPU first, then second-worst (relieving it can help)
            targets = sorted(range(gpu_num), key=lambda g: -kvpr(mem, load, g))[:2]
            done = False
            for a in targets:
                # move a model off the target GPU
                for m in list(placement[a]):
                    for b in range(gpu_num):
                        if b == a or mem[b] - m.model_size <= 1e-9:
                            continue
                        move(m, a, b)
                        if kmax(mem, load) < cur - 1e-12:
                            done = True
                            break
                        move(m, b, a)
                    if done:
                        break
                if done:
                    break
                # swap a model of the target GPU with another GPU's model
                for m in list(placement[a]):
                    for b in range(gpu_num):
                        if b == a:
                            continue
                        for o in list(placement[b]):
                            if (mem[a] + m.model_size - o.model_size <= 1e-9
                                    or mem[b] + o.model_size - m.model_size <= 1e-9):
                                continue
                            move(m, a, b); move(o, b, a)
                            if kmax(mem, load) < cur - 1e-12:
                                done = True
                                break
                            move(o, a, b); move(m, b, a)
                        if done:
                            break
                    if done:
                        break
                if done:
                    break
            if not done:
                break
        return placement

    r = lambda m: m.req_rate / m.slo
    orders = [
        sorted(models, key=r, reverse=True),
        sorted(models, key=lambda m: m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) / m.model_size, reverse=True),
        sorted(models, key=lambda m: r(m) * m.model_size, reverse=True),
        list(models),
    ]
    for _ in range(30):
        o = list(models)
        random.shuffle(o)
        orders.append(o)
    best_placement, best_max = None, float('inf')
    for order in orders:
        placement, mem, load = build(order)
        placement = refine(placement, mem, load)
        cur = kmax(mem, load)
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
