GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Binary-search a KVPR target and pack models using its equivalent bin weight."""
    weights = [m.req_rate / m.slo for m in models]

    # KVPR <= target is equivalent to:
    # sum(weight + target * model_size) <= target * GPU_MEM_SIZE.
    def pack(target):
        orders = (
            sorted(range(len(models)), key=lambda i: weights[i] + target * models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: weights[i], reverse=True),
        )
        for order in orders:
            placement = {i: [] for i in range(gpu_num)}
            memory = [GPU_MEM_SIZE] * gpu_num
            load = [0.0] * gpu_num
            valid = True
            for index in order:
                model, weight = models[index], weights[index]
                choices = [
                    i for i in range(gpu_num)
                    if memory[i] > model.model_size
                    and load[i] + weight <= target * (memory[i] - model.model_size) + 1e-10
                ]
                if not choices:
                    valid = False
                    break
                gpu = min(choices, key=lambda i: target * (memory[i] - model.model_size) - load[i] - weight)
                placement[gpu].append(model)
                memory[gpu] -= model.model_size
                load[gpu] += weight
            if valid:
                return placement, memory, load
        return None

    low = max((w / (GPU_MEM_SIZE - m.model_size) for m, w in zip(models, weights)), default=0.0)
    high = max(low, 1.0)
    result = pack(high)
    while result is None:
        high *= 2
        result = pack(high)

    for _ in range(28):
        middle = (low + high) / 2
        candidate = pack(middle)
        if candidate is None:
            low = middle
        else:
            high, result = middle, candidate

    placement, memory, load = result

    # A final relocation pass improves the heuristic packing without costly swaps.
    while True:
        current = max(load[i] / memory[i] for i in range(gpu_num))
        best = None
        for src in range(gpu_num):
            for model in placement[src]:
                weight = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if src == dst or memory[dst] <= model.model_size:
                        continue
                    value = max(
                        (load[i] / memory[i] if i != src and i != dst else
                         ((load[src] - weight) / (memory[src] + model.model_size) if i == src else
                          (load[dst] + weight) / (memory[dst] - model.model_size)))
                        for i in range(gpu_num)
                    )
                    if value < current - 1e-12:
                        current, best = value, (src, dst, model, weight)
        if best is None:
            return placement
        src, dst, model, weight = best
        placement[src].remove(model)
        placement[dst].append(model)
        memory[src] += model.model_size
        memory[dst] -= model.model_size
        load[src] -= weight
        load[dst] += weight

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
