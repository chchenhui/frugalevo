GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Binary-search KVPR bounds using several deterministic greedy packing orders."""
    weights = [m.req_rate / m.slo for m in models]

    # KVPR <= target is equivalent to:
    # sum(weight + target * model_size) <= target * GPU_MEM_SIZE.
    def pack(target):
        """Try complementary model orders and tight-pack/spread-pressure choices."""
        orders = (
            sorted(range(len(models)), key=lambda i: weights[i] + target * models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: models[i].model_size, reverse=True),
            sorted(range(len(models)), key=lambda i: weights[i], reverse=True),
            sorted(range(len(models)),
                   key=lambda i: weights[i] / (GPU_MEM_SIZE - models[i].model_size),
                   reverse=True),
            sorted(range(len(models)),
                   key=lambda i: weights[i] / models[i].model_size,
                   reverse=True),
        )
        for order in orders:
            for spread in (False, True):
                placement = {i: [] for i in range(gpu_num)}
                memory = [GPU_MEM_SIZE] * gpu_num
                load = [0.0] * gpu_num
                for index in order:
                    model, weight = models[index], weights[index]
                    choices = [
                        i for i in range(gpu_num)
                        if memory[i] > model.model_size
                        and load[i] + weight <= target * (memory[i] - model.model_size) + 1e-10
                    ]
                    if not choices:
                        break
                    if spread:
                        gpu = min(choices, key=lambda i:
                                  (load[i] + weight) / (memory[i] - model.model_size))
                    else:
                        gpu = min(choices, key=lambda i:
                                  target * (memory[i] - model.model_size) - load[i] - weight)
                    placement[gpu].append(model)
                    memory[gpu] -= model.model_size
                    load[gpu] += weight
                else:
                    return placement, memory, load

        # Greedy packing can reject a feasible lower bound.  For small
        # instances, recover such packings with bounded symmetric search.
        if len(models) <= 20:
            order = sorted(
                range(len(models)),
                key=lambda i: weights[i] + target * models[i].model_size,
                reverse=True,
            )
            remaining = [target * GPU_MEM_SIZE] * gpu_num
            memory = [GPU_MEM_SIZE] * gpu_num
            groups = [[] for _ in range(gpu_num)]
            nodes = 0

            def search(pos):
                """Place transformed items with equivalent GPUs treated symmetrically."""
                nonlocal nodes
                nodes += 1
                if nodes > 30000:
                    return False
                if pos == len(order):
                    return True

                index = order[pos]
                model = models[index]
                item = weights[index] + target * model.model_size
                seen = set()
                choices = sorted(range(gpu_num), key=lambda g: remaining[g])
                for gpu in choices:
                    state = (round(remaining[gpu], 9), round(memory[gpu], 9))
                    if state in seen:
                        continue
                    seen.add(state)
                    if remaining[gpu] + 1e-10 < item or memory[gpu] <= model.model_size:
                        continue
                    remaining[gpu] -= item
                    memory[gpu] -= model.model_size
                    groups[gpu].append(index)
                    if search(pos + 1):
                        return True
                    groups[gpu].pop()
                    memory[gpu] += model.model_size
                    remaining[gpu] += item
                return False

            if search(0):
                placement = {g: [models[i] for i in groups[g]] for g in range(gpu_num)}
                load = [sum(weights[i] for i in groups[g]) for g in range(gpu_num)]
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

    # Improve the feasible packing with the best strict move or pairwise swap.
    # Swaps are important when no single model can be relocated without making
    # the destination GPU worse, but exchanging differently shaped models helps.
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
                        current, best = value, ("move", src, dst, model, weight)

        for left in range(gpu_num):
            for right in range(left + 1, gpu_num):
                for first in placement[left]:
                    first_weight = first.req_rate / first.slo
                    for second in placement[right]:
                        second_weight = second.req_rate / second.slo
                        left_memory = memory[left] + first.model_size - second.model_size
                        right_memory = memory[right] + second.model_size - first.model_size
                        if left_memory <= 0 or right_memory <= 0:
                            continue
                        value = max(
                            (load[i] / memory[i] if i != left and i != right else
                             ((load[left] - first_weight + second_weight) / left_memory
                              if i == left else
                              (load[right] - second_weight + first_weight) / right_memory))
                            for i in range(gpu_num)
                        )
                        if value < current - 1e-12:
                            current, best = value, (
                                "swap", left, right, first, second,
                                first_weight, second_weight
                            )

        if best is None:
            return placement

        if best[0] == "move":
            _, src, dst, model, weight = best
            placement[src].remove(model)
            placement[dst].append(model)
            memory[src] += model.model_size
            memory[dst] -= model.model_size
            load[src] -= weight
            load[dst] += weight
        else:
            _, left, right, first, second, first_weight, second_weight = best
            placement[left].remove(first)
            placement[right].remove(second)
            placement[left].append(second)
            placement[right].append(first)
            memory[left] += first.model_size - second.model_size
            memory[right] += second.model_size - first.model_size
            load[left] += second_weight - first_weight
            load[right] += first_weight - second_weight

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
