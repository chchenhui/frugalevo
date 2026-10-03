GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily place difficult models, then lexicographically reduce GPU KVPRs with moves and swaps."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def pressure(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    def objective(changes=()):
        loads, memory = load[:], remaining[:]
        for g, dw, dm in changes:
            loads[g] += dw
            memory[g] += dm
        return tuple(sorted((pressure(loads[g], memory[g]) for g in range(gpu_num)), reverse=True))

    ordered = sorted(
        models,
        key=lambda m: pressure(m.req_rate / m.slo, GPU_MEM_SIZE - m.model_size),
        reverse=True,
    )
    for model in ordered:
        size, weight = model.model_size, model.req_rate / model.slo
        choices = [g for g in range(gpu_num) if remaining[g] >= size]
        if not choices:
            raise ValueError("Models cannot fit in the available GPU memory")
        best = min(choices, key=lambda g: pressure(load[g] + weight, remaining[g] - size))
        placement[best].append(model)
        load[best] += weight
        remaining[best] -= size

    for _ in range(min(16, len(models))):
        best_value, best_action = objective(), None
        for source in range(gpu_num):
            for model in placement[source]:
                size, weight = model.model_size, model.req_rate / model.slo
                for target in range(gpu_num):
                    if target == source or remaining[target] < size:
                        continue
                    value = objective(((source, -weight, size), (target, weight, -size)))
                    if value < best_value:
                        best_value, best_action = value, ("move", source, target, model)

        for source in range(gpu_num):
            for target in range(source + 1, gpu_num):
                for left in placement[source]:
                    ls, lw = left.model_size, left.req_rate / left.slo
                    for right in placement[target]:
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if remaining[source] + ls < rs or remaining[target] + rs < ls:
                            continue
                        value = objective(((source, rw - lw, ls - rs),
                                           (target, lw - rw, rs - ls)))
                        if value < best_value:
                            best_value, best_action = value, ("swap", source, target, left, right)

        if best_action is None:
            break
        if best_action[0] == "move":
            _, source, target, model = best_action
            size, weight = model.model_size, model.req_rate / model.slo
            placement[source].remove(model)
            placement[target].append(model)
            load[source] -= weight
            load[target] += weight
            remaining[source] += size
            remaining[target] -= size
        else:
            _, source, target, left, right = best_action
            ls, lw = left.model_size, left.req_rate / left.slo
            rs, rw = right.model_size, right.req_rate / right.slo
            placement[source].remove(left)
            placement[target].remove(right)
            placement[source].append(right)
            placement[target].append(left)
            load[source] += rw - lw
            load[target] += lw - rw
            remaining[source] += ls - rs
            remaining[target] += rs - ls

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
