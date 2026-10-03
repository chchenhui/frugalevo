GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily minimize resulting KVPR, then improve it with feasible moves and swaps."""
    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def pressure(weight, memory):
        return weight / memory if memory > 0 else float("inf")

    ordered = sorted(
        models,
        key=lambda m: pressure(m.req_rate / m.slo, GPU_MEM_SIZE - m.model_size),
        reverse=True,
    )

    for model in ordered:
        size = model.model_size
        weight = model.req_rate / model.slo
        best = min(
            (g for g in range(gpu_num) if remaining[g] >= size),
            key=lambda g: pressure(load[g] + weight, remaining[g] - size),
            default=None,
        )
        if best is None:
            raise ValueError("Models cannot fit in the available GPU memory")
        placement[best].append(model)
        load[best] += weight
        remaining[best] -= size

    # Continue improving while strictly better moves or swaps exist; extra rounds
    # help resolve placements that need several memory-balancing adjustments.
    for _ in range(min(16, len(models))):
        current = max(pressure(load[g], remaining[g]) for g in range(gpu_num))
        best_value, best_action = current, None

        for source in range(gpu_num):
            for model in placement[source]:
                size, weight = model.model_size, model.req_rate / model.slo
                for target in range(gpu_num):
                    if target == source or remaining[target] < size:
                        continue
                    value = max(
                        pressure(
                            load[g] + (weight if g == target else -weight if g == source else 0),
                            remaining[g] - (size if g == target else -size if g == source else 0),
                        )
                        for g in range(gpu_num)
                    )
                    if value < best_value:
                        best_value, best_action = value, ("move", source, target, model)

        # A swap can free memory on a crowded GPU when no beneficial relocation fits.
        for source in range(gpu_num):
            for target in range(source + 1, gpu_num):
                for left in placement[source]:
                    ls, lw = left.model_size, left.req_rate / left.slo
                    for right in placement[target]:
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if remaining[source] + ls < rs or remaining[target] + rs < ls:
                            continue
                        value = max(
                            pressure(
                                load[g] + (rw - lw if g == source else lw - rw if g == target else 0),
                                remaining[g] + (ls - rs if g == source else rs - ls if g == target else 0),
                            )
                            for g in range(gpu_num)
                        )
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
