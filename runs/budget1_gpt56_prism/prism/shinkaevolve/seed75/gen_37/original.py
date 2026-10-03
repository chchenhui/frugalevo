GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Place models by binary-searching a maximum KVPR threshold.

    For a target pressure T, a GPU with total demand W and total model
    memory M satisfies:

        W / (GPU_MEM_SIZE - M) <= T

    exactly when:

        W + T * M <= T * GPU_MEM_SIZE.

    Thus each model has transformed size demand + T * model_size, subject
    to both transformed capacity and physical memory capacity.
    """
    if gpu_num <= 0:
        if models:
            raise ValueError("Cannot place models without GPUs.")
        return {}

    model_list = list(models)
    if not model_list:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    EPS = 1e-10
    sizes = [model.model_size for model in model_list]
    demands = [model.req_rate / model.slo for model in model_list]

    if any(size > GPU_MEM_SIZE + EPS for size in sizes):
        raise ValueError("A model is larger than GPU memory.")

    total_size = sum(sizes)
    total_demand = sum(demands)

    if total_size > gpu_num * GPU_MEM_SIZE + EPS:
        raise ValueError("Total model memory exceeds available GPU memory.")

    def max_pressure(loads, remaining):
        result = 0.0
        for load, memory in zip(loads, remaining):
            if memory <= EPS:
                if load > EPS:
                    return float("inf")
            else:
                result = max(result, load / memory)
        return result

    def pack_at_threshold(threshold):
        """
        Try several deterministic multidimensional best-fit orders.
        Returns the best feasible packing found, or None.
        """
        transformed = [
            demands[i] + threshold * sizes[i]
            for i in range(len(model_list))
        ]
        transformed_capacity = threshold * GPU_MEM_SIZE

        orders = [
            sorted(
                range(len(model_list)),
                key=lambda i: (
                    max(
                        sizes[i] / GPU_MEM_SIZE,
                        transformed[i] / max(transformed_capacity, EPS),
                    ),
                    transformed[i],
                    sizes[i],
                ),
                reverse=True,
            ),
            sorted(
                range(len(model_list)),
                key=lambda i: (sizes[i], demands[i]),
                reverse=True,
            ),
            sorted(
                range(len(model_list)),
                key=lambda i: (demands[i], sizes[i]),
                reverse=True,
            ),
        ]

        best_state = None
        best_value = float("inf")

        for order in orders:
            placement = {gpu_id: [] for gpu_id in range(gpu_num)}
            loads = [0.0] * gpu_num
            remaining = [GPU_MEM_SIZE] * gpu_num
            transformed_used = [0.0] * gpu_num
            feasible = True

            for index in order:
                size = sizes[index]
                demand = demands[index]
                cost = transformed[index]

                best_gpu = None
                best_key = None

                for gpu_id in range(gpu_num):
                    new_remaining = remaining[gpu_id] - size
                    new_cost = transformed_used[gpu_id] + cost
                    new_load = loads[gpu_id] + demand

                    if new_remaining < -EPS:
                        continue
                    if new_cost > transformed_capacity + EPS:
                        continue

                    # A positive load must retain KV-cache memory.
                    if new_load > EPS and new_remaining <= EPS:
                        continue

                    # Multidimensional best fit: fill the tightest compatible
                    # GPU, preserving roomy GPUs for future large models.
                    key = (
                        max(
                            new_cost / max(transformed_capacity, EPS),
                            1.0 - new_remaining / GPU_MEM_SIZE,
                        ),
                        new_cost / max(transformed_capacity, EPS),
                        -new_remaining,
                        gpu_id,
                    )

                    if best_key is None or key > best_key:
                        best_key = key
                        best_gpu = gpu_id

                if best_gpu is None:
                    feasible = False
                    break

                placement[best_gpu].append(model_list[index])
                loads[best_gpu] += demand
                remaining[best_gpu] -= size
                transformed_used[best_gpu] += cost

            if feasible:
                value = max_pressure(loads, remaining)
                if value < best_value:
                    best_value = value
                    best_state = (placement, loads, remaining)

        return best_state

    # Analytical lower bounds for the minimax pressure.
    remaining_total = gpu_num * GPU_MEM_SIZE - total_size
    if remaining_total <= EPS and total_demand > EPS:
        raise ValueError("No KV-cache memory remains after model placement.")

    lower = 0.0
    if total_demand > EPS:
        lower = total_demand / max(remaining_total, EPS)

    for size, demand in zip(sizes, demands):
        if demand > EPS:
            if size >= GPU_MEM_SIZE - EPS:
                raise ValueError("A positive-demand model leaves no KV-cache memory.")
            lower = max(lower, demand / (GPU_MEM_SIZE - size))

    # Find a feasible upper bound.  The threshold packing is retried while
    # increasing T because larger T relaxes the transformed capacity.
    upper = max(lower * 1.25, 1e-8)
    best = pack_at_threshold(upper)

    for _ in range(50):
        if best is not None:
            break
        upper *= 2.0
        best = pack_at_threshold(upper)

    if best is None:
        raise ValueError("Unable to construct a memory-feasible GPU placement.")

    # Binary search the smallest threshold for which the multidimensional
    # packing procedure finds a feasible assignment.
    for _ in range(36):
        middle = (lower + upper) / 2.0
        candidate = pack_at_threshold(middle)

        if candidate is not None:
            upper = middle
            best = candidate
        else:
            lower = middle

    return best[0]

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