GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Compute a memory-feasible placement minimizing maximum KVPR."""

    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining = [float(GPU_MEM_SIZE) for _ in range(gpu_num)]
    pressure = [0.0 for _ in range(gpu_num)]

    def kvpr(load, free_memory):
        if free_memory <= 0:
            return float("inf") if load > 0 else 0.0
        return load / free_memory

    def maximum_kvpr():
        return max(kvpr(pressure[gpu_id], remaining[gpu_id])
                   for gpu_id in range(gpu_num))

    # Place high-pressure models first so their large impact is balanced early.
    sorted_models = sorted(
        models,
        key=lambda model: (model.req_rate / model.slo, model.model_size),
        reverse=True,
    )

    for model in sorted_models:
        model_pressure = model.req_rate / model.slo
        best_gpu = None
        best_score = float("inf")

        for gpu_id in range(gpu_num):
            if model.model_size > remaining[gpu_id]:
                continue

            candidate_score = 0.0
            for other_gpu in range(gpu_num):
                if other_gpu == gpu_id:
                    score = kvpr(
                        pressure[other_gpu] + model_pressure,
                        remaining[other_gpu] - model.model_size,
                    )
                else:
                    score = kvpr(pressure[other_gpu], remaining[other_gpu])
                candidate_score = max(candidate_score, score)

            if candidate_score < best_score:
                best_score = candidate_score
                best_gpu = gpu_id

        if best_gpu is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining}"
            )

        placement[best_gpu].append(model)
        remaining[best_gpu] -= model.model_size
        pressure[best_gpu] += model_pressure

    # Use the complete pressure profile as the objective.  This preserves the
    # minimum maximum KVPR and also removes ties that otherwise block useful
    # exchanges.
    def pressure_profile(candidate_remaining, candidate_pressure):
        return tuple(sorted(
            (kvpr(candidate_pressure[g], candidate_remaining[g])
             for g in range(gpu_num)),
            reverse=True,
        ))

    # Greedy placement can leave a hotspot requiring either a relocation or a
    # simultaneous reverse two-for-one exchange.
    while True:
        current_profile = pressure_profile(remaining, pressure)
        best_move = None
        best_profile = current_profile

        for source_gpu in range(gpu_num):
            for model in placement[source_gpu]:
                model_pressure = model.req_rate / model.slo
                for target_gpu in range(gpu_num):
                    if target_gpu == source_gpu or model.model_size > remaining[target_gpu]:
                        continue

                    trial_remaining = remaining[:]
                    trial_pressure = pressure[:]
                    trial_remaining[source_gpu] += model.model_size
                    trial_remaining[target_gpu] -= model.model_size
                    trial_pressure[source_gpu] -= model_pressure
                    trial_pressure[target_gpu] += model_pressure
                    candidate_profile = pressure_profile(
                        trial_remaining, trial_pressure
                    )

                    if candidate_profile < best_profile:
                        best_profile = candidate_profile
                        best_move = (model, source_gpu, target_gpu, model_pressure)

        if best_move is not None:
            model, source_gpu, target_gpu, model_pressure = best_move
            placement[source_gpu].remove(model)
            placement[target_gpu].append(model)
            remaining[source_gpu] += model.model_size
            remaining[target_gpu] -= model.model_size
            pressure[source_gpu] -= model_pressure
            pressure[target_gpu] += model_pressure
            continue

        # A bottleneck may need to release two models before it can accept a
        # larger model held by another GPU.  This is the reverse of a
        # one-for-two exchange and is restricted to bottlenecks for speed.
        bottleneck = current_profile[0]
        bottleneck_gpus = [
            g for g in range(gpu_num)
            if abs(kvpr(pressure[g], remaining[g]) - bottleneck) <= 1e-12
        ]
        best_exchange = None
        best_profile = current_profile

        for source_gpu in bottleneck_gpus:
            source_models = placement[source_gpu]
            for left in range(len(source_models)):
                first = source_models[left]
                first_pressure = first.req_rate / first.slo
                for right in range(left + 1, len(source_models)):
                    second = source_models[right]
                    second_pressure = second.req_rate / second.slo
                    released_size = first.model_size + second.model_size
                    released_pressure = first_pressure + second_pressure

                    for target_gpu in range(gpu_num):
                        if target_gpu == source_gpu:
                            continue
                        for incoming in placement[target_gpu]:
                            if (incoming.model_size >
                                    remaining[source_gpu] + released_size):
                                continue
                            if (released_size >
                                    remaining[target_gpu] + incoming.model_size):
                                continue

                            incoming_pressure = (
                                incoming.req_rate / incoming.slo
                            )
                            trial_remaining = remaining[:]
                            trial_pressure = pressure[:]
                            trial_remaining[source_gpu] += (
                                released_size - incoming.model_size
                            )
                            trial_remaining[target_gpu] += (
                                incoming.model_size - released_size
                            )
                            trial_pressure[source_gpu] += (
                                incoming_pressure - released_pressure
                            )
                            trial_pressure[target_gpu] += (
                                released_pressure - incoming_pressure
                            )
                            candidate_profile = pressure_profile(
                                trial_remaining, trial_pressure
                            )

                            if candidate_profile < best_profile:
                                best_profile = candidate_profile
                                best_exchange = (
                                    source_gpu, target_gpu, first, second,
                                    incoming, first_pressure, second_pressure,
                                    incoming_pressure,
                                )

        if best_exchange is None:
            break

        (source_gpu, target_gpu, first, second, incoming, first_pressure,
         second_pressure, incoming_pressure) = best_exchange
        placement[source_gpu].remove(first)
        placement[source_gpu].remove(second)
        placement[target_gpu].remove(incoming)
        placement[source_gpu].append(incoming)
        placement[target_gpu].extend((first, second))
        released_size = first.model_size + second.model_size
        remaining[source_gpu] += released_size - incoming.model_size
        remaining[target_gpu] += incoming.model_size - released_size
        released_pressure = first_pressure + second_pressure
        pressure[source_gpu] += incoming_pressure - released_pressure
        pressure[target_gpu] += released_pressure - incoming_pressure

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