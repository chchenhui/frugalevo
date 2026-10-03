GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Place models while minimizing the lexicographically sorted GPU KVPR vector."""
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    records = []
    for index, model in enumerate(models):
        size = model.model_size
        if size > GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place model of size {size} GB on a {GPU_MEM_SIZE} GB GPU."
            )
        records.append((index, model, size, model.req_rate / model.slo))

    def kvpr(load, remaining):
        return load / remaining if remaining > 0 else float("inf")

    def state_key(loads, remaining):
        # The first component is maximum KVPR; later components break ties and
        # allow progress when several GPUs share the current bottleneck.
        return tuple(sorted(
            (kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)),
            reverse=True,
        ))

    def build_greedy(order):
        placement = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        remaining = [GPU_MEM_SIZE] * gpu_num

        for record_id in order:
            _, _, size, weight = records[record_id]
            choice = None

            for gpu in range(gpu_num):
                if size > remaining[gpu]:
                    continue

                new_remaining = remaining[gpu] - size
                new_pressure = kvpr(loads[gpu] + weight, new_remaining)

                # Prefer the lowest resulting pressure.  More free memory is
                # preferred on exact pressure ties to reduce fragmentation.
                candidate = (new_pressure, -new_remaining, gpu)
                if choice is None or candidate < choice:
                    choice = candidate

            if choice is None:
                return None

            gpu = choice[2]
            placement[gpu].append(record_id)
            loads[gpu] += weight
            remaining[gpu] -= size

        return placement, loads, remaining

    count = len(records)
    all_ids = list(range(count))

    # Different orders are cheap deterministic restarts and produce layouts
    # with substantially different memory fragmentation characteristics.
    orderings = [
        sorted(all_ids, key=lambda i: (records[i][2], records[i][3]), reverse=True),
        sorted(all_ids, key=lambda i: (records[i][3], records[i][2]), reverse=True),
        sorted(
            all_ids,
            key=lambda i: (
                records[i][3] / records[i][2] if records[i][2] > 0 else float("inf"),
                records[i][2],
            ),
            reverse=True,
        ),
        sorted(all_ids, key=lambda i: (records[i][2], -records[i][3]), reverse=True),
    ]

    best_state = None
    best_key = None
    last_remaining = [GPU_MEM_SIZE] * gpu_num

    for order in orderings:
        result = build_greedy(order)
        if result is None:
            continue
        placement, loads, remaining = result
        key = state_key(loads, remaining)
        if best_key is None or key < best_key:
            best_key = key
            best_state = (placement, loads, remaining)
        last_remaining = remaining

    if best_state is None:
        raise ValueError(
            "Unable to place all models on available GPUs. "
            f"Remaining per-GPU memory: {last_remaining}"
        )

    placement, loads, remaining = best_state
    epsilon = 1e-12

    def candidate_key(changes):
        """Score a set of atomic model moves without modifying current state."""
        next_loads = list(loads)
        next_remaining = list(remaining)

        for record_id, source, destination in changes:
            _, _, size, weight = records[record_id]
            next_loads[source] -= weight
            next_remaining[source] += size
            next_loads[destination] += weight
            next_remaining[destination] -= size

        if any(memory < -epsilon for memory in next_remaining):
            return None
        return state_key(next_loads, next_remaining)

    def apply_changes(changes):
        for record_id, source, destination in changes:
            _, _, size, weight = records[record_id]
            placement[source].remove(record_id)
            placement[destination].append(record_id)
            loads[source] -= weight
            remaining[source] += size
            loads[destination] += weight
            remaining[destination] -= size

    # Every accepted action strictly improves the full sorted pressure vector.
    # This avoids getting stuck merely because multiple GPUs tie for maximum KVPR.
    while True:
        current_key = state_key(loads, remaining)
        current_pressures = [kvpr(loads[gpu], remaining[gpu]) for gpu in range(gpu_num)]
        bottleneck_pressure = current_key[0]
        bottlenecks = [
            gpu for gpu in range(gpu_num)
            if abs(current_pressures[gpu] - bottleneck_pressure) <= epsilon
        ]

        best_action = None
        best_action_key = current_key

        def consider(changes):
            nonlocal best_action, best_action_key
            key = candidate_key(changes)
            if key is not None and key < best_action_key:
                best_action_key = key
                best_action = changes

        for source in bottlenecks:
            for model_id in list(placement[source]):
                _, _, model_size, _ = records[model_id]

                # Direct relocation.
                for destination in range(gpu_num):
                    if destination == source or model_size > remaining[destination]:
                        continue
                    consider([(model_id, source, destination)])

                # Pairwise exchange.
                for destination in range(gpu_num):
                    if destination == source:
                        continue
                    for other_id in list(placement[destination]):
                        _, _, other_size, _ = records[other_id]
                        if (remaining[source] + model_size < other_size or
                                remaining[destination] + other_size < model_size):
                            continue
                        consider([
                            (model_id, source, destination),
                            (other_id, destination, source),
                        ])

                # Bounded ejection chain:
                # source model -> destination, destination victim -> third GPU.
                # Third GPUs are considered in pressure order so useful repairs
                # are found early while candidate enumeration remains focused.
                third_candidates = sorted(
                    range(gpu_num),
                    key=lambda gpu: (current_pressures[gpu], -remaining[gpu]),
                )[:min(gpu_num, 4)]

                for destination in range(gpu_num):
                    if destination == source:
                        continue

                    for victim_id in list(placement[destination]):
                        _, _, victim_size, _ = records[victim_id]

                        if remaining[destination] + victim_size < model_size:
                            continue

                        for third in third_candidates:
                            if third == source or third == destination:
                                continue
                            if victim_size > remaining[third]:
                                continue

                            consider([
                                (victim_id, destination, third),
                                (model_id, source, destination),
                            ])

        if best_action is None:
            break

        apply_changes(best_action)

    return {
        gpu: [records[record_id][1] for record_id in placement[gpu]]
        for gpu in range(gpu_num)
    }

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