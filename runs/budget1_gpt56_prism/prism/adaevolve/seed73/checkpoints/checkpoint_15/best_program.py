GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Greedily minimize the sorted GPU-KVPR profile, then improve it by moves and swaps."""

    placement = {i: [] for i in range(gpu_num)}
    remaining = [GPU_MEM_SIZE] * gpu_num
    load = [0.0] * gpu_num

    def maximum():
        # The first item is the objective; later items break ties by balancing
        # the other GPUs and can expose better subsequent local-search moves.
        return tuple(sorted(
            (load[i] / remaining[i] if remaining[i] else float("inf")
             for i in range(gpu_num)),
            reverse=True
        ))

    # A model's standalone pressure accounts for both its request load and the
    # KV-cache capacity it removes, unlike request rate alone.
    for model in sorted(
        models,
        key=lambda m: (-(m.req_rate / m.slo) /
                       max(1e-9, GPU_MEM_SIZE - m.model_size), -m.model_size)
    ):
        size = model.model_size
        weight = model.req_rate / model.slo
        best_gpu, best_score = None, None

        for gpu in range(gpu_num):
            if size > remaining[gpu]:
                continue
            remaining[gpu] -= size
            load[gpu] += weight
            score = maximum()
            remaining[gpu] += size
            load[gpu] -= weight
            if best_score is None or score < best_score:
                best_gpu, best_score = gpu, score

        if best_gpu is None:
            raise ValueError("Models do not fit in the available GPU memory")

        placement[best_gpu].append(model)
        remaining[best_gpu] -= size
        load[best_gpu] += weight

    for _ in range(min(5, len(models))):
        best, best_score = None, maximum()

        for src in range(gpu_num):
            for model in placement[src]:
                size = model.model_size
                weight = model.req_rate / model.slo
                for dst in range(gpu_num):
                    if src == dst or size > remaining[dst]:
                        continue
                    remaining[src] += size
                    load[src] -= weight
                    remaining[dst] -= size
                    load[dst] += weight
                    score = maximum()
                    remaining[src] -= size
                    load[src] += weight
                    remaining[dst] += size
                    load[dst] -= weight
                    if score < best_score:
                        best, best_score = ("move", src, dst, model), score

        for a in range(gpu_num):
            for b in range(a + 1, gpu_num):
                for left in placement[a]:
                    for right in placement[b]:
                        ls, lw = left.model_size, left.req_rate / left.slo
                        rs, rw = right.model_size, right.req_rate / right.slo
                        if rs > remaining[a] + ls or ls > remaining[b] + rs:
                            continue
                        remaining[a] += ls - rs
                        load[a] += rw - lw
                        remaining[b] += rs - ls
                        load[b] += lw - rw
                        score = maximum()
                        remaining[a] -= ls - rs
                        load[a] -= rw - lw
                        remaining[b] -= rs - ls
                        load[b] -= lw - rw
                        if score < best_score:
                            best, best_score = ("swap", a, b, left, right), score

        if best is None:
            break

        if best[0] == "move":
            _, src, dst, model = best
            placement[src].remove(model)
            placement[dst].append(model)
            remaining[src] += model.model_size
            load[src] -= model.req_rate / model.slo
            remaining[dst] -= model.model_size
            load[dst] += model.req_rate / model.slo
        else:
            _, a, b, left, right = best
            placement[a].remove(left)
            placement[b].remove(right)
            placement[a].append(right)
            placement[b].append(left)
            remaining[a] += left.model_size - right.model_size
            load[a] += right.req_rate / right.slo - left.req_rate / left.slo
            remaining[b] += right.model_size - left.model_size
            load[b] += left.req_rate / left.slo - right.req_rate / right.slo

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
