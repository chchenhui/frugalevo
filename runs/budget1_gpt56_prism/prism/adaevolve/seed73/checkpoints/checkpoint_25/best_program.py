GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """Use exact MILP set partitioning over feasible GPU subsets, with greedy fallback."""

    n = len(models)
    size = [m.model_size for m in models]
    weight = [m.req_rate / m.slo for m in models]

    def greedy():
        result = {g: [] for g in range(gpu_num)}
        free, load = [GPU_MEM_SIZE] * gpu_num, [0.0] * gpu_num
        for i in sorted(range(n), key=lambda i: (-size[i], -weight[i])):
            choices = [g for g in range(gpu_num) if free[g] >= size[i]]
            if not choices:
                raise ValueError("Models do not fit in the available GPU memory")
            g = min(choices, key=lambda g: max(
                (load[j] + (weight[i] if j == g else 0.0)) /
                (free[j] - (size[i] if j == g else 0.0))
                for j in range(gpu_num)))
            result[g].append(models[i])
            free[g] -= size[i]
            load[g] += weight[i]
        return result

    # A configuration is (used memory, request load, model-index bitmask).
    # Cap enumeration because the exact formulation has one binary per subset.
    configs = [(0.0, 0.0, 0)]
    limit = 12000
    for i in range(n):
        extra = []
        for used, load, mask in configs:
            if used + size[i] < GPU_MEM_SIZE - 1e-10:
                extra.append((used + size[i], load + weight[i], mask | (1 << i)))
        configs += extra
        if len(configs) > limit:
            return greedy()

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix, vstack
    except ImportError:
        return greedy()

    q = len(configs)
    pressure = np.array([load / (GPU_MEM_SIZE - used)
                         for used, load, mask in configs])

    # Each model is in exactly one selected configuration and exactly gpu_num
    # configurations are selected.  z >= pressure[c] for every selected c.
    rows, cols = [], []
    for c, (_, _, mask) in enumerate(configs):
        for i in range(n):
            if mask & (1 << i):
                rows.append(i)
                cols.append(c)
    cover = coo_matrix((np.ones(len(rows)), (rows, cols)),
                       shape=(n, q)).tocsr()
    count = coo_matrix((np.ones(q), (np.zeros(q), np.arange(q))),
                       shape=(1, q)).tocsr()
    base = vstack((cover, count))
    ratio = coo_matrix(
        (np.r_[-pressure, np.ones(q)],
         (np.r_[np.arange(q), np.arange(q)],
          np.r_[np.arange(q), np.full(q, q)])),
        shape=(q, q + 1)).tocsr()
    constraints = [
        LinearConstraint(
            coo_matrix((np.ones(base.nnz), (base.nonzero()[0], base.nonzero()[1])),
                       shape=(n + 1, q + 1)).tocsr(),
            np.r_[np.ones(n), gpu_num], np.r_[np.ones(n), gpu_num]),
        LinearConstraint(ratio, np.zeros(q), np.full(q, np.inf))
    ]

    result = milp(
        c=np.r_[np.zeros(q), 1.0],
        integrality=np.r_[np.ones(q), 0],
        bounds=Bounds(np.zeros(q + 1), np.full(q + 1, np.inf)),
        constraints=constraints,
        options={"time_limit": 30}
    )
    if result.x is None:
        return greedy()

    placement = {g: [] for g in range(gpu_num)}
    selected = [c for c in range(q) if result.x[c] > 0.5]
    if len(selected) != gpu_num:
        return greedy()
    for g, c in enumerate(selected):
        mask = configs[c][2]
        placement[g] = [models[i] for i in range(n) if mask & (1 << i)]
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
