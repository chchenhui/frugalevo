# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points by directly maximizing the smallest normalized
    triangle area in barycentric coordinates.
    """
    n = 11
    rng = np.random.default_rng(11031991)
    triples = np.array(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=int,
    )

    def score(p: np.ndarray) -> float:
        a = p[triples[:, 1]] - p[triples[:, 0]]
        b = p[triples[:, 2]] - p[triples[:, 0]]
        return float(np.min(np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))

    def feasible(q: np.ndarray) -> np.ndarray:
        q = np.maximum(q, 0.0)
        total = float(q[0] + q[1])
        if total > 1.0:
            q /= total
        return q

    # The first three entries are the simplex vertices.  The remaining
    # points begin as a slightly perturbed, well-spread triangular pattern.
    template = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.25, 0.00], [0.50, 0.00], [0.75, 0.00],
        [0.00, 0.35], [0.00, 0.68],
        [0.32, 0.25], [0.58, 0.20], [0.28, 0.52],
    ])
    best = template.copy()
    best_value = score(best)

    # Multi-start annealing is deterministic and keeps all candidate points
    # in the reference simplex.  A short final greedy stage removes thermal
    # noise from the returned configuration.
    for restart in range(14):
        if restart == 0:
            current = template.copy()
            current[3:] = np.array([feasible(q + rng.normal(0.0, 0.025, 2))
                                    for q in current[3:]])
        else:
            current = np.empty((n, 2))
            current[:3] = template[:3]
            raw = rng.exponential(1.0, size=(8, 3))
            current[3:] = raw[:, :2] / raw.sum(axis=1, keepdims=True)

        current_value = score(current)
        iterations = 18000
        for step in range(iterations):
            fraction = step / (iterations - 1)
            sigma = 0.085 * (1.0 - fraction) ** 1.7 + 0.0015
            temperature = 0.0045 * (1.0 - fraction) ** 2.2 + 1.0e-6
            index = int(rng.integers(3, n))
            proposal = current.copy()
            proposal[index] = feasible(
                proposal[index] + rng.normal(0.0, sigma, size=2)
            )
            candidate_value = score(proposal)
            if (candidate_value >= current_value or
                    rng.random() < np.exp((candidate_value - current_value) /
                                          temperature)):
                current = proposal
                current_value = candidate_value
                if current_value > best_value:
                    best = current.copy()
                    best_value = current_value

        for _ in range(3500):
            index = int(rng.integers(3, n))
            proposal = best.copy()
            proposal[index] = feasible(
                proposal[index] + rng.normal(0.0, 0.0018, size=2)
            )
            candidate_value = score(proposal)
            if candidate_value > best_value:
                best = proposal
                best_value = candidate_value

    # (u, v) barycentric simplex coordinates map to the requested triangle.
    points = np.empty_like(best)
    points[:, 0] = best[:, 0] + 0.5 * best[:, 1]
    points[:, 1] = (np.sqrt(3.0) / 2.0) * best[:, 1]
    return points


# EVOLVE-BLOCK-END