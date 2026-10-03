# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Deterministically optimize ten points inside a fixed triangular hull by multi-start annealing."""
    rng = np.random.default_rng(137)
    n = 13

    # Keeping these corners fixes the convex hull to this triangle.  Consequently
    # its area is exactly 1/2 throughout the optimization and all returned points
    # are automatically in a single convex region.
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def areas(p: np.ndarray) -> np.ndarray:
        q = p[triples]
        return 0.5 * np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def merit(p: np.ndarray) -> tuple[float, float]:
        """Use the minimum plus a lower-tail bonus during the nonsmooth search."""
        a = areas(p)
        low = np.partition(a, 11)[:12]
        return float(low[0]), float(low[0] + 0.12 * low.mean())

    best = None
    best_min = -1.0

    # Several independent starts are useful because the max-min landscape has
    # many local optima.  The seed makes all starts and accepted moves repeatable.
    for restart in range(10):
        u = rng.random((10, 2))
        # Uniform sampling in the reference triangle.
        mask = u.sum(axis=1) > 1.0
        u[mask] = 1.0 - u[mask]
        p = np.vstack((corners, u))
        current_min, current_merit = merit(p)

        for it in range(30000):
            # Large early moves find a basin; small late moves polish active triples.
            fraction = it / 29999.0
            step = 0.095 * (1.0 - fraction) ** 1.7 + 0.0012
            temperature = 0.0018 * (1.0 - fraction) ** 2 + 0.000002

            idx = int(rng.integers(3, n))
            old = p[idx].copy()
            candidate = old + rng.normal(0.0, step, size=2)

            # Reject rather than clip outside moves, avoiding artificial piles on edges.
            if candidate[0] < 0.0 or candidate[1] < 0.0 or candidate.sum() > 1.0:
                continue

            p[idx] = candidate
            trial_min, trial_merit = merit(p)
            delta = trial_merit - current_merit
            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                current_min, current_merit = trial_min, trial_merit
            else:
                p[idx] = old

            if trial_min > best_min:
                best_min = trial_min
                best = p.copy()

        if current_min > best_min:
            best_min = current_min
            best = p.copy()

    # The reference hull has area 1/2; affine normalization by an evaluator
    # therefore leaves the normalized minimum triangle area unchanged.
    return best


# EVOLVE-BLOCK-END
