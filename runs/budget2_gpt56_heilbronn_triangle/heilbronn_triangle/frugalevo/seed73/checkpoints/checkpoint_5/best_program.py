# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Optimize 11 points by deterministic multi-start simulated annealing.

    Points are searched in affine coordinates (u, v) in the simplex
    u >= 0, v >= 0, u + v <= 1, then mapped to the requested equilateral
    triangle by (x, y) = (u + v/2, sqrt(3)*v/2).  The objective is the
    smallest absolute determinant over all 165 triples.
    """
    n = 11
    rng = np.random.default_rng(918273)

    # Precompute the 165 unordered triples once.
    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.int64,
    )

    def project_simplex(z: np.ndarray) -> np.ndarray:
        """Project rows onto the triangle u >= 0, v >= 0, u + v <= 1."""
        z = np.maximum(z, 0.0)
        excess = z[:, 0] + z[:, 1] - 1.0
        mask = excess > 0.0
        if np.any(mask):
            # Projection onto the hypotenuse of the 2D simplex.
            z[mask] -= (excess[mask, None] * 0.5)
            z[mask] = np.maximum(z[mask], 0.0)
            # Handle the rare case where the previous projection reaches
            # an axis and numerical clipping is needed.
            sums = z[mask].sum(axis=1)
            over = sums > 1.0
            if np.any(over):
                zmask = z[mask]
                zmask[over] /= sums[over, None]
                z[mask] = zmask
        return z

    def objective(p: np.ndarray) -> float:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
        det -= (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        return float(np.min(np.abs(det)))

    best_points = None
    best_value = -1.0

    # Use many independent uniform-simplex starts.  The minimum-triangle
    # objective is nonsmooth, so additional basins are more valuable than
    # making any one annealing trajectory excessively long.  Two-point moves
    # help repair multiple simultaneously active near-collinear triples.
    for restart in range(55):
        bary = rng.dirichlet((1.0, 1.0, 1.0), size=n)
        q = bary[:, :2].copy()
        current_value = objective(q)

        for iteration in range(7000):
            fraction = iteration / 7000.0
            late = max(0.0, 1.0 - fraction)
            scale = 0.085 * late ** 1.45 + 0.00020
            temperature = 0.010 * late ** 2.4 + 2.0e-8

            candidate = q.copy()
            if rng.random() < 0.30:
                indices = rng.choice(n, size=2, replace=False)
                candidate[indices] += rng.normal(0.0, scale, (2, 2))
            else:
                index = int(rng.integers(n))
                candidate[index] += rng.normal(0.0, scale, 2)

            candidate = project_simplex(candidate)
            candidate_value = objective(candidate)
            delta = candidate_value - current_value

            if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                q = candidate
                current_value = candidate_value

            if current_value > best_value:
                best_value = current_value
                best_points = q.copy()

        # Deterministic local polishing using both single-point and coordinated
        # two-point moves.  Coupled moves are important when several active
        # minimum-area triples share vertices.
        for iteration in range(1400):
            fraction = iteration / 1400.0
            scale = 0.0020 * (1.0 - fraction) ** 1.3 + 1.0e-5
            candidate = q.copy()

            if rng.random() < 0.45:
                indices = rng.choice(n, size=2, replace=False)
                candidate[indices] += rng.normal(0.0, scale, (2, 2))
            else:
                index = int(rng.integers(n))
                candidate[index] += rng.normal(0.0, scale, 2)

            candidate = project_simplex(candidate)
            candidate_value = objective(candidate)
            if candidate_value >= current_value:
                q = candidate
                current_value = candidate_value
                if current_value > best_value:
                    best_value = current_value
                    best_points = q.copy()

    # Convert affine simplex coordinates to Cartesian equilateral-triangle
    # coordinates.  The slight clipping guards against floating-point drift.
    u = best_points[:, 0]
    v = best_points[:, 1]
    points = np.column_stack((u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v))
    return np.clip(points, 0.0, 1.0)


# EVOLVE-BLOCK-END
