# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    # Use seven lines through the origin, returning both unit vectors on
    # every line.  Antipodality makes the diameter exactly 2, so maximizing
    # the distance ratio is equivalent to minimizing the largest absolute
    # dot product among the seven directions.
    rng = np.random.default_rng(271828)
    n_lines = 7
    n_starts = 96

    directions = rng.normal(size=(n_starts, n_lines, 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)

    def descend(v, beta, steps, rate):
        """Riemannian descent for log-sum-exp(max |vi dot vj|^2)."""
        momentum = np.zeros_like(v)
        for _ in range(steps):
            dots = np.einsum("aid,ajd->aij", v, v)
            values = beta * dots * dots
            values -= np.max(values, axis=(1, 2), keepdims=True)

            # Diagonal terms are constant on the sphere and must not
            # contribute to the soft maximum or its gradient.
            weights = np.exp(values)
            diagonal = np.arange(n_lines)
            weights[:, diagonal, diagonal] = 0.0
            weights /= np.sum(weights, axis=(1, 2), keepdims=True)

            grad = 2.0 * np.einsum("aij,aij,ajd->aid", weights, dots, v)
            # Tangential projection keeps each direction on its sphere.
            grad -= np.sum(grad * v, axis=2, keepdims=True) * v
            momentum = 0.82 * momentum + grad
            v -= rate * momentum
            v /= np.linalg.norm(v, axis=2, keepdims=True)
        return v

    # Broad search first: low beta establishes balanced configurations,
    # while later stages increasingly approximate the true maximum.
    for beta, steps, rate in ((8.0, 400, 0.075),
                              (25.0, 400, 0.055),
                              (75.0, 450, 0.035),
                              (180.0, 450, 0.020)):
        directions = descend(directions, beta, steps, rate)

    dots = np.abs(np.einsum("aid,ajd->aij", directions, directions))
    dots[:, np.arange(n_lines), np.arange(n_lines)] = 0.0
    best_indices = np.argsort(np.max(dots, axis=(1, 2)))[:10]
    directions = directions[best_indices]

    # Spend the expensive sharp minimax refinement only on the candidates
    # that already have the best exact contact distance.
    for beta, steps, rate in ((350.0, 1000, 0.012),
                              (700.0, 1200, 0.007),
                              (1400.0, 1400, 0.0035)):
        directions = descend(directions, beta, steps, rate)

    dots = np.abs(np.einsum("aid,ajd->aij", directions, directions))
    dots[:, np.arange(n_lines), np.arange(n_lines)] = 0.0
    best = directions[np.argmin(np.max(dots, axis=(1, 2)))]

    points = np.vstack((best, -best))
    return points


# EVOLVE-BLOCK-END