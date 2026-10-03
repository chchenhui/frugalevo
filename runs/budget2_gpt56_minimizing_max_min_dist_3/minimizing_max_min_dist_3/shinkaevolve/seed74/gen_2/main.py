# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 three-dimensional points as seven antipodal pairs.

    The antipodal construction fixes the diameter at 2 while reducing the
    problem to minimizing the largest absolute inner product among seven
    unit vectors (a spherical line-packing problem).
    """
    rng = np.random.default_rng(20250308)
    n_lines = 7

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def coherence(u: np.ndarray) -> float:
        gram = np.abs(u @ u.T)
        np.fill_diagonal(gram, 0.0)
        return float(np.max(gram))

    # Six icosahedral lines have especially low mutual coherence in R^3.
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    base6 = np.array(
        [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
        ],
        dtype=float,
    )
    base6 = normalize_rows(base6)

    # Deterministically select a good seventh line before jointly refining
    # all seven lines.  Sampling is vectorized and therefore inexpensive.
    trial = rng.normal(size=(24000, 3))
    trial = normalize_rows(trial)
    initial_cost = np.max(np.abs(trial @ base6.T), axis=1)
    seventh = trial[np.argmin(initial_cost)]
    seed = np.vstack((base6, seventh))

    best = seed.copy()
    best_value = coherence(best)

    # Smooth max continuation: progressively sharpen log-sum-exp of squared
    # correlations.  Gradients are projected to each sphere's tangent plane.
    betas = (10.0, 25.0, 60.0, 140.0, 320.0, 700.0)
    steps = (150, 180, 220, 260, 280, 320)
    rates = (0.055, 0.042, 0.030, 0.020, 0.012, 0.007)

    # A few small, reproducible perturbations explore distinct basins while
    # retaining the strong icosahedral starting geometry.
    for restart in range(8):
        if restart == 0:
            u = seed.copy()
        else:
            noise_scale = 0.045 + 0.025 * restart
            u = normalize_rows(seed + noise_scale * rng.normal(size=(n_lines, 3)))

        for beta, count, rate in zip(betas, steps, rates):
            velocity = np.zeros_like(u)

            for iteration in range(count):
                dots = u @ u.T
                sq = dots * dots
                np.fill_diagonal(sq, -np.inf)

                # Stable softmax weights over unordered line pairs.
                peak = np.max(sq)
                weights = np.exp(beta * (sq - peak))
                np.fill_diagonal(weights, 0.0)
                weights /= np.sum(weights)

                # Gradient of the soft maximum of squared correlations.
                grad = 2.0 * ((weights * dots) @ u)

                # Riemannian projection onto tangent spaces of S^2.
                grad -= np.sum(grad * u, axis=1, keepdims=True) * u

                # Light momentum improves movement through nearly symmetric
                # configurations without changing the deterministic result.
                velocity = 0.78 * velocity + grad
                velocity -= np.sum(velocity * u, axis=1, keepdims=True) * u
                u = normalize_rows(u - rate * velocity)

                if iteration % 20 == 0 or iteration == count - 1:
                    value = coherence(u)
                    if value < best_value:
                        best_value = value
                        best = u.copy()

    # Antipodal pairs make the maximum distance exactly 2 and preserve all
    # line-packing separations as Euclidean point separations.
    points = np.vstack((best, -best))
    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END
