# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 reproducible points in R^3 with a large minimum-distance to
    diameter ratio.  Translation and uniform scale are normalized away during
    optimization because they do not affect the objective.
    """
    n, d = 14, 3
    rng = np.random.default_rng(20240517)
    iu, ju = np.triu_indices(n, 1)
    eps = 1.0e-15

    def normalize(p: np.ndarray) -> np.ndarray:
        p -= p.mean(axis=0)
        p /= np.sqrt(np.mean(p * p))
        return p

    def exact_ratio(p: np.ndarray) -> float:
        delta = p[iu] - p[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(dsq.min() / dsq.max())

    def optimize(points: np.ndarray, steps: int,
                 beta_lo: float, beta_hi: float,
                 lr_hi: float, lr_lo: float) -> np.ndarray:
        first = np.zeros_like(points)
        second = np.zeros_like(points)

        for step in range(steps):
            t = step / max(steps - 1, 1)
            beta = beta_lo + (beta_hi - beta_lo) * t

            delta = points[iu] - points[ju]
            dsq = np.einsum("ij,ij->i", delta, delta)
            log_dist = 0.5 * np.log(dsq + eps)

            soft_min = -beta * log_dist
            soft_min -= soft_min.max()
            w_min = np.exp(soft_min)
            w_min /= w_min.sum()

            soft_max = beta * log_dist
            soft_max -= soft_max.max()
            w_max = np.exp(soft_max)
            w_max /= w_max.sum()

            pair_gradient = (
                (w_min - w_max)[:, None] * delta / (dsq[:, None] + eps)
            )
            gradient = np.zeros_like(points)
            np.add.at(gradient, iu, pair_gradient)
            np.add.at(gradient, ju, -pair_gradient)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            learning_rate = lr_hi + (lr_lo - lr_hi) * t
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            normalize(points)

        return points

    best_points = None
    best_ratio = -np.inf

    # Include the deterministic initialization used by the lightweight prior
    # implementation, then use independent modern-generator starts.
    legacy_rng = np.random.RandomState(42)
    initial_points = [legacy_rng.randn(n, d)]
    initial_points.extend(rng.normal(size=(n, d)) for _ in range(12))

    for initial in initial_points:
        points = normalize(np.asarray(initial, dtype=float).copy())

        # Broad continuation phase: allows the contact graph to reorganize.
        points = optimize(
            points, steps=7200,
            beta_lo=5.0, beta_hi=120.0,
            lr_hi=0.028, lr_lo=0.0028,
        )

        # Sharp final phase: resolves closest and diameter contacts accurately.
        points = optimize(
            points, steps=2400,
            beta_lo=120.0, beta_hi=650.0,
            lr_hi=0.006, lr_lo=0.0027,
        )

        ratio = exact_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END