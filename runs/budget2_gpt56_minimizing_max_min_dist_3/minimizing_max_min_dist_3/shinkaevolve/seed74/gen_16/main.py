# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen points by deterministic multi-start maximin relaxation.

    The optimization uses a smooth approximation to the minimum and
    maximum squared pairwise distances.  Translation and scale are removed
    after every step, since neither changes the requested ratio.
    """
    n, d = 14, 3
    pair_i, pair_j = np.triu_indices(n, 1)
    rng = np.random.default_rng(20240517)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))

    def exact_ratio(x: np.ndarray) -> float:
        delta = x[pair_i] - x[pair_j]
        squared = np.einsum("ij,ij->i", delta, delta)
        return float(squared.min() / squared.max())

    best_points = None
    best_ratio = -np.inf

    # First locate several unrelated basins.  The later starts are small
    # perturbations of the incumbent, which is substantially more useful
    # than random restarts once a nearly feasible packing has been found.
    for restart in range(24):
        if restart < 16:
            points = normalize(rng.standard_normal((n, d)))
        else:
            points = normalize(
                best_points + 0.10 * rng.standard_normal((n, d))
            )

        for iteration in range(1500):
            delta = points[pair_i] - points[pair_j]
            squared = np.einsum("ij,ij->i", delta, delta)

            # Use a small continuation portfolio: different basins often
            # benefit from different times at moderate sharpness before the
            # objective concentrates on the eventual contact pairs.
            progress = iteration / 1499.0
            schedule = restart % 3
            if schedule == 0:
                # The original balanced geometric cooling path.
                temperature = 0.09 * (0.008 / 0.09) ** progress
            elif schedule == 1:
                # Retain a broad force field longer, then refine more
                # sharply during the last part of the trajectory.
                temperature = 0.09 * (0.004 / 0.09) ** (progress ** 1.65)
            else:
                # A piecewise path first settles a moderate contact graph,
                # followed by a short, highly localized polishing phase.
                if progress < 0.60:
                    temperature = 0.09 * (0.025 / 0.09) ** (progress / 0.60)
                else:
                    temperature = 0.025 * (0.003 / 0.025) ** (
                        (progress - 0.60) / 0.40
                    )

            low_shift = squared.min()
            low_weight = np.exp(-(squared - low_shift) / temperature)
            low_weight /= low_weight.sum()
            smooth_min = low_shift - temperature * np.log(
                np.exp(-(squared - low_shift) / temperature).sum()
            )

            high_shift = squared.max()
            high_weight = np.exp((squared - high_shift) / temperature)
            high_weight /= high_weight.sum()
            smooth_max = high_shift + temperature * np.log(
                np.exp((squared - high_shift) / temperature).sum()
            )

            # Gradient of log(smooth_min / smooth_max) with respect to
            # each squared pair distance.
            pair_weight = low_weight / smooth_min - high_weight / smooth_max
            weights = np.zeros((n, n), dtype=float)
            weights[pair_i, pair_j] = pair_weight
            weights[pair_j, pair_i] = pair_weight

            gradient = 2.0 * (
                weights.sum(axis=1)[:, None] * points - weights @ points
            )
            gradient -= gradient.mean(axis=0, keepdims=True)

            # A normalized step makes the schedule stable across starts.
            gradient_norm = np.sqrt(np.mean(np.sum(gradient * gradient, axis=1)))
            step = 0.045 * (1.0 - 0.70 * iteration / 1499.0)
            points = normalize(points + step * gradient / (gradient_norm + 1e-12))

            if iteration % 50 == 0:
                ratio = exact_ratio(points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = points.copy()

        ratio = exact_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END