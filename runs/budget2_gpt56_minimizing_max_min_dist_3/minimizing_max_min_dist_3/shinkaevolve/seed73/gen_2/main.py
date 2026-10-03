# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen points by directly searching for a large
    minimum-distance / diameter ratio.
    """
    n, d = 14, 3
    rng = np.random.default_rng(41829)
    ii, jj = np.triu_indices(n, 1)

    def normalize(points: np.ndarray) -> np.ndarray:
        points = points - points.mean(axis=0, keepdims=True)
        delta = points[:, None, :] - points[None, :, :]
        diameter = np.sqrt(np.sum(delta * delta, axis=2).max())
        return points / diameter

    def distances_squared(points: np.ndarray) -> np.ndarray:
        delta = points[ii] - points[jj]
        return np.einsum("ij,ij->i", delta, delta)

    best_points = None
    best_score = -np.inf

    # A few independent contact graphs are considerably more reliable than
    # one long run for this nonsmooth maximin problem.
    for restart in range(8):
        points = normalize(rng.normal(size=(n, d)))
        score = distances_squared(points).min()

        for iteration in range(2600):
            q = distances_squared(points)
            qmin = q.min()
            qmax = q.max()

            # Smooth active-set weights: close pairs repel and diameter pairs
            # attract.  The temperature narrows as the contact graph settles.
            width = 0.035 * (1.0 - iteration / 2600.0) + 0.0015
            near_weight = np.exp(-(q - qmin) / width)
            far_weight = np.exp(-(qmax - q) / width)

            displacement = points[ii] - points[jj]
            gradient = np.zeros_like(points)
            np.add.at(gradient, ii, near_weight[:, None] * displacement)
            np.add.at(gradient, jj, -near_weight[:, None] * displacement)

            # Keeping the largest separations short is essential because the
            # configuration is subsequently normalized to unit diameter.
            balance = qmin / max(qmax, 1.0e-12)
            np.add.at(
                gradient,
                ii,
                -balance * far_weight[:, None] * displacement,
            )
            np.add.at(
                gradient,
                jj,
                balance * far_weight[:, None] * displacement,
            )

            gnorm = np.sqrt(np.mean(gradient * gradient))
            if gnorm > 1.0e-14:
                gradient /= gnorm

            fraction = iteration / 2600.0
            step = 0.030 * (1.0 - fraction) + 0.0010
            noise = rng.normal(size=(n, d)) * (0.006 * (1.0 - fraction) ** 2)
            candidate = normalize(points + step * gradient + noise)
            candidate_score = distances_squared(candidate).min()

            # A small annealed acceptance probability avoids being trapped by
            # an early, inferior set of nearest-neighbour contacts.
            temperature = 0.0015 * (1.0 - fraction) ** 2 + 1.0e-6
            if candidate_score >= score or rng.random() < np.exp(
                (candidate_score - score) / temperature
            ):
                points, score = candidate, candidate_score

        # Finish each restart with several noise-free active-set updates.
        for _ in range(160):
            q = distances_squared(points)
            qmin, qmax = q.min(), q.max()
            near_weight = np.exp(-(q - qmin) / 0.001)
            far_weight = np.exp(-(qmax - q) / 0.001)
            displacement = points[ii] - points[jj]
            gradient = np.zeros_like(points)
            np.add.at(gradient, ii, near_weight[:, None] * displacement)
            np.add.at(gradient, jj, -near_weight[:, None] * displacement)
            balance = qmin / qmax
            np.add.at(gradient, ii, -balance * far_weight[:, None] * displacement)
            np.add.at(gradient, jj, balance * far_weight[:, None] * displacement)
            norm = np.sqrt(np.mean(gradient * gradient))
            if norm > 1.0e-14:
                candidate = normalize(points + 0.0015 * gradient / norm)
                candidate_score = distances_squared(candidate).min()
                if candidate_score > score:
                    points, score = candidate, candidate_score

        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END