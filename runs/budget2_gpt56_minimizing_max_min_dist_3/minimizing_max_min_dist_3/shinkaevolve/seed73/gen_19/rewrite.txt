# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 reproducible points in R^3 with a large minimum-distance to
    diameter ratio.  Translation and uniform scale are normalized away because
    they do not affect the objective.
    """
    n, d = 14, 3
    rng = np.random.default_rng(20240517)
    iu, ju = np.triu_indices(n, 1)

    def normalize(points: np.ndarray) -> np.ndarray:
        points -= points.mean(axis=0)
        points /= np.sqrt(np.mean(points * points))
        return points

    def exact_ratio(points: np.ndarray) -> float:
        delta = points[iu] - points[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(dsq.min() / dsq.max())

    def soft_gradient(points: np.ndarray, beta: float) -> np.ndarray:
        delta = points[iu] - points[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        log_dist = 0.5 * np.log(dsq + 1.0e-15)

        low = -beta * log_dist
        low -= low.max()
        w_min = np.exp(low)
        w_min /= w_min.sum()

        high = beta * log_dist
        high -= high.max()
        w_max = np.exp(high)
        w_max /= w_max.sum()

        pair_gradient = (
            (w_min - w_max)[:, None] * delta / (dsq[:, None] + 1.0e-15)
        )
        gradient = np.zeros_like(points)
        np.add.at(gradient, iu, pair_gradient)
        np.add.at(gradient, ju, -pair_gradient)
        return gradient

    best_points = None
    best_ratio = -np.inf

    for restart in range(12):
        points = normalize(rng.normal(size=(n, d)))
        first = np.zeros_like(points)
        second = np.zeros_like(points)

        # Broad smooth search establishes a good global contact graph.
        for step in range(8000):
            beta = 5.0 + 115.0 * step / 7999.0
            gradient = soft_gradient(points, beta)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            learning_rate = 0.028 * (0.10 + 0.90 * (1.0 - step / 8000.0))
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            normalize(points)

        # Sharper smoothing resolves the final nearest/farthest contacts.
        first.fill(0.0)
        second.fill(0.0)
        for step in range(2200):
            beta = 120.0 + 480.0 * step / 2199.0
            gradient = soft_gradient(points, beta)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            learning_rate = 0.006 * (1.0 - 0.55 * step / 2200.0)
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            normalize(points)

        # The evaluator uses the hard ratio, so perform only strict
        # hard-objective improvements after the differentiable optimization.
        current_ratio = exact_ratio(points)
        proposal_scale = 0.012
        stalled_batches = 0

        for _ in range(500):
            candidate_points = points
            candidate_ratio = current_ratio

            for _ in range(4):
                trial = points + proposal_scale * rng.normal(size=(n, d))
                normalize(trial)
                trial_ratio = exact_ratio(trial)

                if trial_ratio > candidate_ratio:
                    candidate_ratio = trial_ratio
                    candidate_points = trial

            if candidate_ratio > current_ratio:
                points = candidate_points
                current_ratio = candidate_ratio
                proposal_scale = min(0.02, proposal_scale * 1.06)
                stalled_batches = 0
            else:
                stalled_batches += 1
                if stalled_batches % 12 == 0:
                    proposal_scale *= 0.62
                    if proposal_scale < 2.0e-6:
                        break

        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END