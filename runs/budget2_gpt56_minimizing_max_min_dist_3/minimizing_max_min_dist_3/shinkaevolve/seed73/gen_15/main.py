# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct a reproducible 14 point diameter-packing configuration in R^3.

    Translation and uniform scale are removed during optimization because they
    are null directions of the min-distance / diameter objective.
    """
    n, d = 14, 3
    rng = np.random.default_rng(20240517)
    iu, ju = np.triu_indices(n, 1)

    def exact_ratio(p: np.ndarray) -> float:
        delta = p[iu] - p[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(dsq.min() / dsq.max())

    best_points = None
    best_ratio = -np.inf

    # Independent starts are useful because the diameter-packing landscape has
    # several symmetric local optima.  The soft objective is sharpened slowly
    # from a broad repulsion to the true closest/farthest-pair objective.
    for restart in range(12):
        points = rng.normal(size=(n, d))
        points -= points.mean(axis=0)
        points /= np.sqrt(np.mean(points * points))

        first = np.zeros_like(points)
        second = np.zeros_like(points)

        for step in range(8000):
            beta = 5.0 + 115.0 * step / 7999.0

            delta = points[iu] - points[ju]
            dsq = np.einsum("ij,ij->i", delta, delta)
            log_dist = 0.5 * np.log(dsq + 1.0e-15)

            # Derivatives of softmin(log distance)-softmax(log distance).
            a = -beta * log_dist
            a -= a.max()
            w_min = np.exp(a)
            w_min /= w_min.sum()

            b = beta * log_dist
            b -= b.max()
            w_max = np.exp(b)
            w_max /= w_max.sum()

            pair_weight = w_min - w_max
            pair_gradient = pair_weight[:, None] * delta / (dsq[:, None] + 1.0e-15)
            gradient = np.zeros_like(points)
            np.add.at(gradient, iu, pair_gradient)
            np.add.at(gradient, ju, -pair_gradient)

            # Adam ascent is considerably more stable than a raw force step
            # when a closest-pair contact changes.
            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))
            learning_rate = 0.028 * (0.10 + 0.90 * (1.0 - step / 8000.0))
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)

            points -= points.mean(axis=0)
            points /= np.sqrt(np.mean(points * points))

        # The final contact graph is best resolved with a much sharper
        # softmin/softmax than is safe during the global search phase.
        first.fill(0.0)
        second.fill(0.0)
        for polish_step in range(2200):
            beta = 120.0 + 480.0 * polish_step / 2199.0

            delta = points[iu] - points[ju]
            dsq = np.einsum("ij,ij->i", delta, delta)
            log_dist = 0.5 * np.log(dsq + 1.0e-15)

            a = -beta * log_dist
            a -= a.max()
            w_min = np.exp(a)
            w_min /= w_min.sum()

            b = beta * log_dist
            b -= b.max()
            w_max = np.exp(b)
            w_max /= w_max.sum()

            pair_weight = w_min - w_max
            pair_gradient = pair_weight[:, None] * delta / (dsq[:, None] + 1.0e-15)
            gradient = np.zeros_like(points)
            np.add.at(gradient, iu, pair_gradient)
            np.add.at(gradient, ju, -pair_gradient)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (polish_step + 1))
            vhat = second / (1.0 - 0.999 ** (polish_step + 1))
            learning_rate = 0.006 * (1.0 - 0.55 * polish_step / 2200.0)
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)

            points -= points.mean(axis=0)
            points /= np.sqrt(np.mean(points * points))

        # The smoothed objective can stop slightly away from the optimum of
        # the hard min/max ratio.  Finish with an exact-objective hill climb:
        # all proposed moves are projected back to the centered, fixed-scale
        # representation, and only strict improvements are retained.
        current_ratio = exact_ratio(points)
        proposal_scale = 0.012
        stalled_batches = 0
        for exact_step in range(500):
            improved = False
            candidate_points = points
            candidate_ratio = current_ratio

            # A small batch makes it practical to explore several directions
            # at each scale without accepting any surrogate-objective loss.
            for _ in range(4):
                trial = points + proposal_scale * rng.normal(size=(n, d))
                trial -= trial.mean(axis=0)
                trial /= np.sqrt(np.mean(trial * trial))
                trial_ratio = exact_ratio(trial)
                if trial_ratio > candidate_ratio:
                    candidate_ratio = trial_ratio
                    candidate_points = trial
                    improved = True

            if improved:
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

        ratio = current_ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END