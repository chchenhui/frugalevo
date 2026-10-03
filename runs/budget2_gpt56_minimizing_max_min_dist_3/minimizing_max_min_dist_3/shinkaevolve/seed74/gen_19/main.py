# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 three-dimensional points with a large minimum pairwise
    distance relative to their maximum pairwise distance.

    The construction is deterministic and uses multi-start projected
    optimization of a smooth, annealed min/max-distance objective.
    """
    n = 14
    dim = 3
    rng = np.random.default_rng(917263)

    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.sum(x * x) / n)
        return x / rms

    def exact_ratio_squared(x: np.ndarray) -> float:
        delta = x[ii] - x[jj]
        dsq = np.sum(delta * delta, axis=1)
        return float(dsq.min() / dsq.max())

    def smooth_gradient(x: np.ndarray, beta: float) -> np.ndarray:
        delta = x[ii] - x[jj]
        dsq = np.sum(delta * delta, axis=1)

        # Stable soft minimum of squared distances.
        mn = float(dsq.min())
        low = np.exp(-beta * (dsq - mn))
        low /= low.sum()
        soft_min = mn - np.log(np.exp(-beta * (dsq - mn)).sum()) / beta

        # Stable soft maximum of squared distances.
        mx = float(dsq.max())
        high = np.exp(beta * (dsq - mx))
        high /= high.sum()
        soft_max = mx + np.log(np.exp(beta * (dsq - mx)).sum()) / beta

        # Derivative of log(soft_min / soft_max).
        coeff = low / max(soft_min, 1.0e-12) - high / max(soft_max, 1.0e-12)
        grad = np.zeros_like(x)
        weighted = 2.0 * coeff[:, None] * delta
        np.add.at(grad, ii, weighted)
        np.add.at(grad, jj, -weighted)

        # Translation and scale directions are removed because configurations
        # are re-centered and re-normalized after every update.
        grad -= grad.mean(axis=0, keepdims=True)
        radial = np.sum(grad * x) / np.sum(x * x)
        grad -= radial * x
        return grad

    best_points = None
    best_value = -np.inf

    # Independent starts are useful because the extremal-distance landscape
    # has several distinct local optima.
    starts = 18
    iterations = 1450

    for restart in range(starts):
        x = normalize(rng.normal(size=(n, dim)))

        # Adam-style projected ascent of the annealed smooth objective.
        first_moment = np.zeros_like(x)
        second_moment = np.zeros_like(x)

        for step in range(iterations):
            u = step / (iterations - 1)

            # Start with a broad repulsion field and gradually focus on the
            # actual closest and farthest pairs.
            beta = 10.0 + 210.0 * (u ** 1.65)
            grad = smooth_gradient(x, beta)

            first_moment = 0.88 * first_moment + 0.12 * grad
            second_moment = 0.96 * second_moment + 0.04 * (grad * grad)

            # Large initial moves explore the basin; later moves polish it.
            learning_rate = 0.045 * (1.0 - u) + 0.0045
            x = x + learning_rate * first_moment / (np.sqrt(second_moment) + 1.0e-8)
            x = normalize(x)

            # Preserve the exact best state, rather than only the smoothed one.
            if step % 25 == 0 or step == iterations - 1:
                value = exact_ratio_squared(x)
                if value > best_value:
                    best_value = value
                    best_points = x.copy()

    # A final sharp continuation is checkpointed against the actual
    # nonsmooth objective.  This is useful when the identity of an active
    # closest or diameter pair changes during polishing.
    x = best_points.copy()
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    step_scale = 1.0
    checkpoint_value = best_value
    regressions = 0

    for step in range(750):
        t = step / 749.0
        beta = 300.0 + 650.0 * (t * t)
        grad = smooth_gradient(x, beta)
        first_moment = 0.90 * first_moment + 0.10 * grad
        second_moment = 0.975 * second_moment + 0.025 * (grad * grad)
        lr = step_scale * (0.0065 * (1.0 - t) + 0.0008)
        x = normalize(x + lr * first_moment / (np.sqrt(second_moment) + 1.0e-8))

        if step % 25 == 24 or step == 749:
            value = exact_ratio_squared(x)
            if value > best_value:
                best_value = value
                best_points = x.copy()

            # Repeated declines indicate momentum is following a smoothed
            # contact set that no longer matches the exact extrema.
            if value + 1.0e-12 < checkpoint_value:
                regressions += 1
                if regressions >= 2:
                    step_scale *= 0.65
                    first_moment.fill(0.0)
                    second_moment *= 0.25
                    regressions = 0
            else:
                regressions = 0
                step_scale = min(1.15, step_scale * 1.025)
            checkpoint_value = value

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END