# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Deterministically construct fourteen points in R^3 maximizing the exact
    squared minimum-distance to diameter ratio.

    Translation and uniform scale are removed after every update.  Multiple
    smooth-continuation schedules explore different contact graphs, followed
    by an exact-ratio-checkpointed sharp polishing phase.
    """
    n, dim = 14, 3
    rng = np.random.default_rng(917263)
    ii, jj = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.sum(x * x) / n)
        return x / max(float(rms), 1.0e-15)

    def exact_ratio_squared(x: np.ndarray) -> float:
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(dsq) / np.max(dsq))

    def smooth_gradient(x: np.ndarray, beta: float) -> np.ndarray:
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)

        lo = float(np.min(dsq))
        low_raw = np.exp(-beta * (dsq - lo))
        low_sum = float(np.sum(low_raw))
        low_weight = low_raw / low_sum
        soft_low = lo - np.log(low_sum) / beta

        hi = float(np.max(dsq))
        high_raw = np.exp(beta * (dsq - hi))
        high_sum = float(np.sum(high_raw))
        high_weight = high_raw / high_sum
        soft_high = hi + np.log(high_sum) / beta

        coeff = low_weight / max(soft_low, 1.0e-12)
        coeff -= high_weight / max(soft_high, 1.0e-12)

        grad = np.zeros_like(x)
        force = 2.0 * coeff[:, None] * delta
        np.add.at(grad, ii, force)
        np.add.at(grad, jj, -force)

        grad -= grad.mean(axis=0, keepdims=True)
        grad -= (np.sum(grad * x) / np.sum(x * x)) * x
        return grad

    best_value = -np.inf
    best_points = None

    starts = 18
    iterations = 1450

    for restart in range(starts):
        x = normalize(rng.normal(size=(n, dim)))
        first = np.zeros_like(x)
        second = np.zeros_like(x)
        schedule = restart % 3

        for step in range(iterations):
            t = step / (iterations - 1)

            if schedule == 0:
                beta = 6.0 + 94.0 * t * t
            elif schedule == 1:
                beta = 6.0 + 194.0 * (t ** 2.3)
            else:
                if t < 0.55:
                    beta = 12.0
                else:
                    u = (t - 0.55) / 0.45
                    beta = 12.0 + 288.0 * u * u

            grad = smooth_gradient(x, beta)
            first = 0.88 * first + 0.12 * grad
            second = 0.96 * second + 0.04 * grad * grad

            lr = 0.045 * (1.0 - t) + 0.0045
            x = normalize(x + lr * first / (np.sqrt(second) + 1.0e-8))

            if step % 25 == 0 or step == iterations - 1:
                value = exact_ratio_squared(x)
                if value > best_value:
                    best_value = value
                    best_points = x.copy()

    x = best_points.copy()
    local_best = best_value
    local_points = x.copy()
    first = np.zeros_like(x)
    second = np.zeros_like(x)
    step_scale = 1.0
    regressions = 0

    for step in range(750):
        t = step / 749.0
        beta = 300.0 + 650.0 * t * t
        grad = smooth_gradient(x, beta)

        first = 0.90 * first + 0.10 * grad
        second = 0.975 * second + 0.025 * grad * grad
        lr = step_scale * (0.0065 * (1.0 - t) + 0.0008)
        x = normalize(x + lr * first / (np.sqrt(second) + 1.0e-8))

        if step % 20 == 19 or step == 749:
            value = exact_ratio_squared(x)

            if value > local_best:
                local_best = value
                local_points = x.copy()

            if value > best_value:
                best_value = value
                best_points = x.copy()

            if value + 1.0e-12 < local_best:
                regressions += 1
                if regressions >= 2:
                    x = local_points.copy()
                    first.fill(0.0)
                    second *= 0.20
                    step_scale *= 0.64
                    regressions = 0
            else:
                regressions = 0
                step_scale = min(1.15, step_scale * 1.02)

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END