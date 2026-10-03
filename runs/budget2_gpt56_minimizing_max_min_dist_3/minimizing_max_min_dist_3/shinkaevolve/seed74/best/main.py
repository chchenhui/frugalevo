# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct exactly fourteen points in R^3 with a large squared ratio of
    minimum pairwise distance to maximum pairwise distance.

    Translation and uniform scale are removed after every update, leaving a
    compact optimization problem.  The returned configuration is selected
    exclusively by the exact nonsmooth distance ratio used by the evaluator.
    """
    n, dim = 14, 3
    rng = np.random.default_rng(917263)
    ii, jj = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        norm = np.sqrt(np.sum(x * x) / n)
        return x / max(norm, 1.0e-15)

    def exact_ratio_squared(x: np.ndarray) -> float:
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(dsq) / np.max(dsq))

    def smooth_gradient(x: np.ndarray, beta: float) -> np.ndarray:
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)

        lo = float(np.min(dsq))
        a = np.exp(-beta * (dsq - lo))
        suma = float(np.sum(a))
        low_weights = a / suma
        soft_low = lo - np.log(suma) / beta

        hi = float(np.max(dsq))
        b = np.exp(beta * (dsq - hi))
        sumb = float(np.sum(b))
        high_weights = b / sumb
        soft_high = hi + np.log(sumb) / beta

        coeff = low_weights / max(soft_low, 1.0e-12)
        coeff -= high_weights / max(soft_high, 1.0e-12)

        grad = np.zeros_like(x)
        force = 2.0 * coeff[:, None] * delta
        np.add.at(grad, ii, force)
        np.add.at(grad, jj, -force)

        grad -= grad.mean(axis=0, keepdims=True)
        grad -= (np.sum(grad * x) / np.sum(x * x)) * x
        return grad

    best_value = -np.inf
    best_points = None

    # Broad multistart continuation finds several distinct candidate contact
    # graphs.  A common schedule makes the quality of starts directly
    # comparable rather than depending on an arbitrary schedule assignment.
    starts = 20
    iterations = 1500

    for restart in range(starts):
        x = normalize(rng.normal(size=(n, dim)))
        first = np.zeros_like(x)
        second = np.zeros_like(x)

        for step in range(iterations):
            t = step / (iterations - 1)
            beta = 9.0 + 245.0 * (t ** 1.7)
            grad = smooth_gradient(x, beta)

            first = 0.88 * first + 0.12 * grad
            second = 0.96 * second + 0.04 * grad * grad
            lr = 0.044 * (1.0 - t) + 0.004
            x = normalize(x + lr * first / (np.sqrt(second) + 1.0e-8))

            if step % 25 == 0 or step == iterations - 1:
                value = exact_ratio_squared(x)
                if value > best_value:
                    best_value = value
                    best_points = x.copy()

    # High-sharpness polish with exact-ratio checkpointing.  At this stage
    # contact-pair changes can make a smooth ascent direction temporarily
    # harmful, so regressions restore the last best exact checkpoint.
    for attempt in range(4):
        if attempt == 0:
            x = best_points.copy()
        else:
            noise = rng.normal(size=(n, dim))
            noise -= noise.mean(axis=0, keepdims=True)
            noise -= (
                np.sum(noise * best_points) / np.sum(best_points * best_points)
            ) * best_points
            x = normalize(best_points + (0.010 + 0.004 * attempt) * noise)

        first = np.zeros_like(x)
        second = np.zeros_like(x)
        local_best = exact_ratio_squared(x)
        local_point = x.copy()
        step_scale = 1.0
        regressions = 0

        for step in range(850):
            t = step / 849.0
            beta = 260.0 + 850.0 * (t * t)
            grad = smooth_gradient(x, beta)

            first = 0.90 * first + 0.10 * grad
            second = 0.975 * second + 0.025 * grad * grad
            lr = step_scale * (0.0070 * (1.0 - t) + 0.00065)
            x = normalize(x + lr * first / (np.sqrt(second) + 1.0e-8))

            if step % 25 == 24 or step == 849:
                value = exact_ratio_squared(x)

                if value > local_best:
                    local_best = value
                    local_point = x.copy()

                if value > best_value:
                    best_value = value
                    best_points = x.copy()

                if value + 1.0e-12 < local_best:
                    regressions += 1
                    if regressions >= 2:
                        x = local_point.copy()
                        first.fill(0.0)
                        second *= 0.15
                        step_scale *= 0.62
                        regressions = 0
                else:
                    regressions = 0
                    step_scale = min(1.10, step_scale * 1.015)

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END