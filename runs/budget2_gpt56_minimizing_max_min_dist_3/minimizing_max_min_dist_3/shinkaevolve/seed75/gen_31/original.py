# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in R^3 with a large minimum-pair-distance to
    diameter ratio.  Translation and uniform scaling are normalized away.
    """
    n = 14
    rng = np.random.default_rng(918273645)
    ii, jj = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - x.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(scale, 1.0e-14)

    def distances_sq(x: np.ndarray) -> np.ndarray:
        d = x[ii] - x[jj]
        return np.einsum("ij,ij->i", d, d)

    def ratio_sq(x: np.ndarray) -> float:
        q = distances_sq(x)
        return float(q.min() / q.max())

    def sphere_start() -> np.ndarray:
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return normalize(x)

    def gaussian_start() -> np.ndarray:
        return normalize(rng.normal(size=(n, 3)))

    def antiprism_start(phase: float, height: float) -> np.ndarray:
        k = np.arange(7, dtype=float)
        a = 2.0 * np.pi * k / 7.0
        r = np.sqrt(max(1.0e-12, 1.0 - height * height))
        top = np.column_stack((r * np.cos(a), r * np.sin(a),
                               np.full(7, height)))
        bottom = np.column_stack((r * np.cos(a + phase),
                                  r * np.sin(a + phase),
                                  np.full(7, -height)))
        return normalize(np.vstack((top, bottom)))

    def fibonacci_start(offset: float) -> np.ndarray:
        k = np.arange(n, dtype=float)
        z = 1.0 - 2.0 * (k + 0.5) / n
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        phi = np.pi * (3.0 - np.sqrt(5.0))
        a = phi * k + offset
        return normalize(np.column_stack((r * np.cos(a), r * np.sin(a), z)))

    starts = [
        fibonacci_start(0.0),
        fibonacci_start(0.31),
        fibonacci_start(0.73),
        antiprism_start(np.pi / 7.0, 0.34),
        antiprism_start(np.pi / 7.0, 0.48),
        antiprism_start(np.pi / 6.0, 0.39),
    ]
    starts.extend(sphere_start() for _ in range(7))
    starts.extend(gaussian_start() for _ in range(5))

    best = starts[0].copy()
    best_value = ratio_sq(best)

    # Explicit contact-graph relaxation.  At early stages broad bands of
    # near-short and near-long pairs are used; later stages approach the
    # nonsmooth exact min/max objective.
    stages = [
        (0.24, 260),
        (0.15, 330),
        (0.085, 420),
        (0.042, 500),
        (0.018, 600),
        (0.007, 650),
    ]

    for start_index, start in enumerate(starts):
        x = start.copy()
        momentum = np.zeros_like(x)
        velocity_sq = np.zeros_like(x)

        for band, steps in stages:
            for step_index in range(steps):
                q = distances_sq(x)
                qlo = float(np.min(q))
                qhi = float(np.max(q))

                # Active lower and upper contact sets.  The inverse-distance
                # factors are the subgradient of log(qlo / qhi).
                low = q <= qlo * (1.0 + band)
                high = q >= qhi * (1.0 - band)

                coeff = np.zeros(len(ii), dtype=float)
                coeff[low] = 1.0 / (max(1, int(np.sum(low))) * qlo)
                coeff[high] -= 1.0 / (max(1, int(np.sum(high))) * qhi)

                # Assemble graph-Laplacian contact forces without pairwise
                # scatter operations.
                cmat = np.zeros((n, n), dtype=float)
                cmat[ii, jj] = coeff
                cmat[jj, ii] = coeff
                grad = 2.0 * ((cmat.sum(axis=1)[:, None] * x) - cmat @ x)

                # Remove translation/radial components before the projected
                # update; normalization handles the remaining scale freedom.
                grad -= grad.mean(axis=0, keepdims=True)
                grad -= np.sum(grad * x) / np.sum(x * x) * x

                momentum = 0.88 * momentum + 0.12 * grad
                velocity_sq = 0.992 * velocity_sq + 0.008 * grad * grad

                local_t = step_index / max(1, steps - 1)
                step_size = (0.038 * (1.0 - local_t) + 0.004) * (
                    0.65 + 0.35 * band / 0.24
                )
                x = normalize(
                    x + step_size * momentum / (np.sqrt(velocity_sq) + 2.0e-8)
                )

                # Small reproducible basin changes only while broad contact
                # sets are active.
                if band >= 0.085 and step_index % 95 == 0 and step_index > 0:
                    kick = rng.normal(size=x.shape)
                    kick -= kick.mean(axis=0, keepdims=True)
                    x = normalize(x + 0.010 * band / 0.24 * kick)

                if step_index % 35 == 0:
                    value = ratio_sq(x)
                    if value > best_value:
                        best_value = value
                        best = x.copy()

        value = ratio_sq(x)
        if value > best_value:
            best_value = value
            best = x.copy()

    # Exact objective polishing: accepted coordinate perturbations are judged
    # exclusively by the evaluator's squared ratio, not by a surrogate.
    x = best.copy()
    current = best_value
    sigma = 0.030

    for trial in range(5500):
        candidate = x.copy()
        point = trial % n
        candidate[point] += rng.normal(scale=sigma, size=3)
        candidate = normalize(candidate)
        value = ratio_sq(candidate)

        if value > current:
            x = candidate
            current = value
            if value > best_value:
                best_value = value
                best = value * 0.0 + x
        if (trial + 1) % 550 == 0:
            sigma *= 0.66

    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END