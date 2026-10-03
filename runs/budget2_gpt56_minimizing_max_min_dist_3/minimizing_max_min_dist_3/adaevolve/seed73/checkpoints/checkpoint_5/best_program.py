# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Return the best 14-point configuration found by multiscale deterministic
    perturbations of a rhombic-dodecahedral seed and annealed smooth
    maximin-distance-ratio ascent."""
    rng = np.random.default_rng(20260912)
    n = 14
    ii, jj = np.triu_indices(n, 1)

    # Six axial vertices plus eight cube vertices are a strong symmetric
    # diameter-packing initialization (rhombic-dodecahedral arrangement).
    a = np.sqrt(3.0)
    axial = np.array([
        [ a, 0.0, 0.0], [-a, 0.0, 0.0],
        [0.0,  a, 0.0], [0.0, -a, 0.0],
        [0.0, 0.0,  a], [0.0, 0.0, -a],
    ])
    corners = np.array([
        [sx, sy, sz]
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ])
    seed = np.vstack((axial, corners))

    def normalize(x):
        x = x - x.mean(axis=0)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))

    def ratio_squared(x):
        delta = x[ii] - x[jj]
        q = np.einsum("ij,ij->i", delta, delta)
        return q.min() / q.max()

    best = normalize(seed)
    best_value = ratio_squared(best)

    # A range of perturbation sizes explores both nearby symmetry-breaking
    # basins and genuinely distinct configurations.  The exact symmetric
    # seed remains available in case no perturbation improves it.
    starts = [best]
    for noise in (0.045, 0.080, 0.120, 0.180, 0.260, 0.360):
        for _ in range(4):
            starts.append(
                normalize(seed + noise * rng.standard_normal((n, 3)))
            )

    for x in starts:
        x = x.copy()
        for it in range(4200):
            delta = x[ii] - x[jj]
            q = np.einsum("ij,ij->i", delta, delta)

            # Differentiable soft minimum and maximum of squared distances.
            tau = 0.045 * (0.12 ** (it / 4199.0))
            qmin = q.min()
            emin = np.exp(-(q - qmin) / tau)
            wmin = emin / emin.sum()
            smooth_min = qmin - tau * np.log(emin.sum())

            qmax = q.max()
            emax = np.exp((q - qmax) / tau)
            wmax = emax / emax.sum()
            smooth_max = qmax + tau * np.log(emax.sum())

            # Gradient of log(smooth_min / smooth_max).
            weights = wmin / smooth_min - wmax / smooth_max
            grad = np.zeros_like(x)
            contribution = 2.0 * weights[:, None] * delta
            np.add.at(grad, ii, contribution)
            np.add.at(grad, jj, -contribution)

            step = 0.030 * (1.0 - 0.70 * it / 4199.0)
            x = normalize(x + step * grad)

            value = ratio_squared(x)
            if value > best_value:
                best_value = value
                best = x.copy()

    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END
