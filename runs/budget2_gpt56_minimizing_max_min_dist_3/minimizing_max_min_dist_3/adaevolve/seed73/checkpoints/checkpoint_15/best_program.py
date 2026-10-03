# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize rhombic-dodecahedral perturbations by annealed soft ascent,
    then polish the best exact squared-distance ratio by direct local search."""
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

    # Polish the exact nonsmooth objective.  Vertices belonging to an active
    # shortest or longest pair are substantially more useful mutation targets
    # than uniformly cycling through all vertices.  A small annealed acceptance
    # probability helps cross the sharp local ridges of the true objective.
    current = best.copy()
    current_value = best_value
    radius = 0.005
    for trial in range(70000):
        delta = current[ii] - current[jj]
        q = np.einsum("ij,ij->i", delta, delta)
        lo, hi = q.min(), q.max()

        # Include nearly active pairs too: exact ties are often split by roundoff
        # after a single accepted mutation.
        active = (q <= lo * 1.0015) | (q >= hi * 0.9985)
        vertices = np.unique(np.concatenate((ii[active], jj[active])))
        p = vertices[rng.integers(len(vertices))]

        candidate = current.copy()
        candidate[p] += radius * rng.standard_normal(3)
        candidate = normalize(candidate)
        value = ratio_squared(candidate)

        # The temperature is expressed in the scale of the squared ratio.
        phase = trial % 17500
        temperature = 2.5e-5 * (0.03 ** (phase / 17499.0))
        if value >= current_value or rng.random() < np.exp(
            (value - current_value) / temperature
        ):
            current = candidate
            current_value = value
            radius = min(0.020, radius * 1.006)
            if value > best_value:
                best = candidate.copy()
                best_value = value
        else:
            radius = max(1.5e-5, radius * 0.99955)

        # Restart each annealing epoch from the incumbent, retaining all gains.
        if phase == 17499:
            current = best.copy()
            current_value = best_value
            radius = 0.004

    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END
