# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Deterministic two-tier optimization for fourteen points in R^3.

    The objective is the exact minimum squared pairwise distance divided by
    the exact maximum squared pairwise distance.  Translation and uniform
    scale are removed after every update.
    """
    n = 14
    rng = np.random.default_rng(730291)
    ii, jj = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - np.mean(x, axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(rms, 1.0e-14)

    def exact_score(x: np.ndarray) -> float:
        d = x[ii] - x[jj]
        q = np.einsum("ij,ij->i", d, d)
        return float(np.min(q) / np.max(q))

    def force(x: np.ndarray, beta: float) -> np.ndarray:
        """
        Projected gradient of log(smooth_min_squared / smooth_max_squared).
        """
        d = x[ii] - x[jj]
        q = np.einsum("ij,ij->i", d, d)

        lo = float(np.min(q))
        lw_raw = np.exp(-beta * (q - lo))
        lw = lw_raw / np.sum(lw_raw)
        smooth_lo = lo - np.log(np.sum(lw_raw)) / beta

        hi = float(np.max(q))
        hw_raw = np.exp(beta * (q - hi))
        hw = hw_raw / np.sum(hw_raw)
        smooth_hi = hi + np.log(np.sum(hw_raw)) / beta

        coeff = lw / max(smooth_lo, 1.0e-12) - hw / max(smooth_hi, 1.0e-12)
        pair_force = 2.0 * coeff[:, None] * d

        g = np.zeros_like(x)
        np.add.at(g, ii, pair_force)
        np.add.at(g, jj, -pair_force)

        g -= np.mean(g, axis=0, keepdims=True)
        g -= (np.sum(g * x) / max(np.sum(x * x), 1.0e-14)) * x
        return g

    def refine(seed: np.ndarray, count: int, stage: int) -> np.ndarray:
        """
        Projected momentum ascent.  Stage 0 is deliberately broad; later
        stages increasingly concentrate force on limiting contact pairs.
        """
        x = normalize(seed.copy())
        incumbent = x.copy()
        incumbent_value = exact_score(x)
        velocity = np.zeros_like(x)

        for step in range(count):
            t = step / max(count - 1, 1)

            if stage == 0:
                beta = 9.0 + 105.0 * t ** 1.55
                rate = 0.052 * (1.0 - 0.58 * t)
                momentum = 0.79
            elif stage == 1:
                beta = 42.0 + 310.0 * t ** 1.65
                rate = 0.027 * (1.0 - 0.62 * t)
                momentum = 0.82
            else:
                beta = 180.0 + 1050.0 * t ** 1.8
                rate = 0.0125 * (1.0 - 0.70 * t)
                momentum = 0.74

            g = force(x, beta)
            gn = np.sqrt(np.mean(np.sum(g * g, axis=1)))
            if gn > 1.0e-14:
                g /= gn

            velocity = momentum * velocity + (1.0 - momentum) * g
            velocity -= np.mean(velocity, axis=0, keepdims=True)
            velocity -= (
                np.sum(velocity * x) / max(np.sum(x * x), 1.0e-14)
            ) * x

            x = normalize(x + rate * velocity)

            if step % 8 == 7 or step == count - 1:
                value = exact_score(x)
                if value > incumbent_value:
                    incumbent_value = value
                    incumbent = x.copy()

        return incumbent

    def layered(radius: float, height: float, twist: float,
                pole: float, jitter: float, seed: int) -> np.ndarray:
        a = np.arange(6, dtype=float) * (np.pi / 3.0)
        low = np.column_stack((
            radius * np.cos(a),
            radius * np.sin(a),
            -height * np.ones(6),
        ))
        high = np.column_stack((
            radius * np.cos(a + twist),
            radius * np.sin(a + twist),
            height * np.ones(6),
        ))
        x = np.vstack((low, high, [[0.0, 0.0, -pole], [0.0, 0.0, pole]]))
        if jitter > 0.0:
            x += jitter * np.random.default_rng(seed).normal(size=(n, 3))
        return x

    seeds = []

    # Variants of the useful twelve-ring-plus-two-pole contact topology.
    parameters = (
        (1.00, 0.430, np.pi / 6.0, 1.300),
        (1.00, 0.472, 0.500,      1.355),
        (1.03, 0.450, 0.535,      1.350),
        (0.97, 0.505, 0.575,      1.390),
        (1.06, 0.405, 0.470,      1.290),
        (0.94, 0.545, 0.610,      1.430),
        (1.02, 0.485, np.pi / 6.0, 1.370),
    )
    for k, p in enumerate(parameters):
        seeds.append(layered(*p, jitter=0.018, seed=1901 + 41 * k))

    # Cube plus axial points provides a substantially different initial graph.
    cube = np.array(
        [[a, b, c] for a in (-1.0, 1.0)
                   for b in (-1.0, 1.0)
                   for c in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.diag((1.48, 1.66, 1.82))
    seeds.append(np.vstack((cube, axes, -axes)))

    # Seven antipodal spherical lines derived from six icosahedral lines.
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    lines = np.array([
        [0.0, 1.0, phi], [0.0, 1.0, -phi],
        [1.0, phi, 0.0], [1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [phi, 0.0, -1.0],
    ])
    lines /= np.linalg.norm(lines, axis=1, keepdims=True)
    trial = rng.normal(size=(8000, 3))
    trial /= np.linalg.norm(trial, axis=1, keepdims=True)
    seventh = trial[np.argmin(np.max(np.abs(trial @ lines.T), axis=1))]
    antipodal = np.vstack((lines, seventh))
    seeds.append(np.vstack((antipodal, -antipodal)))

    # A pair of unstructured starts retains access to non-layered basins.
    seeds.append(rng.normal(size=(n, 3)))
    seeds.append(rng.normal(size=(n, 3)))

    # First tier: inexpensive common continuation for all basins.
    first_tier = []
    for seed in seeds:
        candidate = refine(seed, 560, 0)
        first_tier.append((exact_score(candidate), candidate))

    first_tier.sort(key=lambda z: z[0], reverse=True)
    survivors = [x.copy() for _, x in first_tier[:5]]

    # Second tier: refine only the strongest basins and nearby broken-symmetry
    # variants.  Exact score is always used for survivor selection.
    finalists = []
    perturb_rng = np.random.default_rng(493817)
    for rank, survivor in enumerate(survivors):
        finalists.append(refine(survivor, 1080, 1))
        amplitude = 0.018 + 0.008 * rank
        broken = survivor + amplitude * perturb_rng.normal(size=(n, 3))
        finalists.append(refine(broken, 820, 1))

    finalists.sort(key=exact_score, reverse=True)
    best = finalists[0].copy()
    best_value = exact_score(best)

    # Use several final continuation paths.  The limiting contact graph is
    # sensitive to when the soft extrema become sharp, so a moderate branch
    # and tiny deterministic symmetry breaks complement the direct sharp pass.
    final_rng = np.random.default_rng(904117)
    for rank, candidate in enumerate(finalists[:3]):
        branches = [
            refine(candidate, 1180, 2),
            refine(refine(candidate, 680, 1), 900, 2),
        ]

        # This perturbation is much smaller than the earlier basin-search
        # noise: it changes tied contacts without discarding the established
        # packing structure.
        tangent_noise = final_rng.normal(size=(n, 3))
        tangent_noise -= tangent_noise.mean(axis=0, keepdims=True)
        tangent_noise -= (
            np.sum(tangent_noise * candidate)
            / max(np.sum(candidate * candidate), 1.0e-14)
        ) * candidate
        tangent_noise /= max(
            np.sqrt(np.mean(np.sum(tangent_noise * tangent_noise, axis=1))),
            1.0e-14,
        )
        branches.append(refine(
            candidate + (0.010 + 0.003 * rank) * tangent_noise, 1020, 2
        ))

        for polished in branches:
            value = exact_score(polished)
            if value > best_value:
                best_value = value
                best = polished

    return np.asarray(best, dtype=np.float64)


# EVOLVE-BLOCK-END