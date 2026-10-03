# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """Optimize a deterministic 13-point configuration using annealed soft-min search."""
    n = 13
    rng = np.random.default_rng(42)

    try:
        from scipy.optimize import minimize
        from scipy.spatial import ConvexHull
        from scipy.special import logsumexp
    except Exception:
        # Deterministic fallback if SciPy is unavailable.
        return np.array([
            [0.03, 0.08], [0.22, 0.04], [0.48, 0.03], [0.76, 0.06],
            [0.96, 0.20], [0.98, 0.52], [0.94, 0.84], [0.70, 0.97],
            [0.39, 0.98], [0.10, 0.88], [0.04, 0.61],
            [0.28, 0.28], [0.67, 0.29]
        ], dtype=float)

    # Low-discrepancy-like deterministic seeds, with a few independent jitters.
    base = np.array([
        [0.06, 0.08], [0.27, 0.05], [0.51, 0.04], [0.76, 0.07],
        [0.94, 0.22], [0.97, 0.54], [0.91, 0.82], [0.68, 0.95],
        [0.40, 0.97], [0.13, 0.87], [0.04, 0.59],
        [0.28, 0.29], [0.67, 0.30]
    ], dtype=float)

    pairs = np.array([(i, j, k) for i in range(n)
                      for j in range(i + 1, n)
                      for k in range(j + 1, n)], dtype=int)

    def triangle_areas(x):
        p = x.reshape(n, 2)
        a = p[pairs[:, 0]]
        b = p[pairs[:, 1]]
        c = p[pairs[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def normalized_min(x):
        p = x.reshape(n, 2)
        try:
            hull_area = ConvexHull(p).volume
        except Exception:
            return 0.0
        if hull_area <= 1e-10:
            return 0.0
        return float(np.min(triangle_areas(x)) / hull_area)

    def objective(x, temperature):
        p = x.reshape(n, 2)
        try:
            hull_area = ConvexHull(p).volume
        except Exception:
            return 1e3
        if hull_area <= 1e-10:
            return 1e3
        areas = triangle_areas(x) / hull_area
        # Soft minimum gives useful gradients while preserving the true
        # minimum as the temperature is reduced.
        return float(temperature * logsumexp(-areas / temperature))

    best = base.copy()
    best_score = normalized_min(best.ravel())

    for restart in range(7):
        if restart == 0:
            start = base.copy()
        else:
            start = np.clip(
                base + rng.normal(0.0, 0.035 + 0.008 * restart, base.shape),
                0.015, 0.985
            )

        x = start.ravel()
        # Continuation from a smooth objective to an increasingly accurate
        # approximation of the minimum triangle area.
        # Continue to substantially colder temperatures so that the final
        # iterates are governed by the actual worst triangles rather than by
        # a broad average of near-worst triangles.
        for temperature in (0.004, 0.002, 0.001, 0.0005, 0.00025,
                            0.00012, 0.00006):
            result = minimize(
                lambda z, t=temperature: objective(z, t),
                x,
                method="L-BFGS-B",
                bounds=[(0.002, 0.998)] * (2 * n),
                options={"maxiter": 450 if temperature <= 0.00025 else 550,
                         "ftol": 1e-13, "gtol": 1e-8, "maxls": 40}
            )
            if result.success or np.isfinite(result.fun):
                x = result.x

        score = normalized_min(x)
        if score > best_score:
            best_score = score
            best = x.reshape(n, 2).copy()

    # Polish the soft-min result against the exact nonsmooth objective.
    # This combines deterministic coordinate-direction probing with seeded
    # stochastic multi-point moves and adaptive step sizes.  The incumbent is
    # never discarded, so polishing cannot reduce the reported score.
    exact_score = normalized_min(best.ravel())
    polish_rng = np.random.default_rng(314159)
    stagnation = 0
    iterations = 24000

    for iteration in range(iterations):
        fraction = iteration / float(iterations)
        scale = 0.014 * (1.0 - fraction) ** 1.35 + 0.00012
        trial = best.copy()

        # Probe axis-aligned directions regularly.  These moves are useful
        # because the active smallest triangles often change piecewise-linearly
        # with one coordinate.
        if iteration % 5 in (0, 1, 2, 3):
            index = (iteration // 5) % n
            axis = (iteration // (5 * n)) % 2
            direction = 1.0 if ((iteration // (5 * n * 2)) % 2 == 0) else -1.0
            trial[index, axis] += direction * scale
        elif iteration % 17 == 0:
            # Correlated motion of two points helps escape configurations where
            # two or more near-degenerate triangles are active simultaneously.
            indices = polish_rng.choice(n, size=2, replace=False)
            trial[indices] += polish_rng.normal(0.0, scale, (2, 2))
        else:
            index = int(polish_rng.integers(n))
            trial[index] += polish_rng.normal(0.0, scale, 2)

        trial = np.clip(trial, 0.001, 0.999)
        trial_score = normalized_min(trial.ravel())

        if trial_score > exact_score:
            best = trial
            exact_score = trial_score
            stagnation = 0
        else:
            stagnation += 1

        # A deterministic, bounded kick prevents long periods of rejecting
        # proposals after the active triangle set changes.
        if stagnation >= 1800:
            trial = best.copy()
            indices = np.array([
                (iteration // 7) % n,
                (iteration // 11 + 3) % n,
                (iteration // 17 + 6) % n,
            ])
            kick_scale = min(0.025, 6.0 * scale)
            trial[indices] += polish_rng.normal(
                0.0, kick_scale, (len(indices), 2)
            )
            trial = np.clip(trial, 0.001, 0.999)
            trial_score = normalized_min(trial.ravel())
            if trial_score > exact_score:
                best = trial
                exact_score = trial_score
            stagnation = 0

    # Remove insignificant numerical boundary excursions while retaining
    # deterministic output and exactly 13 unique points.
    best = np.clip(best, 0.0, 1.0)
    return best


# EVOLVE-BLOCK-END
