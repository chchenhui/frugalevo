# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 three-dimensional points as seven antipodal pairs.

    The antipodal construction fixes the diameter at 2 while reducing the
    problem to minimizing the largest absolute inner product among seven
    unit vectors (a spherical line-packing problem).
    """
    rng = np.random.default_rng(20250308)
    n_lines = 7

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def coherence(u: np.ndarray) -> float:
        gram = np.abs(u @ u.T)
        np.fill_diagonal(gram, 0.0)
        return float(np.max(gram))

    # Six icosahedral lines have especially low mutual coherence in R^3.
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    base6 = np.array(
        [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [phi, 0.0, -1.0],
        ],
        dtype=float,
    )
    base6 = normalize_rows(base6)

    # Deterministically select a good seventh line before jointly refining
    # all seven lines.  Sampling is vectorized and therefore inexpensive.
    trial = rng.normal(size=(24000, 3))
    trial = normalize_rows(trial)
    initial_cost = np.max(np.abs(trial @ base6.T), axis=1)
    seventh = trial[np.argmin(initial_cost)]
    seed = np.vstack((base6, seventh))

    best = seed.copy()
    best_value = coherence(best)

    # Smooth max continuation: progressively sharpen log-sum-exp of squared
    # correlations.  Gradients are projected to each sphere's tangent plane.
    betas = (10.0, 25.0, 60.0, 140.0, 320.0, 700.0)
    steps = (150, 180, 220, 260, 280, 320)
    rates = (0.055, 0.042, 0.030, 0.020, 0.012, 0.007)

    # A few small, reproducible perturbations explore distinct basins while
    # retaining the strong icosahedral starting geometry.
    for restart in range(8):
        if restart == 0:
            u = seed.copy()
        else:
            noise_scale = 0.045 + 0.025 * restart
            u = normalize_rows(seed + noise_scale * rng.normal(size=(n_lines, 3)))

        for beta, count, rate in zip(betas, steps, rates):
            velocity = np.zeros_like(u)

            for iteration in range(count):
                dots = u @ u.T
                sq = dots * dots
                np.fill_diagonal(sq, -np.inf)

                # Stable softmax weights over unordered line pairs.
                peak = np.max(sq)
                weights = np.exp(beta * (sq - peak))
                np.fill_diagonal(weights, 0.0)
                weights /= np.sum(weights)

                # Gradient of the soft maximum of squared correlations.
                grad = 2.0 * ((weights * dots) @ u)

                # Riemannian projection onto tangent spaces of S^2.
                grad -= np.sum(grad * u, axis=1, keepdims=True) * u

                # Light momentum improves movement through nearly symmetric
                # configurations without changing the deterministic result.
                velocity = 0.78 * velocity + grad
                velocity -= np.sum(velocity * u, axis=1, keepdims=True) * u
                u = normalize_rows(u - rate * velocity)

                if iteration % 20 == 0 or iteration == count - 1:
                    value = coherence(u)
                    if value < best_value:
                        best_value = value
                        best = u.copy()

    # The line packing is an excellent seed, but the optimum for fourteen
    # points need not be antipodal.  Refine it in the full 42-dimensional
    # point configuration space, retaining the antipodal arrangement as an
    # incumbent in case a symmetry-breaking restart is unhelpful.
    seed_points = np.vstack((best, -best))
    pair_i, pair_j = np.triu_indices(14, 1)
    pair_count = len(pair_i)

    def normalize_configuration(x: np.ndarray) -> np.ndarray:
        x = x - np.mean(x, axis=0, keepdims=True)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))

    def actual_ratio(x: np.ndarray) -> float:
        delta = x[pair_i] - x[pair_j]
        distances_squared = np.sum(delta * delta, axis=1)
        return float(np.min(distances_squared) / np.max(distances_squared))

    points = normalize_configuration(seed_points)
    best_ratio = actual_ratio(points)

    # log-mean-exp versions of the minimum and maximum are smooth, positive
    # approximations with gradients concentrated on the active contact pairs.
    # Increasing beta progressively changes broad rearrangements into contact
    # refinement.
    betas = (18.0, 45.0, 110.0, 260.0, 600.0)
    rates = (0.030, 0.022, 0.015, 0.010, 0.006)
    iterations = (130, 150, 180, 210, 240)

    for restart in range(12):
        if restart == 0:
            x = points.copy()
        else:
            # Independent perturbations deliberately destroy pairwise
            # antipodality while remaining in the basin of the strong seed.
            scale = 0.025 + 0.012 * restart
            x = normalize_configuration(
                seed_points + scale * rng.normal(size=seed_points.shape)
            )

        velocity = np.zeros_like(x)
        for beta, rate, count in zip(betas, rates, iterations):
            for iteration in range(count):
                delta = x[pair_i] - x[pair_j]
                values = np.sum(delta * delta, axis=1)

                low_shift = np.min(values)
                low_weights = np.exp(-beta * (values - low_shift))
                low_weights /= np.sum(low_weights)
                smooth_min = low_shift - np.log(np.mean(
                    np.exp(-beta * (values - low_shift))
                )) / beta

                high_shift = np.max(values)
                high_weights = np.exp(beta * (values - high_shift))
                high_weights /= np.sum(high_weights)
                smooth_max = high_shift + np.log(np.mean(
                    np.exp(beta * (values - high_shift))
                )) / beta

                # Gradient of log(smooth_min / smooth_max).  Accumulating
                # pair forces directly avoids materializing a 14x14x3 tensor.
                coefficients = low_weights / smooth_min - high_weights / smooth_max
                pair_gradient = 2.0 * coefficients[:, None] * delta
                gradient = np.zeros_like(x)
                np.add.at(gradient, pair_i, pair_gradient)
                np.add.at(gradient, pair_j, -pair_gradient)

                velocity = 0.72 * velocity + gradient
                x = normalize_configuration(x + rate * velocity)

                if iteration % 20 == 0 or iteration == count - 1:
                    ratio = actual_ratio(x)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        points = x.copy()

    # Final exact-objective contact polishing.  The continuation objective is
    # necessarily slightly biased by its finite softmax temperature; here a
    # very sharp contact force proposes moves, but the true evaluator ratio
    # is the sole acceptance criterion.  Consequently every rejection is an
    # immediate rollback to the last exact-best configuration.
    x = points.copy()
    current_ratio = actual_ratio(x)
    step = 0.009
    velocity = np.zeros_like(x)
    sharp_beta = 1800.0

    for _ in range(1400):
        delta = x[pair_i] - x[pair_j]
        values = np.sum(delta * delta, axis=1)

        low_shift = np.min(values)
        low_weights = np.exp(-sharp_beta * (values - low_shift))
        low_weights /= np.sum(low_weights)

        high_shift = np.max(values)
        high_weights = np.exp(sharp_beta * (values - high_shift))
        high_weights /= np.sum(high_weights)

        # Use the exact extrema in the scale factors while distributing each
        # contact force smoothly among tied or nearly tied active pairs.
        coefficients = (
            low_weights / low_shift - high_weights / high_shift
        )
        pair_gradient = 2.0 * coefficients[:, None] * delta
        gradient = np.zeros_like(x)
        np.add.at(gradient, pair_i, pair_gradient)
        np.add.at(gradient, pair_j, -pair_gradient)

        velocity = 0.68 * velocity + gradient
        candidate = normalize_configuration(x + step * velocity)
        candidate_ratio = actual_ratio(candidate)

        if candidate_ratio > current_ratio + 1.0e-13:
            x = candidate
            current_ratio = candidate_ratio
            step = min(0.028, step * 1.018)
        else:
            # Keep x unchanged: this is an exact-best rollback.  Damping the
            # accumulated direction avoids repeatedly overshooting a narrow
            # active-contact basin.
            velocity *= 0.12
            step *= 0.52
            if step < 2.0e-8:
                break

    if current_ratio > best_ratio:
        points = x

    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END