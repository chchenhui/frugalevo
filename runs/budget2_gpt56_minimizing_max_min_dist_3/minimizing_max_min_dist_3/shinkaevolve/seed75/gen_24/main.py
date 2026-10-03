# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct a deterministic 14-point three-dimensional distance code."""

    n = 14
    iu, ju = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(scale, 1e-15)

    def ratio2(x: np.ndarray) -> float:
        delta = x[iu] - x[ju]
        q = np.sum(delta * delta, axis=1)
        return float(q.min() / q.max())

    def active_polish(x: np.ndarray, rounds) -> tuple[np.ndarray, float]:
        """Accept-only polishing on adaptively weighted near-contact pairs."""
        x = normalize(x)
        best_x = x.copy()
        best_value = ratio2(x)

        for tolerance, base_step, iterations in rounds:
            for _ in range(iterations):
                delta = x[iu] - x[ju]
                q = np.sum(delta * delta, axis=1)
                d = np.sqrt(np.maximum(q, 1e-15))
                dmin = d.min()
                dmax = d.max()

                low_mask = d <= dmin * (1.0 + tolerance)
                high_mask = d >= dmax * (1.0 - tolerance)

                # Smooth weights within the selected active sets.  This avoids
                # overreacting to marginal contacts while emphasizing limiting
                # constraints.
                weights = np.zeros_like(d)
                if np.any(low_mask):
                    lw = np.exp(-(d[low_mask] / dmin - 1.0) /
                                max(0.35 * tolerance, 1e-5))
                    weights[low_mask] = lw / lw.sum()
                if np.any(high_mask):
                    hw = np.exp(-(1.0 - d[high_mask] / dmax) /
                                max(0.35 * tolerance, 1e-5))
                    weights[high_mask] -= hw / hw.sum()

                pair_force = weights[:, None] * delta / q[:, None]
                force = np.zeros_like(x)
                np.add.at(force, iu, pair_force)
                np.add.at(force, ju, -pair_force)
                force -= force.mean(axis=0, keepdims=True)

                norm_force = np.sqrt(np.mean(np.sum(force * force, axis=1)))
                if norm_force < 1e-15:
                    break

                accepted = False
                for multiplier in (1.0, 0.55, 0.28, 0.14, 0.07):
                    trial = normalize(x + base_step * multiplier *
                                      force / norm_force)
                    value = ratio2(trial)
                    if value > best_value + 1e-14:
                        x = trial
                        best_x = trial.copy()
                        best_value = value
                        accepted = True
                        break
                if not accepted:
                    break

        return best_x, best_value

    cube = np.array(
        [[a, b, c]
         for a in (-1.0, 1.0)
         for b in (-1.0, 1.0)
         for c in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.array(
        [
            [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    rng = np.random.default_rng(730184)
    best = None
    best_value = -np.inf

    # The radii span cube-dominated, balanced, and axis-dominated basins.
    radii = (1.20, 1.38, 1.56, 1.73, 1.90, 2.08, 2.28)
    smooth_schedule = (
        (3.0, 130, 0.080),
        (6.0, 185, 0.058),
        (12.0, 245, 0.041),
        (24.0, 290, 0.028),
        (48.0, 325, 0.018),
        (96.0, 260, 0.011),
    )

    for axial_radius in radii:
        x = np.vstack((cube, axial_radius * axes))
        x += rng.normal(scale=0.105, size=x.shape)
        x = normalize(x)
        velocity = np.zeros_like(x)

        # Continuation from broad global separation forces to nearly exact
        # closest/farthest-contact forces.
        for power, iterations, step in smooth_schedule:
            momentum = 0.72 if power <= 12.0 else 0.52

            for _ in range(iterations):
                delta = x[iu] - x[ju]
                q = np.sum(delta * delta, axis=1)
                d = np.sqrt(np.maximum(q, 1e-15))
                dmin = d.min()
                dmax = d.max()

                low = (d / dmin) ** (-power)
                high = (d / dmax) ** power
                weights = low / low.sum() - high / high.sum()

                pair_gradient = weights[:, None] * delta / q[:, None]
                gradient = np.zeros_like(x)
                np.add.at(gradient, iu, pair_gradient)
                np.add.at(gradient, ju, -pair_gradient)
                gradient -= gradient.mean(axis=0, keepdims=True)

                gnorm = np.sqrt(np.mean(np.sum(gradient * gradient, axis=1)))
                if gnorm > 1e-15:
                    direction = gradient / gnorm
                    velocity = momentum * velocity + (1.0 - momentum) * direction
                    vnorm = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
                    x = normalize(x + step * velocity / max(vnorm, 1e-15))

        candidate, candidate_value = active_polish(
            x,
            (
                (0.040, 0.031, 170),
                (0.018, 0.017, 225),
                (0.007, 0.0085, 285),
            ),
        )

        if candidate_value > best_value:
            best = candidate.copy()
            best_value = candidate_value

    # Deterministic local coordinate search resolves some nonsmooth contact
    # changes which cannot be reached by a single active-set gradient.
    x = best.copy()
    value = best_value
    sigma = 0.020
    accepted_window = 0

    for iteration in range(4200):
        point = int(rng.integers(n))
        trial = x.copy()
        trial[point] += rng.normal(size=3) * sigma
        trial = normalize(trial)
        trial_value = ratio2(trial)

        if trial_value > value + 1e-14:
            x = trial
            value = trial_value
            accepted_window += 1

        if (iteration + 1) % 140 == 0:
            if accepted_window >= 5:
                sigma = min(0.030, sigma * 1.16)
            elif accepted_window == 0:
                sigma *= 0.58
            else:
                sigma *= 0.83
            accepted_window = 0
            if sigma < 1e-5:
                sigma = 1e-5

    x, value = active_polish(
        x,
        (
            (0.012, 0.009, 220),
            (0.004, 0.0045, 280),
            (0.0015, 0.0018, 240),
        ),
    )

    if value > best_value:
        best = x

    return normalize(best)


# EVOLVE-BLOCK-END