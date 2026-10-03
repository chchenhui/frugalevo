# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen 3D points maximizing the exact squared ratio
    min_pairwise_distance_squared / max_pairwise_distance_squared.

    The method uses:
      1. Differential evolution on a strong geometric packing family.
      2. Full-dimensional active-contact trust-region polishing.
    """
    n = 14
    ii, jj = np.triu_indices(n, 1)
    rng = np.random.default_rng(18472639)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(x * x) / n)
        return x / max(scale, 1.0e-15)

    def distances_squared(x: np.ndarray) -> np.ndarray:
        d = x[ii] - x[jj]
        return np.einsum("ij,ij->i", d, d)

    def score(x: np.ndarray) -> float:
        dsq = distances_squared(x)
        return float(np.min(dsq) / np.max(dsq))

    def layered(params: np.ndarray) -> np.ndarray:
        """
        Six points on each of two staggered rings and two axial points.
        The parameters are radius, half-height, twist and pole distance.
        """
        radius, height, twist, pole = params
        a = np.arange(6, dtype=np.float64) * (np.pi / 3.0)

        lower = np.column_stack((
            radius * np.cos(a),
            radius * np.sin(a),
            np.full(6, -height),
        ))
        upper = np.column_stack((
            radius * np.cos(a + twist),
            radius * np.sin(a + twist),
            np.full(6, height),
        ))
        poles = np.array([[0.0, 0.0, -pole], [0.0, 0.0, pole]])
        return normalize(np.vstack((lower, upper, poles)))

    def parameter_score(p: np.ndarray) -> float:
        # Positivity is enforced by the evolutionary bounds.
        return score(layered(p))

    # Differential evolution is used only in the compact four-parameter
    # family.  This reliably identifies high-quality contact combinatorics
    # before unconstrained polishing breaks unnecessary symmetry.
    bounds_lo = np.array([0.68, 0.20, 0.26, 0.55])
    bounds_hi = np.array([1.28, 0.72, 0.79, 1.38])

    pop_size = 42
    population = bounds_lo + rng.random((pop_size, 4)) * (bounds_hi - bounds_lo)

    # Deliberately include known useful staggered-ring regions.
    population[:8] = np.array([
        [0.93, 0.40, np.pi / 6.0, 0.98],
        [0.95, 0.41, np.pi / 6.0, 1.00],
        [0.98, 0.43, np.pi / 6.0, 1.03],
        [0.90, 0.38, 0.48, 0.96],
        [0.96, 0.44, 0.55, 1.00],
        [0.99, 0.46, 0.52, 1.05],
        [0.88, 0.36, 0.60, 0.93],
        [1.02, 0.47, 0.47, 1.08],
    ])
    population = np.clip(population, bounds_lo, bounds_hi)
    values = np.array([parameter_score(p) for p in population])

    for generation in range(230):
        # A slowly reduced differential weight changes from basin exploration
        # to accurate contact-geometry fitting.
        f = 0.82 - 0.38 * generation / 229.0
        cross_probability = 0.86

        for k in range(pop_size):
            choices = np.delete(np.arange(pop_size), k)
            a, b, c = rng.choice(choices, size=3, replace=False)

            donor = population[a] + f * (population[b] - population[c])
            donor = np.clip(donor, bounds_lo, bounds_hi)

            mask = rng.random(4) < cross_probability
            mask[rng.integers(4)] = True
            trial = np.where(mask, donor, population[k])

            trial_value = parameter_score(trial)
            if trial_value >= values[k]:
                population[k] = trial
                values[k] = trial_value

        # Exact-coordinate local probes prevent differential evolution from
        # losing narrow symmetric optima near contact transitions.
        if generation % 20 == 19:
            elite = int(np.argmax(values))
            base = population[elite].copy()
            width = (bounds_hi - bounds_lo) * (
                0.050 * (1.0 - generation / 230.0) + 0.004
            )
            for coordinate in range(4):
                for sign in (-1.0, 1.0):
                    trial = base.copy()
                    trial[coordinate] = np.clip(
                        trial[coordinate] + sign * width[coordinate],
                        bounds_lo[coordinate],
                        bounds_hi[coordinate],
                    )
                    trial_value = parameter_score(trial)
                    if trial_value > values[elite]:
                        population[elite] = trial
                        values[elite] = trial_value
                        base = trial

    order = np.argsort(values)[::-1]
    seeds = [layered(population[k]) for k in order[:10]]

    # Add several small asymmetric releases of the strongest contact graph.
    base = seeds[0]
    for magnitude in (0.0025, 0.005, 0.009, 0.014):
        q = rng.standard_normal((n, 3))
        q -= q.mean(axis=0, keepdims=True)
        q -= np.sum(q * base) / np.sum(base * base) * base
        seeds.append(normalize(base + magnitude * q))

    def active_direction(x: np.ndarray, looseness: float) -> np.ndarray:
        """
        Subgradient-like force from the currently active shortest and
        farthest pair bands.  Unlike a global soft-min approximation, only
        geometrically relevant contacts are used.
        """
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        low = float(np.min(dsq))
        high = float(np.max(dsq))

        low_mask = dsq <= low * (1.0 + looseness)
        high_mask = dsq >= high * (1.0 - looseness)

        # Equal sharing over contact bands is more stable when contacts swap.
        coeff = np.zeros_like(dsq)
        coeff[low_mask] = 1.0 / max(int(np.sum(low_mask)), 1)
        coeff[high_mask] -= 1.0 / max(int(np.sum(high_mask)), 1)
        coeff /= np.maximum(dsq, 1.0e-14)

        force = 2.0 * coeff[:, None] * delta
        direction = np.zeros_like(x)
        np.add.at(direction, ii, force)
        np.add.at(direction, jj, -force)

        direction -= direction.mean(axis=0, keepdims=True)
        direction -= np.sum(direction * x) / np.sum(x * x) * x
        return direction

    best_points = None
    best_value = -np.inf

    for seed_index, seed in enumerate(seeds):
        x = seed.copy()
        local_value = score(x)
        local_best = x.copy()

        if local_value > best_value:
            best_value = local_value
            best_points = x.copy()

        trust = 0.040
        failed = 0

        for step in range(1250):
            progress = step / 1249.0
            looseness = 0.030 * (1.0 - progress) + 0.0012
            direction = active_direction(x, looseness)
            direction_norm = np.sqrt(np.mean(np.sum(direction * direction, axis=1)))

            if direction_norm > 1.0e-14:
                direction /= direction_norm

            # Exact trust-region line search, including a shorter reverse
            # step because active-set subgradients can change at boundaries.
            accepted = False
            step_sizes = trust * np.array(
                [1.0, 0.58, 0.31, 0.15, -0.16], dtype=np.float64
            )

            trial_best_value = local_value
            trial_best = x

            for alpha in step_sizes:
                candidate = normalize(x + alpha * direction)
                candidate_value = score(candidate)
                if candidate_value > trial_best_value + 1.0e-13:
                    trial_best_value = candidate_value
                    trial_best = candidate

            if trial_best_value > local_value + 1.0e-13:
                x = trial_best
                local_value = trial_best_value
                accepted = True
                failed = 0
                trust = min(0.055, trust * 1.10)

                if local_value > score(local_best):
                    local_best = x.copy()
                if local_value > best_value:
                    best_value = local_value
                    best_points = x.copy()
            else:
                failed += 1
                trust *= 0.72

            # Reproducible local escape moves are attempted only after the
            # active contact system has genuinely stalled.
            if failed >= 7:
                noise = rng.standard_normal((n, 3))
                noise -= noise.mean(axis=0, keepdims=True)
                noise -= np.sum(noise * x) / np.sum(x * x) * x

                # Keep perturbation mostly orthogonal to the active force.
                if direction_norm > 1.0e-14:
                    noise -= (
                        np.sum(noise * direction) / np.sum(direction * direction)
                    ) * direction

                noise_norm = np.sqrt(np.mean(np.sum(noise * noise, axis=1)))
                noise /= max(noise_norm, 1.0e-14)

                amplitude = 0.0025 + 0.008 * (1.0 - progress)
                candidate = normalize(local_best + amplitude * noise)
                candidate_value = score(candidate)

                if candidate_value >= local_value * 0.996:
                    x = candidate
                    local_value = candidate_value
                else:
                    x = local_best.copy()
                    local_value = score(x)

                trust = 0.018
                failed = 0

        final_value = score(local_best)
        if final_value > best_value:
            best_value = final_value
            best_points = local_best.copy()

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END