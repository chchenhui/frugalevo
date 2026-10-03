# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 reproducible points in R^3 maximizing the minimum
    pairwise-distance / diameter ratio.

    The implementation uses a portfolio of structurally distinct starts,
    inexpensive broad scouting, selective sharp refinement, and an exact
    hard-objective finishing pass.
    """
    n, d = 14, 3
    rng = np.random.default_rng(81472931)
    iu, ju = np.triu_indices(n, 1)
    pair_count = len(iu)

    def normalize(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float).copy()
        points -= points.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(points * points))
        if rms < 1.0e-14:
            points = rng.normal(size=(n, d))
            points -= points.mean(axis=0, keepdims=True)
            rms = np.sqrt(np.mean(points * points))
        return points / rms

    def pair_squares(points: np.ndarray) -> np.ndarray:
        differences = points[iu] - points[ju]
        return np.einsum("ij,ij->i", differences, differences)

    def exact_ratio(points: np.ndarray) -> float:
        dsq = pair_squares(points)
        return float(np.min(dsq) / np.max(dsq))

    def smooth_score(points: np.ndarray, beta: float = 180.0) -> float:
        dsq = pair_squares(points)
        logs = 0.5 * np.log(dsq + 1.0e-15)

        lower = -beta * logs
        lower_max = lower.max()
        soft_min = -(lower_max + np.log(np.exp(lower - lower_max).sum())) / beta

        upper = beta * logs
        upper_max = upper.max()
        soft_max = (upper_max + np.log(np.exp(upper - upper_max).sum())) / beta
        return float(soft_min - soft_max)

    def smooth_gradient(points: np.ndarray, beta: float) -> np.ndarray:
        delta = points[iu] - points[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        log_dist = 0.5 * np.log(dsq + 1.0e-15)

        lo = -beta * log_dist
        lo -= lo.max()
        min_weight = np.exp(lo)
        min_weight /= min_weight.sum()

        hi = beta * log_dist
        hi -= hi.max()
        max_weight = np.exp(hi)
        max_weight /= max_weight.sum()

        weights = min_weight - max_weight
        pair_force = weights[:, None] * delta / (dsq[:, None] + 1.0e-15)

        gradient = np.zeros_like(points)
        np.add.at(gradient, iu, pair_force)
        np.add.at(gradient, ju, -pair_force)
        return gradient

    def adam_ascent(
        points: np.ndarray,
        steps: int,
        beta_begin: float,
        beta_end: float,
        rate_begin: float,
        rate_end: float,
    ) -> np.ndarray:
        """Optimize a log-distance softmin minus softmax continuation."""
        points = normalize(points)
        first = np.zeros_like(points)
        second = np.zeros_like(points)

        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            beta = beta_begin + (beta_end - beta_begin) * fraction
            gradient = smooth_gradient(points, beta)

            first = 0.90 * first + 0.10 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.90 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            rate = rate_begin + (rate_end - rate_begin) * fraction
            points += rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            points = normalize(points)

        return points

    def icosahedral_seed(extra_scale: float) -> np.ndarray:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        base = np.array(
            [
                (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
                (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
                (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
            ],
            dtype=float,
        )
        extras = rng.normal(size=(2, 3))
        extras /= np.linalg.norm(extras, axis=1, keepdims=True)
        extras *= extra_scale
        return normalize(np.vstack((base, extras)) + 0.045 * rng.normal(size=(n, d)))

    def staggered_ring_seed(jitter: float) -> np.ndarray:
        counts = (5, 4, 5)
        heights = (-0.82, 0.0, 0.82)
        radii = (0.73, 1.0, 0.73)
        rings = []
        for level, (count, z, radius) in enumerate(zip(counts, heights, radii)):
            phase = (level * 0.37 + rng.uniform(-0.13, 0.13)) * np.pi
            angle = phase + 2.0 * np.pi * np.arange(count) / count
            rings.append(
                np.column_stack(
                    (radius * np.cos(angle), radius * np.sin(angle),
                     np.full(count, z))
                )
            )
        return normalize(np.vstack(rings) + jitter * rng.normal(size=(n, d)))

    # A deliberately heterogeneous seed portfolio prevents the search from
    # being dominated by one symmetry class.
    seeds = []
    for _ in range(8):
        seeds.append(normalize(rng.normal(size=(n, d))))
    for scale in (0.55, 0.75, 0.95, 1.15, 1.35, 1.60):
        seeds.append(icosahedral_seed(scale))
    for jitter in (0.025, 0.045, 0.070, 0.095, 0.130, 0.170):
        seeds.append(staggered_ring_seed(jitter))

    # Broad, low-cost continuation ranks the portfolio before spending the
    # high-beta contact-resolution budget.
    scouted = []
    for seed in seeds:
        candidate = adam_ascent(
            seed,
            steps=2500,
            beta_begin=4.0,
            beta_end=90.0,
            rate_begin=0.030,
            rate_end=0.007,
        )
        scouted.append(candidate)

    ranking = sorted(
        range(len(scouted)),
        key=lambda k: (smooth_score(scouted[k], 220.0), exact_ratio(scouted[k])),
        reverse=True,
    )
    finalists = [scouted[k] for k in ranking[:9]]

    best_points = None
    best_ratio = -np.inf

    for initial in finalists:
        points = adam_ascent(
            initial,
            steps=5600,
            beta_begin=85.0,
            beta_end=820.0,
            rate_begin=0.010,
            rate_end=0.0018,
        )

        # Exact objective search is intentionally decoupled from the smooth
        # optimizer: accepted moves must improve the evaluator's true metric.
        current = exact_ratio(points)
        scale = 0.011
        stale = 0

        for outer in range(520):
            chosen = points
            chosen_ratio = current

            # Mixture of global isotropic and contact-scale perturbations.
            for proposal in range(6):
                noise = rng.normal(size=(n, d))
                if proposal >= 4:
                    noise *= 0.38
                trial = normalize(points + scale * noise)
                value = exact_ratio(trial)
                if value > chosen_ratio:
                    chosen = trial
                    chosen_ratio = value

            if chosen_ratio > current:
                points = chosen
                current = chosen_ratio
                scale = min(0.024, scale * 1.055)
                stale = 0
            else:
                stale += 1
                if stale % 11 == 0:
                    scale *= 0.61
                if scale < 1.5e-6:
                    break

        if current > best_ratio:
            best_ratio = current
            best_points = points.copy()

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END