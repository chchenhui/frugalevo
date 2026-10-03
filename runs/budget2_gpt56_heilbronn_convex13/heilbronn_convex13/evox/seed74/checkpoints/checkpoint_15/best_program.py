# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize a 13-point, three-fold-symmetric arrangement.

    The convex container is an equilateral triangle.  The points comprise its
    three vertices, the centre, and three independently parameterized rotating
    three-point orbits.  A fixed-seed annealed coordinate search maximizes the
    smallest triangle area normalized by the hull area.
    """
    rng = np.random.default_rng(13031957)

    # The area ratio is invariant under scaling, so this equilateral hull need
    # not explicitly be scaled to unit area.
    outer_radius = 1.0 / np.sqrt(3.0)
    angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(3) / 3.0
    outer = outer_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    centre = np.array([[0.0, 0.0]])

    triples = np.array([
        (i, j, k)
        for i in range(13)
        for j in range(i + 1, 13)
        for k in range(j + 1, 13)
    ], dtype=np.intp)

    hull_det = abs(np.cross(outer[1] - outer[0], outer[2] - outer[0]))
    bary_inverse = np.linalg.inv(
        np.column_stack((outer[0] - outer[2], outer[1] - outer[2]))
    )

    def orbit(radius: float, phase: float) -> np.ndarray:
        orbit_angles = phase + 2.0 * np.pi * np.arange(3) / 3.0
        return radius * np.column_stack((np.cos(orbit_angles),
                                         np.sin(orbit_angles)))

    def inside_hull(radius: float, phase: float) -> bool:
        p = np.array([radius * np.cos(phase), radius * np.sin(phase)])
        b = bary_inverse @ (p - outer[2])
        return bool(b[0] >= -1e-12 and b[1] >= -1e-12
                    and b[0] + b[1] <= 1.0 + 1e-12)

    def make_points(state: np.ndarray) -> np.ndarray:
        return np.vstack((
            centre,
            outer,
            orbit(state[0], state[1]),
            orbit(state[2], state[3]),
            orbit(state[4], state[5]),
        ))

    def low_areas(state: np.ndarray) -> np.ndarray:
        """Return the fourteen smallest normalized triangle areas efficiently."""
        p = make_points(state)
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        det = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ) / hull_det
        return np.partition(det, 13)[:14]

    def minimum_area(state: np.ndarray) -> float:
        """Compute the maximin objective without sorting all 286 triangles."""
        p = make_points(state)
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        det = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return float(np.min(det) / hull_det)

    def merit(state: np.ndarray) -> float:
        """Use a low-tail surrogate to stabilize annealed maximin search."""
        low = low_areas(state)
        return float(np.min(low) + 0.18 * np.mean(low))

    best_state = None
    best_minimum = -np.inf

    # The reduced six-dimensional model permits substantially broader,
    # reproducible multi-start coverage at modest cost.
    for _ in range(96):
        state = np.empty(6)
        for ring in range(3):
            phase = rng.uniform(0.0, 2.0 * np.pi / 3.0)
            limit = outer_radius
            while not inside_hull(limit, phase):
                limit *= 0.985
            state[2 * ring] = rng.uniform(0.055, max(0.056, 0.985 * limit))
            state[2 * ring + 1] = phase

        current = merit(state)
        for iteration in range(1600):
            trial = state.copy()
            coordinate = int(rng.integers(6))
            cooling = 1.0 - iteration / 1600.0

            if coordinate % 2 == 0:
                trial[coordinate] += rng.normal(
                    0.0, 0.105 * (0.12 + cooling)
                )
                trial[coordinate] = np.clip(
                    trial[coordinate], 0.035, outer_radius
                )
            else:
                trial[coordinate] = (
                    trial[coordinate]
                    + rng.normal(0.0, 0.30 * (0.12 + cooling))
                ) % (2.0 * np.pi / 3.0)

            ring = coordinate // 2
            if not inside_hull(trial[2 * ring], trial[2 * ring + 1]):
                continue

            candidate = merit(trial)
            temperature = 0.0014 * cooling * cooling + 1e-6
            if (candidate >= current or
                    rng.random() < np.exp((candidate - current) / temperature)):
                state = trial
                current = candidate

            minimum = minimum_area(state)
            if minimum > best_minimum:
                best_minimum = minimum
                best_state = state.copy()

    # Direct deterministic maximin polishing.  Annealing finds promising
    # basins; this shrinking coordinate search removes residual low-area
    # bottlenecks without changing the feasible symmetric construction.
    if best_state is not None:
        state = best_state.copy()
        current = minimum_area(state)
        radius_step = 0.018
        phase_step = 0.045
        while radius_step > 2e-5 or phase_step > 2e-5:
            improved = False
            for coordinate in range(6):
                for direction in (-1.0, 1.0):
                    trial = state.copy()
                    if coordinate % 2 == 0:
                        trial[coordinate] = np.clip(
                            trial[coordinate] + direction * radius_step,
                            0.035, outer_radius
                        )
                    else:
                        trial[coordinate] = (
                            trial[coordinate] + direction * phase_step
                        ) % (2.0 * np.pi / 3.0)

                    ring = coordinate // 2
                    if not inside_hull(trial[2 * ring], trial[2 * ring + 1]):
                        continue

                    candidate = minimum_area(trial)
                    if candidate > current + 1e-14:
                        state = trial
                        current = candidate
                        improved = True

            if improved:
                best_state = state.copy()
                best_minimum = current
            else:
                radius_step *= 0.5
                phase_step *= 0.5

    # Defensive valid deterministic fallback for unexpected numerical failure.
    if best_state is None:
        best_state = np.array([0.15, 0.10, 0.30, 0.45, 0.46, 0.80])

    return make_points(best_state)


# EVOLVE-BLOCK-END
