# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically anneal and coordinate-polish three three-fold point
    orbits inside an equilateral 13-point hull to maximize minimum area.
    """
    rng = np.random.default_rng(13031957)

    # Three hull vertices define a convex equilateral container.  Area ratios
    # are affine invariant, so no explicit unit-area scaling is necessary.
    outer_radius = 1.0 / np.sqrt(3.0)
    outer_angles = np.pi / 2.0 + 2.0 * np.pi * np.arange(3) / 3.0
    outer = outer_radius * np.column_stack((
        np.cos(outer_angles), np.sin(outer_angles)
    ))
    centre = np.array([[0.0, 0.0]])

    triples = np.array([
        (i, j, k)
        for i in range(13)
        for j in range(i + 1, 13)
        for k in range(j + 1, 13)
    ], dtype=np.intp)

    hull_det = abs(np.cross(outer[1] - outer[0], outer[2] - outer[0]))
    bary_matrix = np.column_stack((outer[0] - outer[2],
                                   outer[1] - outer[2]))
    bary_inverse = np.linalg.inv(bary_matrix)

    def orbit(radius: float, phase: float) -> np.ndarray:
        angles = phase + 2.0 * np.pi * np.arange(3) / 3.0
        return radius * np.column_stack((np.cos(angles), np.sin(angles)))

    def inside_hull(radius: float, phase: float) -> bool:
        point = np.array([radius * np.cos(phase), radius * np.sin(phase)])
        b01 = bary_inverse @ (point - outer[2])
        return bool(b01[0] >= -1e-12 and b01[1] >= -1e-12
                    and 1.0 - b01[0] - b01[1] >= -1e-12)

    def make_points(state: np.ndarray) -> np.ndarray:
        return np.vstack((
            centre,
            outer,
            orbit(state[0], state[1]),
            orbit(state[2], state[3]),
            orbit(state[4], state[5]),
        ))

    def low_areas(state: np.ndarray) -> np.ndarray:
        """Return the fourteen smallest normalized triangle areas."""
        points = make_points(state)
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        determinants = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ) / hull_det
        return np.partition(determinants, 13)[:14]

    def minimum_area(state: np.ndarray) -> float:
        """Compute the exact minimum normalized area over all 286 triangles."""
        points = make_points(state)
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        determinants = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return float(np.min(determinants) / hull_det)

    def merit(state: np.ndarray) -> float:
        """Favor the actual bottleneck while retaining a low-tail tie-breaker."""
        low = low_areas(state)
        return float(np.min(low) + 0.18 * np.mean(low))

    best_state = None
    best_minimum = -np.inf

    def global_objective(state: np.ndarray) -> float:
        """Penalty-constrained lower-tail objective for differential evolution."""
        for ring in range(3):
            if not inside_hull(state[2 * ring], state[2 * ring + 1]):
                # The magnitude keeps infeasible individuals decisively below
                # every valid arrangement while retaining a useful direction
                # toward the triangular container.
                point = np.array([
                    state[2 * ring] * np.cos(state[2 * ring + 1]),
                    state[2 * ring] * np.sin(state[2 * ring + 1]),
                ])
                bary = bary_inverse @ (point - outer[2])
                violation = max(
                    0.0, -bary[0], -bary[1], bary[0] + bary[1] - 1.0
                )
                return 1.0 + 100.0 * violation
        return -merit(state)

    # Population evolution moves all radii and phases together, unlike the
    # former single-coordinate annealer.  Independent fixed-seed runs offer
    # reproducible basin coverage without making the returned construction
    # dependent on platform entropy.
    bounds = [(0.035, outer_radius), (0.0, 2.0 * np.pi / 3.0)] * 3
    for seed in (13031957, 27182818, 31415926, 16180339):
        result = differential_evolution(
            global_objective,
            bounds,
            seed=seed,
            strategy="currenttobest1bin",
            popsize=22,
            maxiter=700,
            tol=2e-8,
            atol=1e-10,
            mutation=(0.45, 1.15),
            recombination=0.82,
            polish=False,
            updating="immediate",
            workers=1,
        )
        state = np.asarray(result.x, dtype=float)
        if all(inside_hull(state[2 * ring], state[2 * ring + 1])
               for ring in range(3)):
            candidate = minimum_area(state)
            if candidate > best_minimum:
                best_state = state.copy()
                best_minimum = candidate

    # Deterministic direct maximin polishing removes residual bottlenecks left
    # by the smoother annealing surrogate.
    if best_state is not None:
        state = best_state.copy()
        current = minimum_area(state)
        radius_step, phase_step = 0.018, 0.045
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
                        state, current, improved = trial, candidate, True

            if improved:
                best_state, best_minimum = state.copy(), current
            else:
                radius_step *= 0.5
                phase_step *= 0.5

    # Defensive fallback is never normally needed, but ensures valid output
    # even if an unexpected numerical optimization failure occurs.
    if best_state is None:
        best_state = np.array([0.15, 0.10, 0.30, 0.45, 0.46, 0.80])

    return make_points(best_state)


# EVOLVE-BLOCK-END
