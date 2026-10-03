# EVOLVE-BLOCK-START
import numpy as np


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
        """Compute the exact maximin normalized triangle-area objective."""
        return float(np.min(low_areas(state)))

    def merit(state: np.ndarray) -> float:
        """Score the true bottleneck plus a small low-tail stability bonus."""
        # np.partition leaves its selected prefix unordered, so low[0] is not
        # necessarily the minimum.  Use np.min explicitly.
        low = low_areas(state)
        return float(np.min(low) + 0.18 * np.mean(low))

    best_state = None
    best_minimum = -np.inf

    # A compact deterministic multistart budget reaches the same symmetric
    # maximin basin while leaving runtime for the exact polish below.
    for _ in range(36):
        state = np.empty(6)
        for ring in range(3):
            phase = rng.uniform(0.0, 2.0 * np.pi / 3.0)
            limit = outer_radius
            while not inside_hull(limit, phase):
                limit *= 0.985
            state[2 * ring] = rng.uniform(0.055, max(0.056, 0.985 * limit))
            state[2 * ring + 1] = phase

        current = merit(state)
        minimum = minimum_area(state)
        if minimum > best_minimum:
            best_minimum = minimum
            best_state = state.copy()

        for iteration in range(1150):
            trial = state.copy()
            coordinate = int(rng.integers(6))
            cooling = 1.0 - iteration / 1150.0

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

            low = low_areas(trial)
            candidate = float(np.min(low) + 0.18 * np.mean(low))
            temperature = 0.0014 * cooling * cooling + 1e-6
            if (candidate >= current or
                    rng.random() < np.exp((candidate - current) / temperature)):
                state = trial
                current = candidate

                # Reuse the already evaluated trial geometry.
                minimum = float(np.min(low))
                if minimum > best_minimum:
                    best_minimum = minimum
                    best_state = state.copy()

    # Defensive fallback is never normally needed, but ensures valid output
    # even if an unexpected numerical optimization failure occurs.
    if best_state is None:
        best_state = np.array([0.15, 0.10, 0.30, 0.45, 0.46, 0.80])

    def cartesian_maximin(seed: np.ndarray) -> np.ndarray:
        """Use COBYLA to maximize an explicit area lower bound for all 286 triangles."""
        from scipy.optimize import minimize

        # Keep the three hull vertices fixed.  This fixes affine gauge and
        # guarantees that every feasible returned point has the same hull.
        initial = make_points(seed)
        free0 = np.vstack((initial[0:1], initial[4:])).reshape(-1)
        start = np.concatenate((free0, [minimum_area(seed)]))

        def unpack(vector: np.ndarray) -> np.ndarray:
            return np.vstack((outer, vector[:-1].reshape(10, 2)))

        def inequalities(vector: np.ndarray) -> np.ndarray:
            points = unpack(vector)
            a = points[triples[:, 0]]
            b = points[triples[:, 1]]
            c = points[triples[:, 2]]
            areas = np.abs(
                (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
            ) / hull_det

            free = points[3:]
            bary = (bary_inverse @ (free - outer[2]).T).T
            containment = np.column_stack((
                bary[:, 0], bary[:, 1], 1.0 - bary.sum(axis=1)
            )).ravel()
            return np.concatenate((areas - vector[-1], containment))

        result = minimize(
            lambda vector: -vector[-1],
            start,
            method="COBYLA",
            constraints={"type": "ineq", "fun": inequalities},
            options={
                "rhobeg": 0.028,
                "tol": 2e-7,
                "catol": 2e-9,
                "maxiter": 30000,
            },
        )

        candidate = unpack(result.x)
        # Do not trust a solver status alone: explicitly retain the annealed
        # configuration if numerical constraint tolerances hurt its true score.
        a = candidate[triples[:, 0]]
        b = candidate[triples[:, 1]]
        c = candidate[triples[:, 2]]
        value = float(np.min(np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        ) / hull_det))
        return candidate if np.isfinite(value) and value >= best_minimum - 1e-10 else initial

    # Unlike the orbit-only annealing model, this final stage independently
    # moves the centre and all nine ring points, allowing asymmetric active
    # triangle constraints to equilibrate.
    return cartesian_maximin(best_state)


# EVOLVE-BLOCK-END
