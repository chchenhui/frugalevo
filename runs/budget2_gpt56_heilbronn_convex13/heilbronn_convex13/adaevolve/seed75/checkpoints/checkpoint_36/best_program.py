# EVOLVE-BLOCK-START
import numpy as np

_HEILBRONN13_CACHE = None


def heilbronn_convex13() -> np.ndarray:
    """Optimize 13 points in an equilateral triangular hull using a deterministic
    3-fold symmetric global search followed by constrained maximin refinement."""
    global _HEILBRONN13_CACHE

    if _HEILBRONN13_CACHE is not None:
        return _HEILBRONN13_CACHE.copy()

    root3 = np.sqrt(3.0)

    # An equilateral hull is convenient because rotations by 120 degrees
    # preserve feasibility.  Normalization by hull determinant is equivalent
    # to normalization by hull area, since both triangle areas include 1/2.
    vertices = np.array([
        [0.0, 1.0],
        [-root3 / 2.0, -0.5],
        [root3 / 2.0, -0.5],
    ], dtype=float)
    hull_det = abs(
        (vertices[1, 0] - vertices[0, 0])
        * (vertices[2, 1] - vertices[0, 1])
        - (vertices[1, 1] - vertices[0, 1])
        * (vertices[2, 0] - vertices[0, 0])
    )

    triples = np.array([
        (i, j, k)
        for i in range(13)
        for j in range(i + 1, 13)
        for k in range(j + 1, 13)
    ], dtype=int)

    rotation = np.array([
        [-0.5, -root3 / 2.0],
        [root3 / 2.0, -0.5],
    ], dtype=float)

    def normalized_areas(points):
        a = points[triples[:, 0]]
        b = points[triples[:, 1]]
        c = points[triples[:, 2]]
        determinants = (
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        return np.abs(determinants) / hull_det

    def make_symmetric_points(parameters):
        # Each pair maps to barycentric coordinates, guaranteeing that the
        # representative point and all of its rotations lie in the hull.
        points = [vertices[0], vertices[1], vertices[2], [0.0, 0.0]]
        for alpha, beta in np.asarray(parameters).reshape(3, 2):
            point = (
                alpha * vertices[0]
                + (1.0 - alpha)
                * (beta * vertices[1] + (1.0 - beta) * vertices[2])
            )
            points.extend((point, rotation @ point, rotation @ (rotation @ point)))
        return np.asarray(points, dtype=float)

    def objective(parameters):
        return -float(np.min(normalized_areas(make_symmetric_points(parameters))))

    # This fallback remains feasible and deterministic if scipy is unavailable.
    fallback_parameters = np.array([
        0.18, 0.71,
        0.48, 0.23,
        0.77, 0.58,
    ])
    best = make_symmetric_points(fallback_parameters)
    best_value = float(np.min(normalized_areas(best)))

    try:
        from scipy.optimize import differential_evolution, minimize

        # Searching six symmetric parameters is substantially more robust than
        # directly searching all 26 coordinates of an unrestricted placement.
        # The active minimum-area triple changes discontinuously throughout
        # parameter space.  Several fixed-seed global restarts are therefore
        # more reliable than relying on one evolutionary population becoming
        # trapped in a single oriented-matroid basin.
        # Boundary points are legitimate and frequently useful in maximin
        # configurations, so do not artificially exclude alpha or beta values
        # near zero or one.  Fixed independent seeds retain reproducibility.
        for seed, strategy in (
            (173, "best1bin"),
            (947, "currenttobest1bin"),
            (2027, "randtobest1bin"),
            (4099, "best2bin"),
            (8111, "rand1bin"),
        ):
            global_result = differential_evolution(
                objective,
                bounds=[(0.0, 1.0)] * 6,
                seed=seed,
                strategy=strategy,
                popsize=30,
                maxiter=1100,
                tol=3e-10,
                atol=1e-11,
                polish=False,
                updating="immediate",
                workers=1,
            )
            trial = make_symmetric_points(global_result.x)
            trial_value = float(np.min(normalized_areas(trial)))
            if np.isfinite(trial_value) and trial_value > best_value:
                best = trial
                best_value = trial_value

        # Direct nonsmooth epigraph refinement.  The three outer vertices are
        # an affine gauge and remain the actual triangular hull; all other ten
        # locations are independently movable.  The signs define the oriented
        # matroid cell of the good deterministic incumbent.  COBYLA then works
        # directly with the many active area inequalities rather than relying
        # on derivatives of abs(det).
        a = best[triples[:, 0]]
        b = best[triples[:, 1]]
        c = best[triples[:, 2]]
        orientations = np.sign(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        orientations[orientations == 0.0] = 1.0

        barycentric_matrix = np.column_stack((
            vertices[0] - vertices[2],
            vertices[1] - vertices[2],
        ))
        barycentric_inverse = np.linalg.inv(barycentric_matrix)

        def unpack(variables):
            """Reconstruct the fixed-hull, 13-point COBYLA configuration."""
            return np.vstack((vertices, variables[:20].reshape(10, 2)))

        def epigraph_constraints(variables):
            points = unpack(variables)
            pa = points[triples[:, 0]]
            pb = points[triples[:, 1]]
            pc = points[triples[:, 2]]
            determinants = (
                (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1])
                - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
            )
            # Since triangle area / hull area is abs(det) / hull_det, t is
            # exactly the evaluator's normalized bottleneck objective.
            return orientations * determinants / hull_det - variables[20]

        def hull_constraints(variables):
            points = unpack(variables)[3:]
            uv = (barycentric_inverse @ (points - vertices[2]).T).T
            return np.concatenate((
                uv[:, 0],
                uv[:, 1],
                1.0 - uv[:, 0] - uv[:, 1],
            ))

        start = np.concatenate((best[3:].ravel(), [best_value]))
        local_result = minimize(
            lambda variables: -variables[20],
            start,
            method="COBYLA",
            constraints=[
                {"type": "ineq", "fun": epigraph_constraints},
                {"type": "ineq", "fun": hull_constraints},
                {"type": "ineq", "fun": lambda x: x[20]},
            ],
            options={
                "maxiter": 50000,
                "rhobeg": 0.035,
                "tol": 2e-10,
                "catol": 2e-10,
                "disp": False,
            },
        )

        # Do not trust the epigraph value blindly: verify actual absolute
        # triangle areas and hull feasibility before accepting a candidate.
        candidate = unpack(local_result.x)
        candidate_value = float(np.min(normalized_areas(candidate)))
        candidate_inside = float(np.min(hull_constraints(local_result.x)))
        candidate_epigraph = float(np.min(epigraph_constraints(local_result.x)))
        if (
            np.isfinite(candidate_value)
            and candidate_inside >= -1e-9
            and candidate_epigraph >= -1e-8
            and candidate_value > best_value + 1e-12
        ):
            best = candidate
            best_value = candidate_value

    except Exception:
        # Preserve the deterministic feasible symmetric construction on any
        # numerical optimization or optional-dependency failure.
        pass

    _HEILBRONN13_CACHE = np.asarray(best, dtype=float)
    return _HEILBRONN13_CACHE.copy()


# EVOLVE-BLOCK-END
