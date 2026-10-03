# EVOLVE-BLOCK-START
import numpy as np


_TRIANGLES_13 = np.asarray(
    [(i, j, k)
     for i in range(13)
     for j in range(i + 1, 13)
     for k in range(j + 1, 13)],
    dtype=np.intp,
)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize thirteen points in the reference simplex
    {(x,y): x >= 0, y >= 0, x+y <= 1}.

    The first three points fix the hull to (0,0), (1,0), (0,1).  Since the
    hull has area 1/2, absolute determinants are exactly normalized triangle
    areas.  Search proposals are driven by gradients of the currently tight
    determinant constraints rather than by uninformed coordinate noise.
    """
    rng = np.random.default_rng(13051957)
    n = 13
    fixed = 3

    def project_simplex(values: np.ndarray) -> None:
        """Fast feasible projection adequate for batched local proposals."""
        np.maximum(values, 0.0, out=values)
        sums = values[..., 0] + values[..., 1]
        scale = np.maximum(1.0, sums)
        values /= scale[..., None]

    def determinants(configs: np.ndarray) -> np.ndarray:
        q = configs[:, _TRIANGLES_13, :]
        return np.abs(
            (q[:, :, 1, 0] - q[:, :, 0, 0])
            * (q[:, :, 2, 1] - q[:, :, 0, 1])
            - (q[:, :, 1, 1] - q[:, :, 0, 1])
            * (q[:, :, 2, 0] - q[:, :, 0, 0])
        )

    def score(values: np.ndarray, softness: float) -> np.ndarray:
        """Stable soft minimum, with a small lower-tail balancing term."""
        lo = values.min(axis=1)
        z = np.clip(-(values - lo[:, None]) / softness, -70.0, 0.0)
        soft = lo - softness * np.log(np.exp(z).sum(axis=1))
        tail = np.partition(values, 15, axis=1)[:, :16].mean(axis=1)
        return soft + 0.045 * tail

    def active_gradients(points: np.ndarray, areas: np.ndarray,
                         softness: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Form a subgradient of a weighted soft lower envelope.  Each triangle
        gives an exact signed determinant gradient to each of its vertices.
        """
        minimum = float(areas.min())
        active_ids = np.argpartition(areas, 27)[:28]
        triples = _TRIANGLES_13[active_ids]

        raw = (
            (points[triples[:, 1], 0] - points[triples[:, 0], 0])
            * (points[triples[:, 2], 1] - points[triples[:, 0], 1])
            - (points[triples[:, 1], 1] - points[triples[:, 0], 1])
            * (points[triples[:, 2], 0] - points[triples[:, 0], 0])
        )
        weights = np.exp(
            np.clip(-(areas[active_ids] - minimum) / max(softness, 2.e-5),
                    -24.0, 0.0)
        )
        weights *= np.where(raw >= 0.0, 1.0, -1.0)

        grad = np.zeros((n, 2), dtype=float)
        pool = []

        for tri, weight in zip(triples, weights):
            a, b, c = tri
            pa, pb, pc = points[a], points[b], points[c]

            # Gradients of cross(pb-pa, pc-pa).
            ga = np.array((pb[1] - pc[1], pc[0] - pb[0])) * weight
            gb = np.array((pc[1] - pa[1], pa[0] - pc[0])) * weight
            gc = np.array((pa[1] - pb[1], pb[0] - pa[0])) * weight
            grad[a] += ga
            grad[b] += gb
            grad[c] += gc

            for index in tri:
                if index >= fixed:
                    pool.append(index)

        if not pool:
            pool = list(range(fixed, n))
        return grad, np.asarray(pool, dtype=np.intp)

    best = None
    best_value = -np.inf
    best_tail = -np.inf

    # Two different deterministic initial incidence patterns offer inexpensive
    # global diversification while leaving most work for active-set refinement.
    for restart in range(2):
        points = np.empty((n, 2), dtype=float)
        points[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

        if restart == 0:
            # A perturbed barycentric lattice has useful initial separation.
            bary = np.array([
                [0.16, 0.16], [0.42, 0.12], [0.70, 0.10],
                [0.12, 0.42], [0.37, 0.34], [0.62, 0.25],
                [0.10, 0.70], [0.27, 0.58], [0.48, 0.43],
                [0.30, 0.18],
            ])
            points[3:] = bary + rng.normal(0.0, 0.035, size=(10, 2))
        else:
            values = rng.random((10, 2))
            outside = values.sum(axis=1) > 1.0
            values[outside] = 1.0 - values[outside]
            points[3:] = values

        project_simplex(points[3:])
        current = points.copy()
        current_areas = determinants(current[None, :, :])[0]

        local_best = current.copy()
        local_value = float(current_areas.min())
        local_tail = float(np.partition(current_areas, 15)[:16].mean())

        batch = 28
        iterations = 4400

        for iteration in range(iterations):
            progress = iteration / (iterations - 1.0)
            step = 0.105 * (0.0011 / 0.105) ** progress
            softness = 0.010 * (0.000035 / 0.010) ** progress
            temperature = 0.0025 * (0.000015 / 0.0025) ** progress

            gradient, pool = active_gradients(current, current_areas, softness)
            chosen_points = pool[rng.integers(0, len(pool), size=batch)]

            candidates = np.repeat(current[None, :, :], batch, axis=0)
            noise = rng.normal(size=(batch, 2))
            directions = gradient[chosen_points] + (
                0.42 + 0.75 * (1.0 - progress)
            ) * noise

            lengths = np.sqrt(np.sum(directions * directions, axis=1))
            directions /= np.maximum(lengths[:, None], 1.e-14)
            candidates[np.arange(batch), chosen_points] += step * directions

            # Deliberate opposing/noise probes preserve escape ability when
            # active constraints have a poor common local subgradient.
            exploratory = np.arange(batch) % 7 == 0
            candidates[
                np.arange(batch)[exploratory], chosen_points[exploratory]
            ] = (
                current[chosen_points[exploratory]]
                + step * rng.normal(size=(exploratory.sum(), 2))
            )

            # Early coordinated moves change incidences that cannot be repaired
            # by a purely one-point local displacement.
            if iteration < 1700:
                rows = np.arange(0, batch, 7)
                second = rng.integers(fixed, n, size=len(rows))
                candidates[rows, second] += rng.normal(
                    scale=0.38 * step, size=(len(rows), 2)
                )

            project_simplex(candidates[:, fixed:, :])
            candidate_areas = determinants(candidates)
            candidate_scores = score(candidate_areas, softness)
            selected = int(np.argmax(candidate_scores))

            current_score = float(score(current_areas[None, :], softness)[0])
            delta = float(candidate_scores[selected] - current_score)
            if delta >= 0.0 or rng.random() < np.exp(
                max(-55.0, delta / max(temperature, 1.e-12))
            ):
                current = candidates[selected]
                current_areas = candidate_areas[selected]

            for config, areas in (
                (candidates[selected], candidate_areas[selected]),
                (current, current_areas),
            ):
                value = float(areas.min())
                tail = float(np.partition(areas, 15)[:16].mean())
                if (value > local_value + 1.e-14 or
                        (abs(value - local_value) <= 1.e-14 and
                         tail > local_tail)):
                    local_best = config.copy()
                    local_value = value
                    local_tail = tail

        if (local_value > best_value + 1.e-14 or
                (abs(local_value - best_value) <= 1.e-14 and
                 local_tail > best_tail)):
            best = local_best
            best_value = local_value
            best_tail = local_tail

    # Strict final polishing: only accept genuine improvements of the actual
    # objective, using both active gradients and an isotropic directional stencil.
    current = best.copy()
    current_areas = determinants(current[None, :, :])[0]
    current_value = float(current_areas.min())

    stencil = np.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [0.7071067811865476, 0.7071067811865476],
        [0.7071067811865476, -0.7071067811865476],
        [-0.7071067811865476, 0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476],
    ])

    probe = 0.009
    for _ in range(18):
        improved = False
        grad, pool = active_gradients(current, current_areas, 0.00012)
        indices = np.unique(pool)

        for index in indices:
            g = grad[index]
            norm = float(np.sqrt(np.dot(g, g)))
            directions = stencil
            if norm > 1.e-14:
                directions = np.vstack((g / norm, -g / norm, stencil))

            for direction in directions:
                trial = current.copy()
                trial[index] += probe * direction
                project_simplex(trial[index:index + 1])
                trial_areas = determinants(trial[None, :, :])[0]
                trial_value = float(trial_areas.min())

                if trial_value > current_value + 1.e-14:
                    current = trial
                    current_areas = trial_areas
                    current_value = trial_value
                    improved = True
                    break

        if not improved:
            probe *= 0.52

    if current_value > best_value:
        best = current

    return best


# EVOLVE-BLOCK-END