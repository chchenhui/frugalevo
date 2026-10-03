# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically search for a 13-point Heilbronn configuration.

    The returned points lie in the unit square.  Since the objective is divided
    by convex-hull area, restricting the search to this fixed convex container
    does not impose a meaningful scale restriction.
    """
    n = 13
    rng = np.random.default_rng(13051957)
    triples = np.asarray(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    def hull_area(p: np.ndarray) -> float:
        """Twice-area monotone-chain hull, with a harmless zero-area guard."""
        ordered = p[np.lexsort((p[:, 1], p[:, 0]))]
        lower = []
        for q in ordered:
            while len(lower) >= 2:
                a, b = lower[-2], lower[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 1e-12:
                    break
                lower.pop()
            lower.append(q)
        upper = []
        for q in ordered[::-1]:
            while len(upper) >= 2:
                a, b = upper[-2], upper[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) > 1e-12:
                    break
                upper.pop()
            upper.append(q)
        polygon = np.asarray(lower[:-1] + upper[:-1])
        if len(polygon) < 3:
            return 0.0
        return abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
                   np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) * 0.5

    def evaluate(p: np.ndarray) -> tuple[float, float]:
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        areas = 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        area_hull = hull_area(p)
        if area_hull <= 1e-10:
            return 0.0, 0.0
        normalized = areas / area_hull
        # The lower-tail term makes moves that improve several critical
        # triangles acceptable before the strict minimum itself changes.
        tail = np.partition(normalized, 11)[:12]
        return float(normalized.min()), float(normalized.min() + 0.12 * tail.mean())

    def critical_vertex_weights(p: np.ndarray) -> np.ndarray:
        """Sampling weights concentrated on vertices of active small triangles."""
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        double_areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        active = triples[np.argpartition(double_areas, 17)[:18]].ravel()
        weights = np.bincount(active, minlength=n).astype(float)
        # Preserve occasional exploration of non-critical vertices.
        weights += 0.4
        return weights / weights.sum()

    best_points = None
    best_score = -np.inf

    # Use several independent basin-discovery runs, then spend the remaining
    # budget reheating the strongest configuration found.  For this highly
    # nonsmooth objective, a good active-triangle sign pattern is much more
    # valuable than another completely unrelated late random restart.
    for restart in range(6):
        exploratory = restart < 4
        if exploratory:
            current = rng.random((n, 2))
            iterations = 12000
            initial_sigma = 0.115
            initial_temperature = 0.0035
        else:
            # A deterministic small perturbation makes the two continuations
            # explore distinct neighboring sign-pattern changes.
            current = best_points + rng.normal(0.0, 0.010, size=(n, 2))
            np.clip(current, 0.0, 1.0, out=current)
            iterations = 18000
            initial_sigma = 0.058
            initial_temperature = 0.0019

        current_score, current_energy = evaluate(current)

        for step in range(iterations):
            progress = step / (iterations - 1.0)
            sigma = initial_sigma * (1.0 - progress) ** 1.7 + 0.0025
            candidate = current.copy()
            changed = 1 if rng.random() < 0.82 else 2
            # Most successful improvements repair one of the triangles that
            # currently realizes (or nearly realizes) the minimum area.
            if rng.random() < 0.78:
                indices = rng.choice(
                    n, size=changed, replace=False,
                    p=critical_vertex_weights(current),
                )
            else:
                indices = rng.choice(n, size=changed, replace=False)
            candidate[indices] += rng.normal(0.0, sigma, size=(changed, 2))
            np.clip(candidate, 0.0, 1.0, out=candidate)

            candidate_score, candidate_energy = evaluate(candidate)
            temperature = initial_temperature * (1.0 - progress) ** 2 + 0.000015
            if (candidate_energy >= current_energy or
                    rng.random() < np.exp((candidate_energy - current_energy) / temperature)):
                current = candidate
                current_score, current_energy = candidate_score, candidate_energy

            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()

    # Deterministic pattern search is considerably more reliable than random
    # last-mile moves on the piecewise-linear minimum-area landscape.
    current = best_points.copy()
    current_score, current_energy = evaluate(current)
    directions = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=float,
    )
    directions[4:] /= np.sqrt(2.0)

    for level in range(42):
        sigma = 0.018 * (0.87 ** level) + 0.00008
        improved = False
        # Prioritize the vertices that participate in active constraints.
        order = np.argsort(-critical_vertex_weights(current))
        for index in order:
            local_best = None
            local_score, local_energy = current_score, current_energy
            for direction in directions:
                candidate = current.copy()
                candidate[index] += sigma * direction
                np.clip(candidate, 0.0, 1.0, out=candidate)
                candidate_score, candidate_energy = evaluate(candidate)
                if (candidate_score > local_score + 1e-14 or
                        (abs(candidate_score - local_score) <= 1e-14 and
                         candidate_energy > local_energy + 1e-14)):
                    local_best = candidate
                    local_score, local_energy = candidate_score, candidate_energy
            if local_best is not None:
                current = local_best
                current_score, current_energy = local_score, local_energy
                improved = True
        # At very fine scales, one complete non-improving poll is sufficient.
        if not improved and sigma < 0.0003:
            break

    # Finish with a strict primary-objective active-set poll.  In particular,
    # normals to opposite edges directly alter the altitude of bottleneck
    # triangles, while centroid radial/tangential directions cover coordinated
    # hull and angular adjustments not represented by the Cartesian stencil.
    probe = 0.008
    for _ in range(16):
        a = current[triples[:, 0]]
        b = current[triples[:, 1]]
        c = current[triples[:, 2]]
        double_areas = np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )
        active = triples[np.argpartition(double_areas, 15)[:16]]
        center = current.mean(axis=0)
        improved = False

        for index in np.unique(active):
            vectors = [
                np.array([1.0, 0.0]), np.array([0.0, 1.0]),
                np.array([0.7071067811865476, 0.7071067811865476]),
                np.array([0.7071067811865476, -0.7071067811865476]),
            ]
            offset = current[index] - center
            length = float(np.hypot(offset[0], offset[1]))
            if length > 1.0e-12:
                radial = offset / length
                vectors.extend((radial, np.array([-radial[1], radial[0]])))

            # For every active triangle containing this vertex, append a unit
            # normal to the opposite edge.  Both signs are tested below.
            for triangle in active:
                if index in triangle:
                    other = triangle[triangle != index]
                    edge = current[other[1]] - current[other[0]]
                    edge_length = float(np.hypot(edge[0], edge[1]))
                    if edge_length > 1.0e-12:
                        vectors.append(
                            np.array([-edge[1], edge[0]]) / edge_length
                        )

            local_best = None
            local_score = current_score
            for vector in vectors:
                for sign in (-1.0, 1.0):
                    candidate = current.copy()
                    candidate[index] += sign * probe * vector
                    np.clip(candidate, 0.0, 1.0, out=candidate)
                    candidate_score, _ = evaluate(candidate)
                    if candidate_score > local_score + 1.0e-14:
                        local_best = candidate
                        local_score = candidate_score

            if local_best is not None:
                current = local_best
                current_score, current_energy = evaluate(current)
                improved = True

        if not improved:
            probe *= 0.52
        if probe < 2.0e-5:
            break

    return current


# EVOLVE-BLOCK-END