# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct 13 points in a unit-area convex triangle.

    The reference domain is x >= 0, y >= 0, x + y <= 1.  Its area is 1/2;
    all returned coordinates are scaled by sqrt(2), making the final
    containing right triangle have area exactly one.

    The local optimizer is a coordinate feasibility pump.  With every point
    except one fixed, each signed triangle determinant containing the moving
    point is affine.  Thus testing whether a target minimum determinant is
    attainable is just intersection of finitely many half-planes.
    """
    n = 13
    rng = np.random.default_rng(20250317)

    triples = np.array(
        [(i, j, k)
         for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    incident = [[] for _ in range(n)]
    for t, (i, j, k) in enumerate(triples):
        incident[i].append(t)
        incident[j].append(t)
        incident[k].append(t)
    incident = [np.asarray(x, dtype=np.intp) for x in incident]

    incident_mask = np.zeros((n, len(triples)), dtype=bool)
    for q in range(n):
        incident_mask[q, incident[q]] = True

    def determinants(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    def clip_halfplane(poly: np.ndarray, ax: float, ay: float,
                       rhs: float) -> np.ndarray:
        """Clip convex polygon by ax*x + ay*y >= rhs."""
        if len(poly) == 0:
            return poly

        values = poly[:, 0] * ax + poly[:, 1] * ay - rhs
        output = []
        previous = poly[-1]
        previous_value = values[-1]
        previous_inside = previous_value >= -2.0e-13

        for current, current_value in zip(poly, values):
            current_inside = current_value >= -2.0e-13
            if current_inside != previous_inside:
                denom = previous_value - current_value
                if abs(denom) > 1.0e-16:
                    ratio = previous_value / denom
                    output.append(previous + ratio * (current - previous))
            if current_inside:
                output.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside

        if not output:
            return np.empty((0, 2), dtype=float)
        return np.asarray(output, dtype=float)

    def feasible_polygon(p: np.ndarray, q: int, target: float,
                         signs: np.ndarray) -> np.ndarray:
        """
        Intersect the reference triangle with all signed determinant
        constraints involving point q.
        """
        poly = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

        for t in incident[q]:
            i, j, k = triples[t]
            a, b, c = p[i], p[j], p[k]

            # det(a,b,c) = d0 + dx*x_q + dy*y_q.
            if q == i:
                d0 = b[0] * c[1] - b[1] * c[0]
                dx = c[1] - b[1]
                dy = b[0] - c[0]
            elif q == j:
                d0 = a[1] * c[0] - a[0] * c[1]
                dx = c[1] - a[1]
                dy = a[0] - c[0]
            else:
                d0 = a[0] * b[1] - a[1] * b[0]
                dx = a[1] - b[1]
                dy = b[0] - a[0]

            s = signs[t]
            poly = clip_halfplane(poly, s * dx, s * dy, target - s * d0)
            if len(poly) == 0:
                break

        return poly

    def improve_coordinate(p: np.ndarray, q: int,
                           dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Maximize the attainable global determinant floor for a single point.
        Signs are fixed during this update; crossing a sign would require a
        zero-area triangle and therefore cannot improve a positive solution.
        """
        signs = np.where(dets >= 0.0, 1.0, -1.0)
        other = np.abs(dets[~incident_mask[q]])
        lower = float(np.min(np.abs(dets)))
        upper = float(np.min(other)) if len(other) else 0.5

        # Binary search is intentionally modest: geometric clipping, rather
        # than a noisy stochastic proposal, supplies the refinement accuracy.
        best_poly = feasible_polygon(p, q, lower * (1.0 - 2.0e-12), signs)
        for _ in range(16):
            middle = 0.5 * (lower + upper)
            poly = feasible_polygon(p, q, middle, signs)
            if len(poly):
                lower = middle
                best_poly = poly
            else:
                upper = middle

        if len(best_poly) == 0:
            return p, dets

        # The centroid is interior whenever the feasible cell has area.  This
        # preserves slack for subsequent coordinates better than an arbitrary
        # extreme intersection point.
        candidate = p.copy()
        candidate[q] = np.mean(best_poly, axis=0)
        new_dets = determinants(candidate)

        if np.min(np.abs(new_dets)) + 1.0e-12 >= np.min(np.abs(dets)):
            return candidate, new_dets
        return p, dets

    best_points = None
    best_value = -np.inf

    # Independent simplex samples choose different determinant-sign chambers.
    # The later sweeps are deterministic given this fixed generator seed.
    for restart in range(36):
        points = np.empty((n, 2), dtype=float)
        points[:3] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        uv = rng.random((10, 2))
        fold = np.sum(uv, axis=1) > 1.0
        uv[fold] = 1.0 - uv[fold]

        # A small restart-dependent cyclic displacement avoids repeatedly
        # selecting identical low-slack orientation patterns.
        shift = (restart * 0.6180339887498949) % 1.0
        uv[:, 0] = (uv[:, 0] + 0.17 * shift) % 1.0
        fold = np.sum(uv, axis=1) > 1.0
        uv[fold] = 1.0 - uv[fold]
        points[3:] = uv

        dets = determinants(points)

        for sweep in range(22):
            # Reordering coordinates helps avoid a directional coordinate-
            # descent bias while retaining completely reproducible behavior.
            order = rng.permutation(np.arange(3, n))
            for q in order:
                points, dets = improve_coordinate(points, int(q), dets)

            value = float(np.min(np.abs(dets)))
            if value > best_value:
                best_value = value
                best_points = points.copy()

            # A very small deterministic kick between early sweeps permits
            # movement between nearby active-constraint faces.  It is accepted
            # only if it retains the current floor.
            if sweep in (5, 11) and restart % 3 == 0:
                trial = points.copy()
                trial[3:] += rng.normal(0.0, 0.006 / (sweep + 1), (10, 2))
                trial[3:] = np.maximum(trial[3:], 0.0)
                sums = np.sum(trial[3:], axis=1)
                over = sums > 0.999999
                trial[3:][over] /= sums[over, None] / 0.999999
                trial_dets = determinants(trial)
                if np.min(np.abs(trial_dets)) >= 0.995 * np.min(np.abs(dets)):
                    points, dets = trial, trial_dets

    if best_points is None or not np.all(np.isfinite(best_points)):
        raise RuntimeError("Heilbronn feasibility pump failed")

    # The reference hull area is 1/2.  Scaling by sqrt(2) gives a unit-area
    # convex containing region without changing normalized triangle areas.
    return best_points * np.sqrt(2.0)


# EVOLVE-BLOCK-END