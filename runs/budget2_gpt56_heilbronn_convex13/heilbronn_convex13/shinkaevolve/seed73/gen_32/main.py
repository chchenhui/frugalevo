# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct a deterministic maximin configuration of 13 points.

    The three first points are the vertices of a reference simplex.  All
    remaining points are projected into that simplex, so its convex hull is
    fixed and every signed determinant is already an area normalized by the
    hull area.
    """
    n = 13
    rng = np.random.default_rng(seed=13051957)

    triangles = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )

    # A triangle of area 1/2; determinant magnitudes equal area/hull_area.
    points = np.empty((n, 2), dtype=float)
    points[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    interior = rng.random((n - 3, 2))
    reflected = interior.sum(axis=1) > 1.0
    interior[reflected] = 1.0 - interior[reflected]
    points[3:] = interior

    def determinants(config):
        p = config[:, triangles]
        return np.abs(
            (p[:, :, 1, 0] - p[:, :, 0, 0])
            * (p[:, :, 2, 1] - p[:, :, 0, 1])
            - (p[:, :, 1, 1] - p[:, :, 0, 1])
            * (p[:, :, 2, 0] - p[:, :, 0, 0])
        )

    def soft_min(values, temperature):
        minimum = values.min(axis=1)
        return minimum - temperature * np.log(
            np.exp(-(values - minimum[:, None]) / temperature).sum(axis=1)
        )

    current = points.copy()
    current_areas = determinants(current[None, :, :])[0]
    best = current.copy()
    best_value = float(current_areas.min())

    # Batched mutations make the search substantially less sensitive to the
    # order in which individual point coordinates are updated.
    batch = 24
    for iteration in range(9000):
        progress = iteration / 8999.0
        temperature = 0.012 * (0.00030 / 0.012) ** progress
        step = 0.105 * (0.0010 / 0.105) ** progress

        candidates = np.repeat(current[None, :, :], batch, axis=0)

        # Most moves are drawn from vertices involved in the lower determinant
        # tail.  These are the constraints that determine the maximin value;
        # retaining some uniform moves still permits changes of incidence
        # pattern when the currently active set is misleading.
        tail_rank = min(17, len(current_areas) - 1)
        active_rows = np.argpartition(current_areas, tail_rank)[:tail_rank + 1]
        active_vertices = np.unique(triangles[active_rows].ravel())
        active_vertices = active_vertices[active_vertices >= 3]
        changed = rng.integers(3, n, size=batch)
        focused = rng.random(batch) < 0.82
        if len(active_vertices):
            changed[focused] = rng.choice(
                active_vertices, size=np.count_nonzero(focused), replace=True
            )
        candidates[np.arange(batch), changed] += rng.normal(
            scale=step, size=(batch, 2)
        )

        # Occasionally perturb two additional points, allowing coordinated
        # changes during the coarse phase without sacrificing fine refinement.
        if iteration < 3500 and iteration % 11 == 0:
            extra = rng.integers(3, n, size=(batch, 2))
            candidates[
                np.arange(batch)[:, None], extra
            ] += rng.normal(scale=0.45 * step, size=(batch, 2, 2))

        xy = candidates[:, 3:, :]
        np.maximum(xy, 0.001, out=xy)
        sums = xy.sum(axis=2)
        scale = np.minimum(1.0, 0.999 / sums)
        xy *= scale[:, :, None]

        candidate_areas = determinants(candidates)
        scores = soft_min(candidate_areas, temperature)
        current_score = soft_min(current_areas[None, :], temperature)[0]
        chosen = int(np.argmax(scores))

        if scores[chosen] > current_score:
            current = candidates[chosen]
            current_areas = candidate_areas[chosen]

        candidate_best = float(candidate_areas[chosen].min())
        if candidate_best > best_value:
            best_value = candidate_best
            best = candidates[chosen].copy()

        current_best = float(current_areas.min())
        if current_best > best_value:
            best_value = current_best
            best = current.copy()

    # A final deterministic direct search uses the actual nonsmooth maximin
    # objective rather than the annealing surrogate.  Restricting probes to
    # vertices in tight triangles makes this a compact active-set polish.
    directions = np.array([
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, -0.7071067811865476),
    ])
    probe = 0.014
    for _ in range(18):
        best_areas = determinants(best[None, :, :])[0]
        active_rows = np.argpartition(best_areas, 23)[:24]
        movable = np.unique(triangles[active_rows].ravel())
        movable = movable[movable >= 3]

        trials = np.repeat(best[None, :, :], len(movable) * len(directions), axis=0)
        trial_indices = np.repeat(movable, len(directions))
        trial_directions = np.tile(directions, (len(movable), 1))
        trials[np.arange(len(trials)), trial_indices] += probe * trial_directions

        xy = trials[:, 3:, :]
        np.maximum(xy, 0.001, out=xy)
        sums = xy.sum(axis=2)
        xy *= np.minimum(1.0, 0.999 / sums)[:, :, None]

        trial_areas = determinants(trials)
        trial_values = trial_areas.min(axis=1)
        selected = int(np.argmax(trial_values))
        selected_value = float(trial_values[selected])

        if selected_value > best_value + 1.0e-13:
            best = trials[selected].copy()
            best_value = selected_value
        else:
            probe *= 0.55
            if probe < 2.0e-5:
                break

    return best


# EVOLVE-BLOCK-END