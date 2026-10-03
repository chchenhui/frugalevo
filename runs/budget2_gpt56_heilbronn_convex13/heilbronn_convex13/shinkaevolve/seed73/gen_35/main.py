# EVOLVE-BLOCK-START
import numpy as np


_TRIANGLES_13 = np.asarray(
    [(i, j, k)
     for i in range(11)
     for j in range(i + 1, 12)
     for k in range(j + 1, 13)],
    dtype=np.intp,
)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct thirteen points with a fixed regular polygonal
    hull.  The free points are kept inside the polygon's incircle, so the
    boundary ring remains the convex hull and normalized areas can be scored
    without repeatedly computing a convex hull.
    """
    rng = np.random.default_rng(13051957)
    n = 13
    best_points = None
    best_value = -np.inf
    best_tail = -np.inf

    def make_hull(count: int, phase: float) -> tuple[np.ndarray, float, float]:
        angles = phase + 2.0 * np.pi * np.arange(count) / count
        boundary = np.column_stack((np.cos(angles), np.sin(angles)))

        # Shoelace area of a unit-circumradius regular count-gon.
        hull_area = 0.5 * count * np.sin(2.0 * np.pi / count)
        # Disk contained in the polygon, hence free points cannot alter hull.
        inner_radius = np.cos(np.pi / count) * 0.997
        return boundary, hull_area, inner_radius

    def project_disk(values: np.ndarray, radius: float) -> None:
        lengths = np.sqrt(np.sum(values * values, axis=-1))
        scale = np.minimum(1.0, radius / np.maximum(lengths, 1.0e-15))
        values *= scale[..., None]

    def areas_batch(configs: np.ndarray) -> np.ndarray:
        q = configs[:, _TRIANGLES_13, :]
        return np.abs(
            (q[:, :, 1, 0] - q[:, :, 0, 0])
            * (q[:, :, 2, 1] - q[:, :, 0, 1])
            - (q[:, :, 1, 1] - q[:, :, 0, 1])
            * (q[:, :, 2, 0] - q[:, :, 0, 0])
        )

    def quality(det_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The strict minimum is primary.  The average lower tail gives useful
        # gradients of preference before the identity of the bottleneck flips.
        low = np.partition(det_values, 11, axis=1)[:, :12]
        strict = low.min(axis=1)
        return strict + 0.16 * low.mean(axis=1), strict

    # These layouts provide different boundary/interior combinatorics while
    # preserving a common, inexpensive fixed-hull normalization.
    # A nine-point near-circular hull with four interior points is a more
    # favorable combinatorial regime for n=13: it shortens boundary chords
    # while retaining enough interior degrees of freedom to balance the
    # active determinant constraints.  The phases are deliberately distinct,
    # since the square-like four-point interior ring has phase-sensitive
    # incidences with the nine boundary vertices.
    layouts = (
        (9, 0.0),
        (9, np.pi / 9.0),
        (9, 0.19),
    )

    for restart, (outer_count, phase) in enumerate(layouts):
        boundary, hull_area, radius = make_hull(outer_count, phase)
        free_count = n - outer_count

        points = np.empty((n, 2), dtype=float)
        points[:outer_count] = boundary

        # A perturbed inner ring is a far better initial condition than purely
        # uniform samples, but a deterministic random component permits each
        # restart to explore a distinct incidence pattern.
        ring_angles = (
            phase + 0.41
            + 2.0 * np.pi * np.arange(free_count) / free_count
        )
        ring_radius = 0.47 + 0.055 * rng.normal(size=free_count)
        points[outer_count:] = np.column_stack((
            ring_radius * np.cos(ring_angles),
            ring_radius * np.sin(ring_angles),
        ))
        points[outer_count:] += rng.normal(
            scale=0.035, size=(free_count, 2)
        )
        project_disk(points[outer_count:], radius)

        current = points.copy()
        current_det = areas_batch(current[None, :, :])[0]
        current_score, current_min = quality(current_det[None, :])
        current_score = float(current_score[0])
        current_min = float(current_min[0])

        local_best = current.copy()
        local_value = current_min
        local_tail = float(np.partition(current_det, 11)[:12].mean())

        batch = 24
        iterations = 4000

        for iteration in range(iterations):
            fraction = iteration / (iterations - 1.0)
            step = 0.125 * (0.012 / 0.125) ** fraction
            temperature = 0.010 * (0.00006 / 0.010) ** fraction

            candidates = np.repeat(current[None, :, :], batch, axis=0)

            # Draw mutation centers from triangles in the lower tail.  This
            # directs work to geometrically relevant points rather than using
            # uniform coordinate proposals.
            cutoff = np.partition(current_det, 15)[15]
            critical = np.flatnonzero(current_det <= cutoff)
            selected_triangles = _TRIANGLES_13[
                critical[rng.integers(0, len(critical), size=batch)]
            ]

            for row in range(batch):
                movable = selected_triangles[row][
                    selected_triangles[row] >= outer_count
                ]
                if len(movable):
                    index = int(movable[rng.integers(len(movable))])
                else:
                    index = int(rng.integers(outer_count, n))
                candidates[row, index] += rng.normal(scale=step, size=2)

                # Coherent two-point moves early in the search allow escape
                # from poor ring alignments.
                if iteration < 1800 and row % 5 == 0:
                    second = int(rng.integers(outer_count, n))
                    candidates[row, second] += rng.normal(
                        scale=0.42 * step, size=2
                    )

            project_disk(candidates[:, outer_count:, :], radius)
            det = areas_batch(candidates)
            scores, minima = quality(det)
            chosen = int(np.argmax(scores))

            candidate_score = float(scores[chosen])
            delta = candidate_score - current_score
            if (delta >= 0.0 or
                    rng.random() < np.exp(max(-60.0, delta / temperature))):
                current = candidates[chosen]
                current_det = det[chosen]
                current_score = candidate_score
                current_min = float(minima[chosen])

            candidate_min = float(minima[chosen])
            candidate_tail = float(np.partition(det[chosen], 11)[:12].mean())
            if (candidate_min > local_value + 1.0e-13 or
                    (abs(candidate_min - local_value) <= 1.0e-13 and
                     candidate_tail > local_tail)):
                local_best = candidates[chosen].copy()
                local_value = candidate_min
                local_tail = candidate_tail

            if current_min > local_value + 1.0e-13:
                local_best = current.copy()
                local_value = current_min
                local_tail = float(np.partition(current_det, 11)[:12].mean())

        normalized_value = local_value / hull_area
        normalized_tail = local_tail / hull_area
        if (normalized_value > best_value + 1.0e-13 or
                (abs(normalized_value - best_value) <= 1.0e-13 and
                 normalized_tail > best_tail)):
            best_points = local_best
            best_value = normalized_value
            best_tail = normalized_tail

    # Small deterministic active-set polishing on the best fixed-hull layout.
    # Infer its fixed outer ring from points of unit radius.
    radii = np.sqrt(np.sum(best_points * best_points, axis=1))
    outer_count = int(np.count_nonzero(radii > 0.999999))
    # The search currently includes a nine-vertex hull.  Retain every fixed
    # hull vertex during polishing and normalize with that same hull's area.
    outer_count = max(3, min(outer_count, 9))
    _, hull_area, radius = make_hull(outer_count, 0.0)

    current = best_points.copy()
    current_det = areas_batch(current[None, :, :])[0]
    current_value = float(current_det.min())

    directions = np.array([
        [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
        [0.7071067811865476, 0.7071067811865476],
        [0.7071067811865476, -0.7071067811865476],
        [-0.7071067811865476, 0.7071067811865476],
        [-0.7071067811865476, -0.7071067811865476],
    ])

    probe = 0.014
    for _ in range(15):
        improved = False
        active = _TRIANGLES_13[np.argpartition(current_det, 17)[:18]]
        indices = np.unique(active[active >= outer_count])

        for index in indices:
            base = current[index].copy()
            for direction in directions:
                trial = current.copy()
                trial[index] = base + probe * direction
                project_disk(trial[index:index + 1], radius)
                trial_det = areas_batch(trial[None, :, :])[0]
                trial_value = float(trial_det.min())
                if trial_value > current_value + 1.0e-14:
                    current = trial
                    current_det = trial_det
                    current_value = trial_value
                    improved = True
                    break

        if not improved:
            probe *= 0.52

    if current_value / hull_area > best_value:
        best_points = current

    return best_points


# EVOLVE-BLOCK-END