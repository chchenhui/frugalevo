# EVOLVE-BLOCK-START
import numpy as np


_N = 13
_FIXED = 4
_TRIPLES = np.asarray(
    [(i, j, k) for i in range(_N - 2) for j in range(i + 1, _N - 1)
     for k in range(j + 1, _N)],
    dtype=np.intp,
)


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically construct a maximin thirteen-point configuration.

    Four points are fixed at the corners of the unit square.  Thus the convex
    hull has exactly unit area throughout optimization, all other points are
    automatically feasible, and a doubled triangle determinant is identical
    to the normalized triangle-area score.
    """
    rng = np.random.default_rng(13051957)

    # Counter-clockwise square: its hull area is one.
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )

    def areas(points: np.ndarray) -> np.ndarray:
        t = points[_TRIPLES]
        return np.abs(
            (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
            - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
        )

    def tail_energy(values: np.ndarray, count: int) -> float:
        """Strict minimum plus a weighted active-constraint continuation."""
        tail = np.partition(values, count - 1)[:count]
        weights = np.linspace(3.8, 1.0, count)
        return float(0.70 * tail[0] +
                     0.30 * np.dot(tail, weights) / weights.sum())

    def active_weights(values: np.ndarray, count: int) -> np.ndarray:
        active = _TRIPLES[np.argpartition(values, count - 1)[:count]].ravel()
        weights = np.bincount(active, minlength=_N).astype(float)
        weights[:_FIXED] = 0.0
        weights[_FIXED:] += 0.22
        return weights / weights.sum()

    def gradient(points: np.ndarray, tri_index: int,
                 vertex: int) -> np.ndarray:
        """Unit gradient increasing absolute doubled area at one vertex."""
        i, j, k = _TRIPLES[tri_index]
        a, b, c = points[i], points[j], points[k]
        signed = ((b[0] - a[0]) * (c[1] - a[1]) -
                  (b[1] - a[1]) * (c[0] - a[0]))
        sign = 1.0 if signed >= 0.0 else -1.0
        if vertex == i:
            edge = c - b
        elif vertex == j:
            edge = a - c
        else:
            edge = b - a
        direction = sign * np.array((-edge[1], edge[0]))
        norm = float(np.hypot(direction[0], direction[1]))
        return direction / max(norm, 1.0e-14)

    def initial_points(restart: int) -> np.ndarray:
        points = np.empty((_N, 2), dtype=float)
        points[:_FIXED] = corners

        # Jittered lattice starts cover the square more evenly than pure
        # random starts; alternating them with random starts preserves basin
        # diversity.
        if restart % 3 != 2:
            grid = np.array([
                [1.0 / 6.0, 1.0 / 6.0], [0.50, 1.0 / 6.0],
                [5.0 / 6.0, 1.0 / 6.0], [1.0 / 6.0, 0.50],
                [0.50, 0.50], [5.0 / 6.0, 0.50],
                [1.0 / 6.0, 5.0 / 6.0], [0.50, 5.0 / 6.0],
                [5.0 / 6.0, 5.0 / 6.0],
            ])
            points[_FIXED:] = grid + rng.normal(0.0, 0.052, grid.shape)
        else:
            points[_FIXED:] = rng.uniform(0.055, 0.945, size=(9, 2))
        np.clip(points[_FIXED:], 0.012, 0.988, out=points[_FIXED:])
        return points

    best_points = None
    best_value = -np.inf

    try:
        # Fixed hull avoids costly hull reconstruction, allowing numerous
        # independent active-set basin searches within the same time budget.
        for restart in range(11):
            points = initial_points(restart)
            current_areas = areas(points)
            current_value = float(current_areas.min())
            current_energy = tail_energy(current_areas, 22)

            for step in range(10500):
                progress = step / 10499.0
                cooling = (1.0 - progress) ** 1.55
                active_count = 26 if progress < 0.42 else (
                    15 if progress < 0.80 else 8
                )
                active = np.argpartition(
                    current_areas, active_count - 1
                )[:active_count]
                trial = points.copy()
                scale = 0.082 * cooling + 0.00055

                # Correlated two-point moves specifically resolve neighboring
                # active constraints which cannot be lifted by a lone vertex.
                if rng.random() < 0.145:
                    tri_index = int(active[rng.integers(active_count)])
                    tri = _TRIPLES[tri_index]
                    movable = tri[tri >= _FIXED]
                    if len(movable) >= 2:
                        moved, other = rng.choice(movable, 2, replace=False)
                        moved, other = int(moved), int(other)
                        normal = gradient(points, tri_index, moved)
                        tangent = np.array((-normal[1], normal[0]))
                        tangent *= 1.0 if rng.random() < 0.5 else -1.0
                        trial[moved] += (0.88 * scale * normal +
                                         rng.normal(0.0, 0.13 * scale, 2))
                        trial[other] += (0.31 * scale * tangent +
                                         rng.normal(0.0, 0.09 * scale, 2))
                    else:
                        index = int(movable[0])
                        trial[index] += rng.normal(0.0, scale, 2)
                else:
                    tri_index = int(active[rng.integers(active_count)])
                    tri = _TRIPLES[tri_index]
                    movable = tri[tri >= _FIXED]
                    if rng.random() < 0.80 and len(movable):
                        index = int(movable[rng.integers(len(movable))])
                    else:
                        weights = active_weights(current_areas, active_count)
                        index = int(rng.choice(_N, p=weights))
                    guided = gradient(points, tri_index, index)
                    noise = rng.normal(size=2)
                    noise /= max(float(np.hypot(noise[0], noise[1])), 1e-14)
                    mix = 0.48 + 0.39 * progress
                    direction = mix * guided + (1.0 - mix) * noise
                    direction /= max(
                        float(np.hypot(direction[0], direction[1])), 1e-14
                    )
                    trial[index] += scale * direction

                np.clip(trial[_FIXED:], 0.0, 1.0, out=trial[_FIXED:])
                trial_areas = areas(trial)
                trial_energy = tail_energy(trial_areas, active_count)
                temperature = 0.00175 * cooling * cooling + 0.0000018
                delta = trial_energy - current_energy

                if delta >= 0.0 or rng.random() < np.exp(delta / temperature):
                    points = trial
                    current_areas = trial_areas
                    current_energy = trial_energy
                    current_value = float(trial_areas.min())

                if current_value > best_value:
                    best_value = current_value
                    best_points = points.copy()

        # Strict maximin polishing: retain only moves that improve the actual
        # grading metric, with tail energy breaking exact minimum ties.
        points = best_points.copy()
        current_areas = areas(points)
        value = float(current_areas.min())
        energy = tail_energy(current_areas, 8)

        directions = np.array(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
             [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
            dtype=float,
        )
        directions[4:] /= np.sqrt(2.0)

        for level in range(34):
            step_size = 0.014 * (0.84 ** level) + 0.000025
            changed = False
            order = np.argsort(-active_weights(current_areas, 10))
            for index in order:
                if index < _FIXED:
                    continue
                chosen = None
                local_value, local_energy = value, energy
                for direction in directions:
                    candidate = points.copy()
                    candidate[index] += step_size * direction
                    np.clip(candidate[index], 0.0, 1.0, out=candidate[index])
                    candidate_areas = areas(candidate)
                    candidate_value = float(candidate_areas.min())
                    candidate_energy = tail_energy(candidate_areas, 8)
                    if (candidate_value > local_value + 1e-14 or
                            (abs(candidate_value - local_value) <= 1e-14 and
                             candidate_energy > local_energy + 1e-14)):
                        chosen = (candidate, candidate_areas)
                        local_value, local_energy = (
                            candidate_value, candidate_energy
                        )
                if chosen is not None:
                    points, current_areas = chosen
                    value, energy = local_value, local_energy
                    changed = True
            if not changed and step_size < 0.00020:
                break

        return points

    except (FloatingPointError, ValueError, RuntimeError):
        fallback = np.empty((_N, 2), dtype=float)
        fallback[:_FIXED] = corners
        fallback[_FIXED:] = np.array([
            [0.18, 0.18], [0.50, 0.18], [0.82, 0.18],
            [0.18, 0.50], [0.50, 0.50], [0.82, 0.50],
            [0.18, 0.82], [0.50, 0.82], [0.82, 0.82],
        ])
        return fallback


# EVOLVE-BLOCK-END