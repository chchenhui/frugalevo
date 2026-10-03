# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically search in a reference triangular hull.

    Affine invariance permits fixing three hull vertices at (0,0), (1,0),
    and (0,1).  The absolute determinant of every triple is consequently
    its area divided by the hull area, avoiding variable-hull noise.
    """
    n = 13
    rng = np.random.default_rng(13051957)
    triples = np.asarray(
        [(i, j, k) for i in range(n - 2) for j in range(i + 1, n - 1)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    hull = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))

    def project_simplex(q: np.ndarray) -> np.ndarray:
        q = np.maximum(q, 0.0005)
        total = float(q[0] + q[1])
        if total > 0.999:
            q *= 0.999 / total
        return q

    def normalized_areas(p: np.ndarray) -> np.ndarray:
        a, b, c = p[triples[:, 0]], p[triples[:, 1]], p[triples[:, 2]]
        return np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
            (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def evaluate(p: np.ndarray) -> tuple[float, float]:
        normalized = normalized_areas(p)
        # The lower-tail term makes moves that improve several critical
        # triangles acceptable before the strict minimum itself changes.
        tail = np.partition(normalized, 11)[:12]
        return float(normalized.min()), float(normalized.min() + 0.12 * tail.mean())

    best_points = None
    best_score = -np.inf

    # Independent deterministic starts substantially reduce sensitivity to
    # the many local optima of the nonsmooth minimum-area objective.
    for restart in range(6):
        interior = rng.random((n - 3, 2))
        reflected = interior.sum(axis=1) > 1.0
        interior[reflected] = 1.0 - interior[reflected]
        current = np.vstack((hull, interior))
        current_score, current_energy = evaluate(current)

        for step in range(14000):
            progress = step / 13999.0
            sigma = 0.115 * (1.0 - progress) ** 1.7 + 0.0025
            candidate = current.copy()
            changed = 1 if rng.random() < 0.82 else 2
            indices = rng.choice(np.arange(3, n), size=changed, replace=False)
            candidate[indices] += rng.normal(0.0, sigma, size=(changed, 2))
            for index in indices:
                candidate[index] = project_simplex(candidate[index])

            candidate_score, candidate_energy = evaluate(candidate)
            temperature = 0.0035 * (1.0 - progress) ** 2 + 0.000015
            if (candidate_energy >= current_energy or
                    rng.random() < np.exp((candidate_energy - current_energy) / temperature)):
                current = candidate
                current_score, current_energy = candidate_score, candidate_energy

            if current_score > best_score:
                best_score = current_score
                best_points = current.copy()

    # Deterministic active-set polishing targets vertices occurring in the
    # lower-tail triangles, rather than spending probes on inactive points.
    current = best_points.copy()
    current_score, _ = evaluate(current)
    directions = np.array(
        ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
         (0.7071067811865476, 0.7071067811865476),
         (0.7071067811865476, -0.7071067811865476),
         (-0.7071067811865476, 0.7071067811865476),
         (-0.7071067811865476, -0.7071067811865476))
    )
    radius = 0.010
    for _ in range(18):
        improved = False
        areas = normalized_areas(current)
        active_triples = triples[np.argpartition(areas, 15)[:16]]
        active = np.unique(active_triples[active_triples >= 3])
        for index in active:
            base = current[index].copy()
            radial = base - np.array((1.0 / 3.0, 1.0 / 3.0))
            radial_norm = np.linalg.norm(radial)
            probes = directions if radial_norm < 1e-12 else np.vstack(
                (directions, radial / radial_norm, -radial / radial_norm)
            )
            for direction in probes:
                candidate = current.copy()
                candidate[index] = project_simplex(base + radius * direction)
                candidate_score, _ = evaluate(candidate)
                if candidate_score > current_score + 1e-14:
                    current, current_score = candidate, candidate_score
                    improved = True
                    break
        if not improved:
            radius *= 0.55

    return current


# EVOLVE-BLOCK-END