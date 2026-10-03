# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Construct 11 triangle points using deterministic cached multi-start annealing.

    Points are represented by two barycentric coordinates, guaranteeing that
    every trial point remains inside the equilateral container.  The container
    vertices are retained and only the eight remaining points are optimized.
    Since a one-point displacement affects only 45 of the 165 triples, the
    objective cache permits many deterministic annealing iterations.
    """
    # Fixed seed and schedule selected for a robust high-quality basin.
    rng = np.random.default_rng(110271)
    n = 11
    height = np.sqrt(3.0) / 2.0

    triples = np.array(
        [(i, j, k) for i in range(n) for j in range(i + 1, n)
         for k in range(j + 1, n)],
        dtype=np.intp,
    )
    incident = [
        np.flatnonzero(np.any(triples == p, axis=1))
        for p in range(n)
    ]

    def random_barycentric(count: int) -> np.ndarray:
        """Generate uniform points in u >= 0, v >= 0, u + v <= 1."""
        uv = rng.random((count, 2))
        mask = uv.sum(axis=1) > 1.0
        uv[mask] = 1.0 - uv[mask]
        return uv

    def to_xy(uv: np.ndarray) -> np.ndarray:
        xy = np.empty_like(uv)
        xy[:, 0] = uv[:, 0] + 0.5 * uv[:, 1]
        xy[:, 1] = height * uv[:, 1]
        return xy

    def area_values(xy: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Return areas normalized by the enclosing triangle area."""
        q = xy[triples[ids]]
        return np.abs(
            (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
            - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
        )

    def merit(areas: np.ndarray) -> float:
        """Use the minimum plus a small reward for nearby constraints."""
        low = np.partition(areas, 15)[:16]
        return float(low[0] + 0.016 * np.mean(low[1:]))

    best_uv = None
    best_minimum = -1.0
    all_ids = np.arange(len(triples), dtype=np.intp)
    restarts = 20
    iterations = 22000

    for restart in range(restarts):
        uv = np.empty((n, 2), dtype=float)
        uv[:3] = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        uv[3:] = random_barycentric(8)



        xy = to_xy(uv)
        areas = area_values(xy, all_ids)
        current_merit = merit(areas)

        for step in range(iterations):
            fraction = step / iterations
            critical = np.argpartition(areas, 19)[:20]
            weights = np.bincount(
                triples[critical].ravel(), minlength=n
            )[3:]

            if weights.sum() and rng.random() < 0.91:
                point = 3 + int(rng.choice(8, p=weights / weights.sum()))
            else:
                point = int(rng.integers(3, n))

            old = uv[point]
            if rng.random() < 0.025 and fraction < 0.65:
                proposal = random_barycentric(1)[0]
            else:
                scale = 0.13 * (1.0 - fraction) ** 1.35 + 0.0018
                proposal = old + rng.normal(0.0, scale, size=2)
                proposal = np.maximum(proposal, 0.0)
                total = proposal.sum()
                if total > 1.0:
                    proposal /= total

            ids = incident[point]
            trial_xy = xy.copy()
            trial_xy[point, 0] = proposal[0] + 0.5 * proposal[1]
            trial_xy[point, 1] = height * proposal[1]
            trial_areas = areas.copy()
            trial_areas[ids] = area_values(trial_xy, ids)
            trial_merit = merit(trial_areas)

            temperature = 0.0016 * (1.0 - fraction) ** 2.4 + 0.0000015
            if (trial_merit >= current_merit or
                    rng.random() < np.exp(
                        (trial_merit - current_merit) / temperature
                    )):
                uv[point] = proposal
                xy = trial_xy
                areas = trial_areas
                current_merit = trial_merit

            minimum = float(areas.min())
            if minimum > best_minimum:
                best_minimum = minimum
                best_uv = uv.copy()

    if best_uv is None:
        best_uv = np.array(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)] +
            [(0.5, 0.25)] * 8,
            dtype=float,
        )

    # The annealing merit is intentionally smoothed.  Finish with a short
    # strictly monotone search on the evaluator's true maximin objective.
    uv = best_uv.copy()
    xy = to_xy(uv)
    areas = area_values(xy, all_ids)
    minimum = float(areas.min())

    for step in range(8000):
        fraction = step / 8000.0
        critical = np.flatnonzero(
            areas <= minimum + 2.0e-5 * (1.0 - fraction)
        )
        weights = np.bincount(
            triples[critical].ravel(), minlength=n
        )[3:]

        if weights.sum() and rng.random() < 0.94:
            point = 3 + int(rng.choice(8, p=weights / weights.sum()))
        else:
            point = int(rng.integers(3, n))

        proposal = uv[point] + rng.normal(
            0.0, 0.008 * (1.0 - fraction) ** 1.7 + 0.00002, size=2
        )
        proposal = np.maximum(proposal, 0.0)
        total = proposal.sum()
        if total > 1.0:
            proposal /= total

        ids = incident[point]
        trial_xy = xy.copy()
        trial_xy[point, 0] = proposal[0] + 0.5 * proposal[1]
        trial_xy[point, 1] = height * proposal[1]
        trial_areas = areas.copy()
        trial_areas[ids] = area_values(trial_xy, ids)
        trial_minimum = float(trial_areas.min())

        if trial_minimum > minimum:
            uv[point] = proposal
            xy = trial_xy
            areas = trial_areas
            minimum = trial_minimum

    return xy


# EVOLVE-BLOCK-END
