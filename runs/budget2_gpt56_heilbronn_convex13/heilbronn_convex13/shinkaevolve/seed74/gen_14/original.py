# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Return a deterministic maximin configuration of thirteen points in the
    unit square.  The four corners are retained, hence the convex hull has
    area one and no post-search hull normalization is necessary.
    """
    rng = np.random.default_rng(13051957)

    # Fixing the square corners both guarantees a unit-area convex hull and
    # leaves the search free to distribute the nine interior points.
    corners = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=float,
    )
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=np.intp,
    )

    def areas(p: np.ndarray) -> np.ndarray:
        a = p[triples[:, 0]]
        b = p[triples[:, 1]]
        c = p[triples[:, 2]]
        return 0.5 * np.abs(
            (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        )

    def merit(p: np.ndarray, blend: float) -> tuple[float, float]:
        values = areas(p)
        smallest = np.partition(values, 11)[:12]
        minimum = float(smallest.min())
        # Early in annealing, improving several tight constraints is much
        # more effective than optimizing just one currently active triangle.
        return minimum + blend * float(smallest.mean()), minimum

    best = None
    best_minimum = -1.0

    # Independent deterministic starts are useful because the maximin
    # landscape contains many narrow local optima.
    for restart in range(4):
        points = np.vstack((corners, rng.uniform(0.065, 0.935, size=(9, 2))))
        current_merit, current_minimum = merit(points, 0.09)

        for iteration in range(45000):
            fraction = iteration / 44999.0
            blend = 0.09 * (1.0 - fraction)
            step = 0.105 * (1.0 - fraction) ** 1.7 + 0.0012

            trial = points.copy()
            index = int(rng.integers(4, 13))
            trial[index] += rng.normal(0.0, step, size=2)
            trial[index] = np.clip(trial[index], 0.065, 0.935)

            trial_merit, trial_minimum = merit(trial, blend)
            temperature = 0.0028 * (1.0 - fraction) ** 2 + 0.000002
            if (trial_merit >= current_merit or
                    rng.random() < np.exp((trial_merit - current_merit) / temperature)):
                points = trial
                current_merit = trial_merit
                current_minimum = trial_minimum

            if current_minimum > best_minimum:
                best = points.copy()
                best_minimum = current_minimum

    # A short coordinate-pattern search removes residual annealing noise while
    # accepting only improvements to the actual primary objective.
    directions = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=float,
    )
    directions[4:] /= np.sqrt(2.0)
    step = 0.012
    for _ in range(8):
        improved = True
        while improved:
            improved = False
            for index in range(4, 13):
                for direction in directions:
                    trial = best.copy()
                    trial[index] = np.clip(
                        trial[index] + step * direction, 0.065, 0.935
                    )
                    candidate_minimum = float(areas(trial).min())
                    if candidate_minimum > best_minimum:
                        best = trial
                        best_minimum = candidate_minimum
                        improved = True
        step *= 0.55

    return best


# EVOLVE-BLOCK-END