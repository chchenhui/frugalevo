# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically optimize ten points inside a fixed unit-area-normalized
    right triangular hull using batched multi-start soft-min local search.

    The three fixed hull vertices are (0,0), (1,0), and (0,1).  Their hull has
    area 1/2, so the normalized area of a triangle is simply the absolute
    cross-product determinant.  This removes hull-normalization instability
    while retaining a valid convex containing region.
    """
    rng = np.random.default_rng(20260912)

    # The enclosing convex hull is the right triangle with these vertices.
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)

    # Indices of all 286 triples, computed once per constructor invocation.
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)],
        dtype=int,
    )

    def triangle_values(configs: np.ndarray) -> np.ndarray:
        """Return normalized (doubled) areas for every triple in each batch."""
        a = configs[:, triples[:, 0]]
        b = configs[:, triples[:, 1]]
        c = configs[:, triples[:, 2]]
        return np.abs(
            (b[:, :, 0] - a[:, :, 0]) * (c[:, :, 1] - a[:, :, 1])
            - (b[:, :, 1] - a[:, :, 1]) * (c[:, :, 0] - a[:, :, 0])
        )

    def maximin_score(areas: np.ndarray) -> np.ndarray:
        """Lexicographically prioritize the smallest triangle areas."""
        small = np.partition(areas, 7, axis=1)[:, :8]
        small.sort(axis=1)
        # The minimum is deliberately dominant.  The remaining active areas
        # only distinguish candidates with nearly identical bottlenecks.
        return (small[:, 0] * 1.0e6
                + small[:, 1] * 1.0e3
                + small[:, 2] * 10.0
                + small[:, 3:].mean(axis=1))

    def random_triangle_points(count: int) -> np.ndarray:
        """Uniform deterministic samples in x >= 0, y >= 0, x+y <= 1."""
        u = rng.random((count, 2))
        reflect = u.sum(axis=1) > 1.0
        u[reflect] = 1.0 - u[reflect]
        return u

    best = None
    best_value = -np.inf
    batch_size = 24

    # Multiple starts make the non-convex maximin search substantially more
    # reliable while remaining inexpensive: only 286 vectorized areas per trial.
    for restart in range(16):
        current = np.vstack((hull, random_triangle_points(10)))

        # A mildly regular first start gives the search a useful non-random basin.
        if restart == 0:
            current[3:] = np.array([
                [0.16, 0.12], [0.42, 0.10], [0.73, 0.10],
                [0.10, 0.38], [0.34, 0.34], [0.59, 0.29],
                [0.10, 0.68], [0.27, 0.56], [0.48, 0.47],
                [0.20, 0.23],
            ])
            current[3:] += rng.normal(0.0, 0.018, size=(10, 2))

        # Project initialization back into the containing triangle.
        current[3:] = np.maximum(current[3:], 0.001)
        sums = current[3:].sum(axis=1)
        mask = sums > 0.998
        current[3:][mask] *= (0.998 / sums[mask])[:, None]

        for step in range(2600):
            progress = step / 2599.0
            sigma = 0.105 * (1.0 - progress) ** 1.7 + 0.0012

            candidates = np.repeat(current[None, :, :], batch_size, axis=0)

            # Most moves alter one point; some coordinated two-point moves help
            # escape the small local traps caused by nearly active triangles.
            for q in range(1, batch_size):
                changed = [int(rng.integers(3, 13))]
                if rng.random() < 0.22:
                    changed.append(int(rng.integers(3, 13)))
                candidates[q, changed] += rng.normal(
                    0.0, sigma, size=(len(changed), 2)
                )

            interior = candidates[:, 3:, :]
            interior[:] = np.maximum(interior, 0.001)
            sums = interior.sum(axis=2)
            outside = sums > 0.998
            interior[outside] *= (0.998 / sums[outside])[:, None]

            areas = triangle_values(candidates)
            chosen = int(np.argmax(maximin_score(areas)))
            current = candidates[chosen]

            # Reuse the already computed values; the selected candidate is
            # exactly current, so a second 286-triangle evaluation is needless.
            candidate_exact = float(areas[chosen].min())
            if candidate_exact > best_value:
                best_value = candidate_exact
                best = current.copy()

    # A final deterministic local polish is important because the main search
    # retains a nonzero exploratory move size.  Starting from the globally best
    # exact configuration, use much smaller one- and two-point perturbations to
    # equalize the few triangles that are active at the maximin optimum.
    if best is not None and np.isfinite(best).all():
        current = best.copy()
        polish_batch = 48
        for step in range(1800):
            progress = step / 1799.0
            sigma = 0.006 * (1.0 - progress) ** 1.8 + 0.00008
            candidates = np.repeat(current[None, :, :], polish_batch, axis=0)

            for q in range(1, polish_batch):
                # Mostly single-coordinate-point corrections; a minority of
                # paired moves avoids being trapped by coupled active triples.
                count = 1 if q < 35 else 2
                changed = rng.choice(10, count, replace=False) + 3
                candidates[q, changed] += rng.normal(
                    0.0, sigma, size=(count, 2)
                )

            interior = candidates[:, 3:, :]
            interior[:] = np.maximum(interior, 0.0002)
            sums = interior.sum(axis=2)
            outside = sums > 0.9996
            interior[outside] *= (0.9996 / sums[outside])[:, None]

            values = triangle_values(candidates)
            chosen = int(np.argmax(maximin_score(values)))
            current = candidates[chosen]
            exact = float(values[chosen].min())
            if exact > best_value:
                best_value = exact
                best = current.copy()

    # Defensive fallback is unreachable in normal operation but guarantees a
    # finite, valid answer if a numerical platform reports an unexpected value.
    if best is None or not np.isfinite(best).all():
        best = np.vstack((hull, random_triangle_points(10)))

    return best


# EVOLVE-BLOCK-END
