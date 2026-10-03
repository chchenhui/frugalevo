# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Deterministically perform multi-start, lexicographic maximin search in the
    unit right-triangle hull, followed by a fine local polish of the best
    arrangement.  Absolute determinants are normalized triangle areas because
    the fixed hull has area one half.
    """
    # Fixed seed makes the multi-start stochastic search reproducible.
    rng = np.random.default_rng(20260912)
    hull = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    triples = np.array(
        [(i, j, k) for i in range(13) for j in range(i + 1, 13)
         for k in range(j + 1, 13)], dtype=np.intp
    )

    def project_triangle(p):
        """Project candidate interior points into the fixed right-triangle hull."""
        p = np.maximum(p, 0.001)
        return p / np.maximum(1.0, p.sum(axis=-1, keepdims=True) / 0.998)

    def areas(configs):
        a, b, c = (configs[:, triples[:, q]] for q in range(3))
        return np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def select(values):
        """Select lexicographically strong candidates using active triangle areas."""
        # Retaining more of the active constraint set produces configurations
        # whose bottleneck is less likely to collapse during final refinement.
        small = np.partition(values, 7, axis=1)[:, :8]
        small.sort(axis=1)
        key = (small[:, 0] * 1.0e6 + small[:, 1] * 1.0e3
               + small[:, 2] * 1.0e1 + small[:, 3:].mean(axis=1))
        return int(np.argmax(key)), small[:, 0]

    best = None
    best_min = -np.inf
    batch = 24

    # Sixteen longer trajectories gave a better exploration/refinement balance
    # than many shorter independent trajectories.
    for restart in range(16):
        u = rng.random((10, 2))
        u[u.sum(axis=1) > 1.0] = 1.0 - u[u.sum(axis=1) > 1.0]
        current = np.vstack((hull, project_triangle(u)))

        # A near-lattice start covers a basin which pure uniform starts seldom
        # enter, while the fixed jitter avoids exact collinear triples.
        if restart == 0:
            current[3:] = project_triangle(np.array([
                [.16, .12], [.42, .10], [.73, .10], [.10, .38],
                [.34, .34], [.59, .29], [.10, .68], [.27, .56],
                [.48, .47], [.20, .23]
            ]) + rng.normal(0.0, 0.018, (10, 2)))

        for iteration in range(2600):
            t = iteration / 2599.0
            sigma = 0.105 * (1.0 - t) ** 1.7 + 0.0012
            candidates = np.repeat(current[None, :, :], batch, axis=0)

            for q in range(1, batch):
                # Draw directly in the valid movable-index interval [3, 13).
                # Independent draws also allow a repeated index in a rare
                # two-point proposal, yielding a useful larger one-point move
                # without ever producing the invalid index 13.
                changed = [int(rng.integers(3, 13))]
                if rng.random() < 0.22:
                    changed.append(int(rng.integers(3, 13)))
                candidates[q, changed] += rng.normal(
                    0.0, sigma, size=(len(changed), 2)
                )

            candidates[:, 3:] = project_triangle(candidates[:, 3:])
            chosen, minima = select(areas(candidates))
            current = candidates[chosen]

            value = float(minima[chosen])
            if value > best_min:
                best_min = value
                best = current.copy()

    # A wider final batch is valuable because only a few nearly equal active
    # triangles remain at this point.  Fine moves equalize those constraints.
    current = best.copy()
    polish_batch = 48
    for iteration in range(1800):
        sigma = 0.006 * (1.0 - iteration / 1799.0) ** 1.8 + 0.00008
        candidates = np.repeat(current[None, :, :], polish_batch, axis=0)
        for q in range(1, polish_batch):
            count = 1 if q < 35 else 2
            changed = rng.choice(10, count, replace=False) + 3
            candidates[q, changed] += rng.normal(0.0, sigma, (count, 2))
        candidates[:, 3:] = project_triangle(candidates[:, 3:])
        chosen, minima = select(areas(candidates))
        current = candidates[chosen]
        if minima[chosen] > best_min:
            best_min = float(minima[chosen])
            best = current.copy()

    # Final strict-maximin active-set refinement.  Unlike the earlier
    # lexicographic surrogate, this phase accepts candidates according to the
    # true objective alone, so every accepted move preserves or improves the
    # smallest normalized triangle area.  It is deliberately started from the
    # global incumbent rather than the last exploratory state.
    if best is not None and np.isfinite(best).all():
        current = best.copy()
        strict_batch = 64
        for iteration in range(1200):
            t = iteration / 1199.0
            sigma = 0.0025 * (1.0 - t) ** 1.65 + 0.000025
            candidates = np.repeat(current[None, :, :], strict_batch, axis=0)

            for q in range(1, strict_batch):
                count = 1 if q < 46 else 2
                changed = rng.choice(10, count, replace=False) + 3
                candidates[q, changed] += rng.normal(
                    0.0, sigma, size=(count, 2)
                )

            candidates[:, 3:] = project_triangle(candidates[:, 3:])
            values = areas(candidates)
            minima = values.min(axis=1)

            # Strictly optimize the actual Heilbronn objective in the final
            # local stage.  Candidate zero is the unchanged incumbent.
            chosen = int(np.argmax(minima))
            current = candidates[chosen]
            if minima[chosen] > best_min:
                best_min = float(minima[chosen])
                best = current.copy()

    if best is None or not np.isfinite(best).all():
        best = np.vstack((hull, np.array([
            [.12, .12], [.38, .10], [.68, .10], [.10, .38], [.34, .32],
            [.58, .25], [.10, .66], [.25, .55], [.46, .43], [.20, .22],
        ])))
    return best


# EVOLVE-BLOCK-END
