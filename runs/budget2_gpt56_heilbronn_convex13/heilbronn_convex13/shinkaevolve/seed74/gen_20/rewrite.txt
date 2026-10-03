# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex13() -> np.ndarray:
    """
    Construct thirteen deterministic points inside an equilateral triangular
    convex hull.  The hull vertices are fixed and the ten remaining points are
    maintained in barycentric coordinates, so every returned point is feasible.

    The search uses parallel batched local-search chains.  It begins with a
    broad lower-tail triangle-area surrogate, then anneals toward the true
    minimum-area objective and finally applies an exact bottleneck polish.
    """
    rng = np.random.default_rng(130131957)

    hull = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3.0) * 0.5],
        ],
        dtype=float,
    )

    triples = np.array(
        [
            (i, j, k)
            for i in range(13)
            for j in range(i + 1, 13)
            for k in range(j + 1, 13)
        ],
        dtype=np.intp,
    )

    chains = 6
    free_count = 10
    proposals = 20

    def normalize_barycentric(weights: np.ndarray) -> np.ndarray:
        """Project positive perturbed weights back to the barycentric simplex."""
        weights = np.maximum(weights, 2.0e-5)
        return weights / weights.sum(axis=-1, keepdims=True)

    def points_from_weights(weights: np.ndarray) -> np.ndarray:
        """
        Convert (..., 10, 3) barycentric weights to (..., 13, 2) coordinates.
        """
        coords = weights @ hull
        lead_shape = coords.shape[:-2]
        fixed = np.broadcast_to(hull, lead_shape + hull.shape)
        return np.concatenate((fixed, coords), axis=-2)

    def doubled_triangle_areas(weights: np.ndarray) -> np.ndarray:
        """
        Return determinants for all triples.  For the fixed hull, maximizing
        this value is precisely equivalent to hull-normalized area maximization.
        """
        points = points_from_weights(weights)
        a = points[..., triples[:, 0], :]
        b = points[..., triples[:, 1], :]
        c = points[..., triples[:, 2], :]
        return np.abs(
            (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
        )

    def scores(area_values: np.ndarray, tail_weight: float) -> tuple:
        """Compute both exact bottleneck values and a robust lower-tail score."""
        lower = np.partition(area_values, 11, axis=-1)[..., :12]
        minimum = lower[..., 0]
        surrogate = minimum + tail_weight * lower.mean(axis=-1)
        return minimum, surrogate

    # Independent Dirichlet starts give broad coverage.  The first starts also
    # receive mild deterministic radial structure, useful for escaping purely
    # accidental random configurations.
    states = rng.dirichlet(np.ones(3), size=(chains, free_count))
    states = 0.018 + 0.946 * states

    template = np.array(
        [
            [0.72, 0.14, 0.14],
            [0.14, 0.72, 0.14],
            [0.14, 0.14, 0.72],
            [0.52, 0.34, 0.14],
            [0.14, 0.52, 0.34],
            [0.34, 0.14, 0.52],
            [0.42, 0.42, 0.16],
            [0.16, 0.42, 0.42],
            [0.42, 0.16, 0.42],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ],
        dtype=float,
    )
    states[0] = template
    states[1] = normalize_barycentric(
        template + rng.normal(0.0, 0.055, size=(free_count, 3))
    )

    state_areas = doubled_triangle_areas(states)
    state_values, _ = scores(state_areas, 0.30)

    best_chain = int(np.argmax(state_values))
    best_weights = states[best_chain].copy()
    best_value = float(state_values[best_chain])

    # Batched exploration: each chain retains its best surrogate proposal from
    # a group of local and coordinated moves.  Candidate zero is the unchanged
    # state, ensuring the annealed surrogate never degrades within a chain.
    exploration_steps = 2300
    for iteration in range(exploration_steps):
        fraction = iteration / float(exploration_steps - 1)
        move_scale = 0.082 * (1.0 - fraction) ** 1.75 + 0.0010
        tail_weight = 0.32 * (1.0 - fraction) ** 1.35 + 0.035

        batch = np.broadcast_to(
            states[:, None, :, :],
            (chains, proposals, free_count, 3),
        ).copy()

        # Most proposals alter a point belonging to the current bottleneck
        # triangle.  Remaining proposals make coordinated moves to change
        # active-triangle combinatorics.
        worst_triples = triples[np.argmin(state_areas, axis=1)]
        for chain in range(chains):
            movable = worst_triples[chain][worst_triples[chain] >= 3] - 3
            for proposal in range(1, proposals):
                if len(movable) and proposal < 15:
                    index = int(movable[(proposal - 1) % len(movable)])
                    batch[chain, proposal, index] += rng.normal(
                        0.0, move_scale, size=3
                    )
                elif proposal < 18:
                    index = int(rng.integers(free_count))
                    batch[chain, proposal, index] += rng.normal(
                        0.0, move_scale * 1.65, size=3
                    )
                else:
                    batch[chain, proposal] += rng.normal(
                        0.0, move_scale * 0.24, size=(free_count, 3)
                    )

        batch = normalize_barycentric(batch)
        batch_areas = doubled_triangle_areas(batch)
        batch_values, batch_scores = scores(batch_areas, tail_weight)

        choices = np.argmax(batch_scores, axis=1)
        rows = np.arange(chains)
        states = batch[rows, choices]
        state_areas = batch_areas[rows, choices]
        state_values = batch_values[rows, choices]

        candidate_chain = int(np.argmax(batch_values))
        candidate_value = float(batch_values[candidate_chain])
        if candidate_value > best_value:
            best_value = candidate_value
            best_weights = batch[candidate_chain, int(
                np.argmax(batch_values[candidate_chain])
            )].copy()

    # Exact maximin polish around the elite.  Each compact batch targets a
    # member of the currently worst triangle, while retaining an unchanged
    # incumbent candidate.
    polish_scales = (0.014, 0.007, 0.0030, 0.0012, 0.00045)
    weights = best_weights.copy()
    areas = doubled_triangle_areas(weights[None, ...])[0]
    value = float(areas.min())

    for scale in polish_scales:
        for _ in range(160):
            batch = np.broadcast_to(weights, (25, free_count, 3)).copy()
            worst = triples[int(np.argmin(areas))]
            movable = worst[worst >= 3] - 3

            for proposal in range(1, 25):
                if len(movable):
                    index = int(movable[(proposal - 1) % len(movable)])
                else:
                    index = int(rng.integers(free_count))

                batch[proposal, index] += rng.normal(0.0, scale, size=3)

                if proposal % 8 == 0:
                    second = int(rng.integers(free_count))
                    batch[proposal, second] += rng.normal(
                        0.0, scale * 0.38, size=3
                    )

            batch = normalize_barycentric(batch)
            candidate_areas = doubled_triangle_areas(batch)
            candidate_values = candidate_areas.min(axis=1)
            chosen = int(np.argmax(candidate_values))

            if candidate_values[chosen] > value + 1.0e-13:
                weights = batch[chosen]
                areas = candidate_areas[chosen]
                value = float(candidate_values[chosen])

    return points_from_weights(weights)


# EVOLVE-BLOCK-END