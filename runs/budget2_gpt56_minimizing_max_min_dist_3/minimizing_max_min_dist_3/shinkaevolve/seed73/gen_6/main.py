# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct a reproducible, diameter-normalized 14 point packing in R^3."""
    n, d = 14, 3
    pair_i, pair_j = np.triu_indices(n, 1)
    rng = np.random.default_rng(42014)

    def normalize_diameter(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        delta = x[pair_i] - x[pair_j]
        diameter = np.sqrt(np.sum(delta * delta, axis=1)).max()
        return x / diameter

    def true_score(x: np.ndarray) -> float:
        delta = x[pair_i] - x[pair_j]
        distances = np.sqrt(np.sum(delta * delta, axis=1))
        return float((distances.min() / distances.max()) ** 2)

    best_points = None
    best_score = -np.inf

    # The scale is removed after every move, so these are starts in the
    # quotient space of configurations modulo translation and scaling.
    for restart in range(18):
        directions = rng.normal(size=(n, d))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.random(n) ** (1.0 / d)
        points = normalize_diameter(directions * radii[:, None])

        velocity = np.zeros_like(points)
        for iteration in range(2200):
            # Annealing changes a broad smooth packing energy into a close
            # approximation of the active shortest and longest pair problem.
            temperature = 0.032 - 0.026 * iteration / 2199.0
            delta = points[pair_i] - points[pair_j]
            distances = np.sqrt(np.sum(delta * delta, axis=1)) + 1e-12

            short_logits = -distances / temperature
            short_weights = np.exp(short_logits - short_logits.max())
            short_weights /= short_weights.sum()

            long_logits = distances / temperature
            long_weights = np.exp(long_logits - long_logits.max())
            long_weights /= long_weights.sum()

            # Repel pairs which control the soft minimum, and contract pairs
            # which control the soft maximum.  Diameter normalization below
            # converts this into ascent of the separation/diameter ratio.
            pair_force = (short_weights - long_weights)[:, None] * delta
            pair_force /= distances[:, None]
            gradient = np.zeros_like(points)
            np.add.at(gradient, pair_i, pair_force)
            np.add.at(gradient, pair_j, -pair_force)

            step = 0.055 * (1.0 - 0.65 * iteration / 2199.0)
            velocity = 0.72 * velocity + step * gradient
            points = normalize_diameter(points + velocity)

            if iteration % 40 == 0:
                score = true_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

        score = true_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    # The broad search generally identifies the correct packing family, but
    # the optimum is controlled by only a few contact pairs.  Re-starting
    # close to the incumbent with a much sharper soft extremum is an effective
    # deterministic basin-hopping polish for those contacts.
    for polish_restart in range(12):
        points = best_points + rng.normal(scale=0.012, size=(n, d))
        points = normalize_diameter(points)
        velocity = np.zeros_like(points)

        for iteration in range(650):
            temperature = 0.012 - 0.0095 * iteration / 649.0
            delta = points[pair_i] - points[pair_j]
            distances = np.sqrt(np.sum(delta * delta, axis=1)) + 1e-12

            short_logits = -distances / temperature
            short_weights = np.exp(short_logits - short_logits.max())
            short_weights /= short_weights.sum()

            long_logits = distances / temperature
            long_weights = np.exp(long_logits - long_logits.max())
            long_weights /= long_weights.sum()

            pair_force = (short_weights - long_weights)[:, None] * delta
            pair_force /= distances[:, None]
            gradient = np.zeros_like(points)
            np.add.at(gradient, pair_i, pair_force)
            np.add.at(gradient, pair_j, -pair_force)

            step = 0.026 * (1.0 - 0.55 * iteration / 649.0)
            velocity = 0.58 * velocity + step * gradient
            points = normalize_diameter(points + velocity)

            if iteration % 20 == 0:
                score = true_score(points)
                if score > best_score:
                    best_score = score
                    best_points = points.copy()

        score = true_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END