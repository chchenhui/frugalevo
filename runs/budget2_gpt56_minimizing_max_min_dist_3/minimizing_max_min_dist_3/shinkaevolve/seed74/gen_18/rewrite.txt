# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen points in R^3 maximizing minimum squared distance
    divided by maximum squared distance.

    The objective is invariant under translation and uniform scaling, so
    every refinement step recenters and RMS-normalizes the configuration.
    """
    n = 14
    pair_i, pair_j = np.triu_indices(n, 1)

    def normalize(points: np.ndarray) -> np.ndarray:
        points = points - points.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(points * points, axis=1)))
        return points / max(rms, 1e-14)

    def exact_ratio(points: np.ndarray) -> float:
        delta = points[pair_i] - points[pair_j]
        squared = np.einsum("ij,ij->i", delta, delta)
        return float(squared.min() / squared.max())

    def soft_gradient(points: np.ndarray, sharpness: float) -> np.ndarray:
        """
        Gradient of a stable soft approximation to
            log(min distance) - log(max distance).
        """
        delta = points[pair_i] - points[pair_j]
        squared = np.einsum("ij,ij->i", delta, delta)
        squared = np.maximum(squared, 1e-14)
        logs = 0.5 * np.log(squared)

        lo = logs.min()
        hi = logs.max()

        near = np.exp(-sharpness * (logs - lo))
        far = np.exp(sharpness * (logs - hi))
        near /= near.sum()
        far /= far.sum()

        # d log(||xi-xj||) / d xi = (xi-xj) / ||xi-xj||^2.
        weights = near - far
        pair_force = weights[:, None] * delta / squared[:, None]

        gradient = np.zeros_like(points)
        np.add.at(gradient, pair_i, pair_force)
        np.add.at(gradient, pair_j, -pair_force)
        gradient -= gradient.mean(axis=0, keepdims=True)
        return gradient

    def refine(seed: np.ndarray, iterations: int, polish: bool = False) -> np.ndarray:
        points = normalize(seed.astype(float, copy=True))
        best = points.copy()
        best_value = exact_ratio(points)
        velocity = np.zeros_like(points)

        for iteration in range(iterations):
            progress = iteration / max(iterations - 1, 1)

            if polish:
                # The final pass concentrates on the actual active contact
                # graph, but starts moderately enough to permit coordinated
                # adjustments of several nearly-active pairs.
                sharpness = 55.0 + 445.0 * progress * progress
                step_size = 0.020 * (1.0 - 0.62 * progress)
                momentum = 0.80
            else:
                # A delayed sharp phase avoids prematurely freezing a
                # suboptimal nearest-pair graph.
                sharpness = 5.0 + 255.0 * progress ** 1.65
                step_size = 0.043 * (1.0 - 0.64 * progress)
                momentum = 0.84

            gradient = soft_gradient(points, sharpness)
            norm = np.sqrt(np.mean(np.sum(gradient * gradient, axis=1)))
            if norm > 1e-15:
                gradient /= norm

            velocity = momentum * velocity + (1.0 - momentum) * gradient
            points = normalize(points + step_size * velocity)

            # The smoothed target is only a surrogate.  Preserve the best
            # unsmoothed evaluator objective during every continuation.
            if iteration % 12 == 11 or iteration == iterations - 1:
                value = exact_ratio(points)
                if value > best_value:
                    best_value = value
                    best = points.copy()

        return best

    def layered_seed(radius: float, height: float, twist: float, pole: float,
                     perturbation: float, seed: int) -> np.ndarray:
        angles = np.arange(6, dtype=float) * (np.pi / 3.0)
        lower = np.column_stack((
            radius * np.cos(angles),
            radius * np.sin(angles),
            -height * np.ones(6),
        ))
        upper = np.column_stack((
            radius * np.cos(angles + twist),
            radius * np.sin(angles + twist),
            height * np.ones(6),
        ))
        points = np.vstack((lower, upper, [[0.0, 0.0, -pole],
                                           [0.0, 0.0, pole]]))
        if perturbation > 0.0:
            rng = np.random.default_rng(seed)
            points = points + perturbation * rng.standard_normal((n, 3))
        return points

    initializers = []

    # Near the highly competitive staggered-hexagon contact graph.  The
    # variations deliberately change all four meaningful shape parameters,
    # rather than merely rotating or uniformly scaling the same layout.
    layer_parameters = (
        (1.00, 0.43, np.pi / 6.0, 1.30),
        (1.00, 0.49, np.pi / 6.0, 1.36),
        (1.06, 0.46, 0.46,       1.37),
        (0.95, 0.53, 0.57,       1.34),
        (1.08, 0.39, 0.50,       1.29),
        (0.99, 0.58, 0.62,       1.43),
    )
    for k, parameters in enumerate(layer_parameters):
        initializers.append(layered_seed(*parameters, perturbation=0.012,
                                         seed=1009 + 97 * k))

    # A differently organized high-symmetry basin: cube corners plus
    # independently anisotropic axial points.
    cube = np.array(
        [[x, y, z]
         for x in (-1.0, 1.0)
         for y in (-1.0, 1.0)
         for z in (-1.0, 1.0)],
        dtype=float,
    )
    for scales in ((1.53, 1.67, 1.78), (1.70, 1.54, 1.63)):
        axes = np.diag(scales)
        initializers.append(np.vstack((cube, axes, -axes)))

    # A small random component still permits discovery of unrelated basins,
    # while keeping most computation devoted to geometric initializers.
    for seed in (271828, 314159):
        rng = np.random.default_rng(seed)
        initializers.append(rng.standard_normal((n, 3)))

    best_points = None
    best_ratio = -np.inf

    for initial in initializers:
        candidate = refine(initial, iterations=1700, polish=False)
        value = exact_ratio(candidate)
        if value > best_ratio:
            best_ratio = value
            best_points = candidate

    # A sharper final continuation is substantially more useful when begun
    # from the strongest contact graph than when spent on every weak start.
    polished = refine(best_points, iterations=1200, polish=True)
    polished_ratio = exact_ratio(polished)
    if polished_ratio > best_ratio:
        best_points = polished

    return np.asarray(best_points, dtype=float)


# EVOLVE-BLOCK-END