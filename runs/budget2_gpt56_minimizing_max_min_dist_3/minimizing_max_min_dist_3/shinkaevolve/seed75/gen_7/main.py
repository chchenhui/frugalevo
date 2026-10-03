# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y) coordinates of the 14 points.

    """

    n = 14
    d = 3
    pair_i, pair_j = np.triu_indices(n, k=1)

    # The ratio is unchanged by centering and uniform rescaling.  Working
    # on this normalized slice prevents either part of the objective from
    # being improved merely by changing scale.
    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        return x / np.sqrt(np.mean(np.sum(x * x, axis=1)))

    def exact_ratio(x: np.ndarray) -> float:
        delta = x[pair_i] - x[pair_j]
        squared_distances = np.einsum("ij,ij->i", delta, delta)
        return float(squared_distances.min() / squared_distances.max())

    rng = np.random.default_rng(314159)
    best_points = None
    best_ratio = -np.inf

    # A soft min/max continuation is substantially less susceptible to
    # closest-pair switching than optimizing a single nonsmooth pair.
    for _ in range(12):
        points = normalize(rng.normal(size=(n, d)))
        first_moment = np.zeros_like(points)
        second_moment = np.zeros_like(points)

        for iteration in range(1800):
            delta = points[pair_i] - points[pair_j]
            squared_distances = np.einsum("ij,ij->i", delta, delta)

            # Increase sharpness gradually, with a scale-relative inverse
            # temperature so that normalization does not affect the search.
            sharpness = 3.0 * (60.0 ** (iteration / 1799.0))
            beta = sharpness / squared_distances.mean()

            minimum_weights = np.exp(
                -beta * (squared_distances - squared_distances.min())
            )
            minimum_weights /= minimum_weights.sum()
            maximum_weights = np.exp(
                beta * (squared_distances - squared_distances.max())
            )
            maximum_weights /= maximum_weights.sum()

            # Gradient of softmin(q) - softmax(q), q = ||p_i-p_j||^2.
            pair_coefficients = minimum_weights - maximum_weights
            gradient = np.zeros_like(points)
            pair_gradient = 2.0 * pair_coefficients[:, None] * delta
            np.add.at(gradient, pair_i, pair_gradient)
            np.add.at(gradient, pair_j, -pair_gradient)

            # Adam-style ascent is useful here because a few active contact
            # pairs can otherwise cause unstable updates late in continuation.
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            step = 0.026 - 0.018 * (iteration / 1799.0)
            points = normalize(
                points + step * first_moment / (np.sqrt(second_moment) + 1.0e-8)
            )

        ratio = exact_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END