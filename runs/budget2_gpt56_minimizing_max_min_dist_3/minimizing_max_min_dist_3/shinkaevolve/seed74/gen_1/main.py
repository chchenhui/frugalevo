# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct a 14 point antipodal spherical code in three dimensions.

    Seven optimized lines are represented by unit vectors, and both
    orientations of every line are returned.  Consequently the diameter is
    exactly 2, while the optimization reduces the largest absolute inner
    product, which is precisely the quantity controlling the shortest
    distance in the antipodal set.
    """
    rng = np.random.default_rng(314159)
    n_lines = 7
    best_vectors = None
    best_coherence = np.inf

    for restart in range(16):
        vectors = rng.normal(size=(n_lines, 3))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        # A soft maximum is smooth enough for optimization.  Increasing its
        # sharpness concentrates the final iterations on the limiting pairs.
        for iteration in range(1800):
            dots = vectors @ vectors.T
            squared_dots = dots * dots
            np.fill_diagonal(squared_dots, -np.inf)

            beta = 6.0 + 74.0 * iteration / 1799.0
            shifted = beta * (squared_dots - np.max(squared_dots))
            weights = np.exp(shifted)
            np.fill_diagonal(weights, 0.0)
            weights /= np.sum(weights)

            # Gradient of the weighted high-inner-product penalty, followed
            # by projection to the tangent plane of each unit sphere.
            gradient = 2.0 * ((weights * dots) @ vectors)
            gradient -= np.sum(gradient * vectors, axis=1, keepdims=True) * vectors

            step = 0.18 * (1.0 - 0.65 * iteration / 1799.0)
            vectors -= step * gradient
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        coherence_matrix = np.abs(vectors @ vectors.T)
        np.fill_diagonal(coherence_matrix, 0.0)
        coherence = np.max(coherence_matrix)
        if coherence < best_coherence:
            best_coherence = coherence
            best_vectors = vectors.copy()

    return np.vstack((best_vectors, -best_vectors))


# EVOLVE-BLOCK-END