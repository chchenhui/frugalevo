# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct seven well-separated lines and return their antipodal endpoints.

    The line-packing formulation is especially useful here: antipodal
    endpoints make the diameter exactly two, while minimizing the largest
    absolute inner product maximizes the minimum endpoint separation.
    """
    rng = np.random.default_rng(1701)
    best_lines = None
    best_coherence = np.inf

    # Successively sharpen a smooth maximum of the 21 absolute correlations.
    # Several starts are inexpensive for seven directions and avoid dependence
    # on a particular local line-packing basin.
    for _ in range(12):
        lines = rng.normal(size=(7, 3))
        lines /= np.linalg.norm(lines, axis=1, keepdims=True)

        for beta, step in ((8.0, 0.030), (24.0, 0.018), (80.0, 0.007)):
            for _ in range(1100):
                gram = lines @ lines.T
                abs_gram = np.abs(gram)
                np.fill_diagonal(abs_gram, -np.inf)

                # Softmax weights over ordered off-diagonal pairs.  Subtracting
                # the maximum keeps the calculation stable at the final beta.
                shifted = beta * (abs_gram - np.max(abs_gram))
                weights = np.exp(shifted)
                np.fill_diagonal(weights, 0.0)
                weights /= weights.sum()

                signed_weights = weights * np.sign(gram)
                gradient = 2.0 * (signed_weights @ lines)

                # Project to each sphere tangent plane before renormalizing.
                gradient -= (gradient * lines).sum(axis=1, keepdims=True) * lines
                lines -= step * gradient
                lines /= np.linalg.norm(lines, axis=1, keepdims=True)

                coherence = np.max(np.abs((lines @ lines.T)[
                    ~np.eye(7, dtype=bool)
                ]))
                if coherence < best_coherence:
                    best_coherence = coherence
                    best_lines = lines.copy()

    return np.vstack((best_lines, -best_lines))


# EVOLVE-BLOCK-END