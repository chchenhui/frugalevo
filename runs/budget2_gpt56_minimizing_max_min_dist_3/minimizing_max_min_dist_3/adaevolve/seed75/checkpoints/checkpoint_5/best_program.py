# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Return seven well-separated unit directions and their antipodes.

    The directions are the three coordinate axes and four cube body-diagonal
    lines.  Taking both signs gives 14 points with exact diameter 2.  The
    largest absolute inner product between distinct direction lines is
    1/sqrt(3), so the minimum squared distance is 2 - 2/sqrt(3).
    """
    a = 1.0 / np.sqrt(3.0)
    directions = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [a, a, a],
            [a, a, -a],
            [a, -a, a],
            [-a, a, a],
        ],
        dtype=float,
    )
    return np.vstack((directions, -directions))


# EVOLVE-BLOCK-END
