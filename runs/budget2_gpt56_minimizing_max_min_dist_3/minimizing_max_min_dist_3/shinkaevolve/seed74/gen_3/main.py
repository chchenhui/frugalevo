# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct a symmetric 14-point diameter-packing configuration in R^3.

    The points are the eight cube vertices and six coordinate-axis
    vertices.  The axis radius balances the two competing diameter
    types: opposite axis points and an axis point versus the opposite
    cube vertex.
    """
    a = (1.0 + np.sqrt(10.0)) / 3.0

    cube = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0,  1.0,  1.0],
        ],
        dtype=float,
    )
    axes = np.array(
        [
            [ a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0,  a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0,  a],
            [0.0, 0.0, -a],
        ],
        dtype=float,
    )

    return np.vstack((cube, axes))


# EVOLVE-BLOCK-END