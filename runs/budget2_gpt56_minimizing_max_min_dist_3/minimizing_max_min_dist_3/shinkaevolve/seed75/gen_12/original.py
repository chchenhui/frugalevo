# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct a symmetric cube--octahedron compound with 14 vertices.

    The cube contributes eight vertices and the coordinate axes contribute
    six further vertices.  Choosing the axial radius equal to the cube
    circumradius gives a well-separated, diameter-controlled configuration.
    """
    cube = np.array(
        [[x, y, z]
         for x in (-1.0, 1.0)
         for y in (-1.0, 1.0)
         for z in (-1.0, 1.0)],
        dtype=float,
    )

    radius = np.sqrt(3.0)
    axial = radius * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )

    return np.vstack((cube, axial))


# EVOLVE-BLOCK-END