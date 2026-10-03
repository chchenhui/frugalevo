"""
Hexagonal grid initialization for circle packing.

A hexagonal (offset) grid provides a good starting arrangement
because it's the densest regular packing pattern. Even rows are
offset by half the spacing, which reduces wasted space.
"""

import numpy as np


def hexagonal_grid(n, margin=0.1):
    """
    Generate n points on a hexagonal grid inside [margin, 1-margin]^2.

    Args:
        n: number of points to generate
        margin: distance from edges to keep clear

    Returns:
        np.array of shape (n, 2) with (x, y) coordinates
    """
    usable = 1.0 - 2 * margin
    cols = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    rows = int(np.ceil(n / cols))

    dx = usable / max(cols - 1, 1)
    dy = usable / max(rows - 1, 1)

    points = []
    for row in range(rows):
        for col in range(cols):
            if len(points) >= n:
                break
            x = margin + col * dx
            if row % 2 == 1:
                x += dx / 2  # offset for hex pattern
            y = margin + row * dy
            x = np.clip(x, margin, 1 - margin)
            y = np.clip(y, margin, 1 - margin)
            points.append([x, y])

    return np.array(points[:n])
