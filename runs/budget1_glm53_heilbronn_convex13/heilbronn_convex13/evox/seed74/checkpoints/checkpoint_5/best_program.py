# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _min_triangle_area(points, tri_idx, tri_i, tri_j, tri_k):
    """Vectorized area of the smallest triangle among all C(n,3) triples."""
    a = points[tri_i]
    b = points[tri_j]
    c = points[tri_k]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _local_search(points, rng, tri_i, tri_j, tri_k, deadline, max_iters=200000):
    """Hill-climbing with adaptive step size maximizing the min triangle area."""
    n = points.shape[0]
    best = points.copy()
    best_val = _min_triangle_area(best, None, tri_i, tri_j, tri_k)
    step = 0.05
    fails = 0
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        i = rng.integers(n)
        delta = rng.normal(size=2) * step
        cand = best.copy()
        cand[i] = np.clip(cand[i] + delta, 0.0, 1.0)
        val = _min_triangle_area(cand, None, tri_i, tri_j, tri_k)
        if val >= best_val:
            best = cand
            best_val = val
            fails = 0
        else:
            fails += 1
            if fails > 200:  # shrink step when stuck, re-expand occasionally
                step *= 0.5
                fails = 0
                if step < 1e-6:
                    step = 0.02
    return best, best_val


def heilbronn_convex13() -> np.ndarray:
    """
    Construct 13 points in the unit square maximizing the smallest triangle area.

    Approach: deterministic multi-restart hill climbing. All C(13,3)=286 triangle
    areas are evaluated in a vectorized manner; a single random point is perturbed
    each iteration with an adaptive step size, accepting non-decreasing min-area
    moves. Restarts use a fixed seed for reproducibility. Points remain in [0,1]^2
    (a convex region), and the score is scale/normalization invariant.
    """
    n = 13
    idx = np.array(list(itertools.combinations(range(n), 3)), dtype=int)
    tri_i, tri_j, tri_k = idx[:, 0], idx[:, 1], idx[:, 2]

    rng = np.random.default_rng(seed=42)
    start_time = time.time()
    total_budget = 120.0

    # Structured initializations: boundary-ish rings plus random restarts
    inits = []
    # Regular-ish ring configuration
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)], axis=1)
    inits.append(ring)
    # Random restarts
    for _ in range(8):
        inits.append(rng.random((n, 2)))

    best_points = None
    best_val = -1.0
    restart = 0
    for init in inits:
        per_run = (total_budget - (time.time() - start_time)) / max(1, (len(inits) - restart))
        if per_run <= 0:
            break
        pts, val = _local_search(
            init.copy(), rng, tri_i, tri_j, tri_k,
            deadline=time.time() + per_run,
        )
        if val > best_val:
            best_val = val
            best_points = pts
        restart += 1

    if best_points is None:
        best_points = inits[0]

    # Final fine polish on the best configuration
    pts, val = _local_search(
        best_points.copy(), rng, tri_i, tri_j, tri_k,
        deadline=start_time + total_budget + 60.0,
    )
    if val > best_val:
        best_points = pts

    return best_points


# EVOLVE-BLOCK-END
