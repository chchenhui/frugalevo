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


def _all_areas(points, tri_i, tri_j, tri_k):
    """Vectorized areas of all C(n,3) triangles."""
    a = points[tri_i]
    b = points[tri_j]
    c = points[tri_k]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return 0.5 * np.abs(cross)


def _local_search(points, rng, tri_i, tri_j, tri_k, deadline, max_iters=5000000):
    """Simulated-annealing ascent maximizing the min triangle area.

    Perturbs one point per iteration with an adaptive step size. Non-improving
    moves are accepted with a decaying Boltzmann probability so the search can
    escape local optima; a best-so-far configuration is tracked separately and
    returned. The step shrinks when stuck and re-expands to keep exploring.
    This approach empirically outperformed bottleneck-targeted hill climbing
    (which gets trapped in shallow local optima).
    """
    n = points.shape[0]
    cur = points.copy()
    cur_val = _min_triangle_area(cur, None, tri_i, tri_j, tri_k)
    best = cur.copy()
    best_val = cur_val
    step = 0.05
    temp = 0.002  # in area units; ~small relative to typical min-area gains
    t0 = time.time()
    span = max(deadline - t0, 1e-9)
    fails = 0
    for _ in range(max_iters):
        now = time.time()
        if now > deadline:
            break
        # geometric temperature cooling with reheat near the end of budget
        frac = (now - t0) / span
        T = temp * (0.995 ** int(frac * 4000)) + 1e-9
        i = rng.integers(n)
        delta = rng.normal(size=2) * step
        cand = cur.copy()
        cand[i] = np.clip(cand[i] + delta, 0.0, 1.0)
        val = _min_triangle_area(cand, None, tri_i, tri_j, tri_k)
        d = val - cur_val
        if d >= 0 or rng.random() < np.exp(d / T):
            cur = cand
            cur_val = val
            if val > best_val:
                best = cand.copy()
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
    total_budget = 250.0

    # Structured initializations: known-good triangular layout, boundary ring,
    # and random restarts for basin diversity.
    inits = []
    # Triangular layout: 3 vertices + edge points + interior (Heilbronn-favored)
    tri_v = np.array([[0.02, 0.02], [0.98, 0.02], [0.5, 0.96]])
    edge = []
    for (p, q) in [(tri_v[0], tri_v[1]), (tri_v[1], tri_v[2]), (tri_v[2], tri_v[0])]:
        for t in np.linspace(0.25, 0.75, 2):
            edge.append(p + t * (q - p))
    interior = np.array([[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.4, 0.15],
                         [0.6, 0.15], [0.5, 0.65]])
    inits.append(np.vstack([tri_v, np.array(edge), interior]))
    # Regular-ish ring configuration
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([0.5 + 0.45 * np.cos(angles), 0.5 + 0.45 * np.sin(angles)], axis=1)
    inits.append(ring)
    # Random restarts
    for _ in range(12):
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
        deadline=start_time + 320.0,
    )
    if val > best_val:
        best_points = pts

    return best_points


# EVOLVE-BLOCK-END
