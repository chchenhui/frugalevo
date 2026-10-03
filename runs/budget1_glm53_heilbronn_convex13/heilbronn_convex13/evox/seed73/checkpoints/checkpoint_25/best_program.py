# EVOLVE-BLOCK-START
import numpy as np
import itertools

_N = 13
# Unit-area equilateral triangle: side s with sqrt(3)/4 * s^2 = 1
_S = 2.0 / 3.0 ** 0.25
_H = _S * np.sqrt(3.0) / 2.0
_TRI_VERTS = np.array(
    [[-_S / 2.0, -_H / 3.0],
     [_S / 2.0, -_H / 3.0],
     [0.0, 2.0 * _H / 3.0]]
)
_TRIP = np.array(list(itertools.combinations(range(_N), 3)))  # (286,3)
# For each point p: the (j,k) pairs of triples containing p (66 pairs)
_PAIRS = []
_NOMASK = []
for _p in range(_N):
    _js, _ks = [], []
    for _j in range(_N):
        for _k in range(_j + 1, _N):
            if _j != _p and _k != _p:
                _js.append(_j)
                _ks.append(_k)
    _PAIRS.append((np.array(_js), np.array(_ks)))
    _NOMASK.append(~np.any(_TRIP == _p, axis=1))  # triples NOT containing p

# Wide-to-fine step schedule for the local search
_RADII = (0.12, 0.06, 0.03, 0.015, 0.008, 0.004, 0.002,
          0.001, 0.0005, 0.0002)


def _full_areas(pts):
    """Areas of all 286 triples (vectorized)."""
    a = pts[_TRIP[:, 0]]
    b = pts[_TRIP[:, 1]]
    c = pts[_TRIP[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _project(cand):
    """Clip candidate positions into the unit-area equilateral triangle."""
    pts = cand.copy()
    for i in range(3):
        a = _TRI_VERTS[i]
        b = _TRI_VERTS[(i + 1) % 3]
        # inward normal for edge a->b (counter-clockwise triangle)
        e = b - a
        nrm = np.array([-e[1], e[0]])
        nrm = nrm / np.hypot(*nrm)
        d = np.dot(pts - a, nrm)
        bad = d < 0
        if np.any(bad):
            pts[bad] -= d[bad, None] * nrm[None, :]
    return pts


def _refine(pts, rng, batch=1024):
    """
    Per-point batched local search maximizing the exact minimum triangle
    area. Moving one point only changes the 66 triples containing it, so
    we evaluate `batch` candidate positions for that point in one
    vectorized (66 x batch) operation; the other 220 triples are read
    from a cached minimum. Shrinking step schedule, accept-if-better.
    """
    pts = pts.copy()
    A = _full_areas(pts)
    best = float(A.min())
    for radius in _RADII:
        improved = True
        while improved:
            improved = False
            for p in range(_N):
                other_min = float(A[_NOMASK[p]].min())
                js, ks = _PAIRS[p]
                cand = _project(pts[p] + rng.normal(0.0, radius, (batch, 2)))
                ax, ay = cand[:, 0][None, :], cand[:, 1][None, :]
                bx, by = pts[js, 0][:, None], pts[js, 1][:, None]
                cx, cy = pts[ks, 0][:, None], pts[ks, 1][:, None]
                areas = 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                cand_min = np.minimum(areas.min(axis=0), other_min)
                i = int(np.argmax(cand_min))
                if cand_min[i] > best + 1e-15:
                    pts[p] = cand[i]
                    A = _full_areas(pts)
                    best = float(cand_min[i])
                    improved = True
    return pts, best


def _perimeter_points(k, phase=0.0):
    """k points equally spaced by arc length on the triangle boundary
    (phase shifts the start by a fraction of one spacing)."""
    edges = [np.linalg.norm(_TRI_VERTS[(i + 1) % 3] - _TRI_VERTS[i]) for i in range(3)]
    total = sum(edges)
    ts = (np.arange(k) + phase) * total / k
    pts = []
    for t in ts:
        acc = 0.0
        for i in range(3):
            a, b = _TRI_VERTS[i], _TRI_VERTS[(i + 1) % 3]
            if t <= acc + edges[i]:
                u = (t - acc) / edges[i]
                pts.append(a + u * (b - a))
                break
            acc += edges[i]
    return np.array(pts)


def _random_in_triangle(rng, k):
    """Uniform random points inside the triangle (barycentric)."""
    u, v = rng.random(k), rng.random(k)
    su = np.sqrt(u)
    b1, b2 = 1.0 - su, su * (1.0 - v)
    b3 = su * v
    return (b1[:, None] * _TRI_VERTS[0] + b2[:, None] * _TRI_VERTS[1]
            + b3[:, None] * _TRI_VERTS[2])


def heilbronn_convex13() -> np.ndarray:
    """
    Multi-start + basin-hopping local search for the n=13 Heilbronn
    problem inside a unit-area equilateral triangle (convex region).
    Boundary (perimeter-distributed) configurations beat disk/gon
    arrangements here, so starts include equally spaced perimeter
    points (two phases), jittered perimeter points, mixed
    boundary/interior layouts (12+1, 10+3, 9+4), a shrunk regular
    13-gon, and seeded random interior points. Each start is refined
    by moving one point at a time (vectorized candidate batches; only
    the 66 affected triples recomputed) with a shrinking step
    schedule. The best result is then perturbed (deterministic kicks)
    and re-refined to escape local optima. Fully deterministic.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates.
    """
    starts = []
    # 1-2) equally spaced perimeter points, two phases
    starts.append(_perimeter_points(_N, 0.0))
    starts.append(_perimeter_points(_N, 0.5))
    # 3) jittered perimeter points (different basin)
    rngj = np.random.default_rng(5)
    starts.append(_project(_perimeter_points(_N) + rngj.normal(0.0, 0.05, (_N, 2))))
    # 4-6) mixed boundary/interior layouts
    centroid = _TRI_VERTS.mean(axis=0)
    starts.append(_project(np.vstack([_perimeter_points(12), centroid[None, :]])))
    starts.append(_project(np.vstack([_perimeter_points(10), _random_in_triangle(rngj, 3)])))
    starts.append(_project(np.vstack([_perimeter_points(9), _random_in_triangle(rngj, 4)])))
    # 7) shrunk regular 13-gon around the centroid
    r_in = _H / 3.0  # inradius of the unit-area triangle
    ang = 2.0 * np.pi * np.arange(_N) / _N
    gon = centroid[None, :] + 0.85 * r_in * np.column_stack([np.cos(ang), np.sin(ang)])
    starts.append(_project(gon))
    # 8-11) seeded random interior configurations
    for seed in (7, 21, 99, 2024):
        starts.append(_random_in_triangle(np.random.default_rng(seed), _N))

    best_pts, best_val = None, -1.0
    for s_idx, pts0 in enumerate(starts):
        pts, val = _refine(pts0, np.random.default_rng(1000 + s_idx))
        if val > best_val:
            best_pts, best_val = pts, val

    # Deterministic basin-hopping: kick the best configuration and re-refine.
    for kick in range(15):
        rng = np.random.default_rng(5000 + kick)
        amp = 0.01 + 0.004 * (kick % 5)
        cand = _project(best_pts + rng.normal(0.0, amp, (_N, 2)))
        pts, val = _refine(cand, rng)
        if val > best_val:
            best_pts, best_val = pts, val

    return best_pts


# EVOLVE-BLOCK-END
