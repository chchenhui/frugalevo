# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations

_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, np.sqrt(3.0) / 2.0])
_H = np.sqrt(3.0) / 2.0


def _min_area(points, idx):
    """Vectorized minimum triangle area over all C(11,3)=165 triplets."""
    pa = points[idx[:, 0]]
    pb = points[idx[:, 1]]
    pc = points[idx[:, 2]]
    areas = 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) -
                         (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))
    return areas.min()


def _bary_to_xy(B):
    """Barycentric coords (n,3) -> Cartesian inside the equilateral triangle."""
    return B[:, 0:1] * _V0 + B[:, 1:2] * _V1 + B[:, 2:3] * _V2


def _all_areas(points, idx):
    """Vectorized areas of all C(11,3)=165 triplets."""
    pa = points[idx[:, 0]]
    pb = points[idx[:, 1]]
    pc = points[idx[:, 2]]
    return 0.5 * np.abs((pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) -
                        (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0]))


def _softmin(areas, t):
    """Smooth lower bound on the minimum area (log-sum-exp)."""
    a = -areas / t
    m = a.max()
    return -t * (m + np.log(np.exp(a - m).sum()))


def _fallback():
    """Deterministic valid configuration: vertices + evenly spaced edge points."""
    B = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.66, 0.34, 0.0], [0.33, 0.67, 0.0],
        [0.0, 0.66, 0.34], [0.0, 0.33, 0.67],
        [0.34, 0.0, 0.66], [0.67, 0.0, 0.33],
        [0.45, 0.27, 0.28], [0.25, 0.45, 0.30],
    ])
    return B


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Approach: deterministic multi-start local search in barycentric coordinates
    (guarantees containment in the equilateral triangle). For each start:
      Phase 1 optimizes a smooth soft-min objective (escapes plateaus where the
      min-area is limited by several triangles at once), Phase 2 polishes the
      exact minimum area. Moves are a mix of deterministic axis-aligned
      barycentric shifts, edge projections, and random Gaussian perturbations,
      with points belonging to near-minimal triangles perturbed preferentially.
    Falls back to a fixed valid configuration on any failure.
    """
    n = 11
    idx = np.array(list(combinations(range(n), 3)), dtype=int)

    rng = np.random.default_rng(20240517)

    # Diverse deterministic starting configurations (barycentric).
    starts = []
    starts.append(_fallback())
    # Hexagonal-ish interior lattice
    starts.append(np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5],
        [0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
        [0.6, 0.2, 0.2], [0.2, 0.6, 0.2],
    ]))
    # Random Dirichlet starts
    for _ in range(4):
        starts.append(rng.dirichlet(np.ones(3) * 1.5, size=n))

    best_pts = None
    best_val = -1.0

    def _project(cand):
        cand = np.clip(cand, 0.0, None)
        s = cand.sum()
        if s <= 1e-12:
            return None
        return cand / s

    def _search(cur, obj, step0, smin, rand_tries, use_soft):
        """Local search on objective obj(B)->float with shrinking step."""
        step = step0
        while step > smin:
            improved = False
            base = obj(cur)
            # Points in near-minimal triangles get extra attention.
            areas = _all_areas(_bary_to_xy(cur), idx)
            thr = areas.min() * 1.6 + 1e-12
            hot = np.zeros(n, dtype=bool)
            bad = idx[areas <= thr]
            hot[bad.ravel()] = True
            order = list(np.where(hot)[0]) + list(np.where(~hot)[0])
            for pi in order:
                cands = []
                # Deterministic axis moves in barycentric space.
                for ax in range(3):
                    for sgn in (-1.0, 1.0):
                        c = cur[pi].copy()
                        c[ax] += sgn * step
                        c = _project(c)
                        if c is not None:
                            cands.append(c)
                # Edge projections (points often optimal on the boundary).
                for ax in range(3):
                    c = cur[pi].copy()
                    c[ax] = 0.0
                    c = _project(c)
                    if c is not None:
                        cands.append(c)
                # Random moves, more for "hot" points.
                for _ in range(rand_tries if hot[pi] else rand_tries // 3):
                    c = _project(cur[pi] + rng.normal(size=3) * step)
                    if c is not None:
                        cands.append(c)
                for c in cands:
                    trial = cur.copy()
                    trial[pi] = c
                    v = obj(trial)
                    if v > base + 1e-15:
                        cur = trial
                        base = v
                        improved = True
            if not improved:
                step *= 0.5
        return cur

    try:
        for start in starts:
            cur = np.clip(start, 0.0, None)
            cur = cur / cur.sum(axis=1, keepdims=True)

            def soft_obj(B):
                a = _all_areas(_bary_to_xy(B), idx)
                t = max(a.min() * 0.3, 1e-7)
                return _softmin(a, t)

            # Phase 1: smooth soft-min objective.
            cur = _search(cur, soft_obj, 0.05, 1e-3, 12, True)
            # Phase 2: exact min-area polish.
            cur = _search(
                cur, lambda B: _all_areas(_bary_to_xy(B), idx).min(),
                0.01, 5e-5, 12, False)

            cur_val = _min_area(_bary_to_xy(cur), idx)
            if cur_val > best_val:
                best_val = cur_val
                best_pts = _bary_to_xy(cur)
    except Exception:
        best_pts = None

    if best_pts is None or not np.all(np.isfinite(best_pts)):
        best_pts = _bary_to_xy(_fallback())

    # Numerical safety clamp into the triangle.
    pts = np.asarray(best_pts, dtype=float)
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 1.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, _H)
    ymax = _H * (1.0 - np.abs(pts[:, 0] - 0.5))
    pts[:, 1] = np.minimum(pts[:, 1], ymax)
    return pts


# EVOLVE-BLOCK-END
