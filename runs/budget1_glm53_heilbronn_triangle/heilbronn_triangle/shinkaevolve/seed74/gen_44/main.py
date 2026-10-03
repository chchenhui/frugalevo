# EVOLVE-BLOCK-START
import numpy as np


def _min_triangle_area(points: np.ndarray) -> float:
    """Smallest (normalized) triangle area among all triplets."""
    n = len(points)
    best = np.inf
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            ax, ay = points[i]
            bx, by = points[j]
            dxc = points[j + 1:, 0] - ax
            dyc = points[j + 1:, 1] - ay
            areas = 0.5 * np.abs((bx - ax) * dyc - (by - ay) * dxc)
            m = areas.min()
            if m < best:
                best = m
    return best


def _project_into_triangle(pts: np.ndarray) -> np.ndarray:
    """Project each point onto (or into) the unit equilateral triangle."""
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3) / 2])
    out = np.empty_like(pts)
    # barycentric coordinates: x = a*A + b*B + c*C, a+b+c=1
    for k, (x, y) in enumerate(pts):
        b = y / np.sqrt(3)
        c = x - 0.5 + 0.5 * y / np.sqrt(3)
        a = 1.0 - b - c
        if a < 0:
            # outside edge BC: clamp onto that edge
            t = (x - 0.5) * 2 + y * 2 / np.sqrt(3)
            t = min(max(t, 0.0), 2.0)
            px = 0.5 + 0.5 * t
            py = np.sqrt(3) / 2 * t
            # actually point on segment BC parameterized by t in [0,2]? use closest-point projection:
            out[k] = _closest_on_segment([x, y], B, C)
        elif b < 0:
            out[k] = _closest_on_segment([x, y], A, C)
        elif c < 0:
            out[k] = _closest_on_segment([x, y], A, B)
        else:
            out[k] = [x, y]
    return out


def _closest_on_segment(p, s1, s2):
    p = np.asarray(p, dtype=float)
    d = s2 - s1
    t = np.dot(p - s1, d) / np.dot(d, d)
    t = min(max(t, 0.0), 1.0)
    return s1 + t * d


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside the unit equilateral triangle
    maximizing the minimum triangle area (Heilbronn problem).

    Uses a fixed-seed simulated annealing search anchored at the three vertices,
    with result caching and a deterministic symmetric fallback.
    """
    n = 11
    if getattr(heilbronn_triangle11, "_cache", None) is not None:
        return heilbronn_triangle11._cache

    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    # Deterministic fallback: symmetric ring-like configuration
    fallback = np.vstack([
        verts,
        [[0.25, 0.0], [0.75, 0.0], [0.5, np.sqrt(3) / 2 / 3],
         [0.125, np.sqrt(3) / 8], [0.875, np.sqrt(3) / 8],
         [0.375, np.sqrt(3) / 4], [0.625, np.sqrt(3) / 4],
         [0.5, np.sqrt(3) / 6]]
    ])

    def _polish(pts, rng):
        """Targeted local search with single-point and correlated pair moves."""
        dirs = np.array([[np.cos(a), np.sin(a)]
                         for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)])
        steps = [0.05, 0.02, 0.008, 0.003, 0.001]
        cur = pts.copy()
        cur_val = _min_triangle_area(cur)
        improved = True
        while improved:
            improved = False
            # find bottleneck triangle vertices
            bot, bval = None, cur_val
            n_ = len(cur)
            for i in range(n_ - 2):
                for j in range(i + 1, n_ - 1):
                    ax, ay = cur[i]
                    bx, by = cur[j]
                    dxc = cur[j + 1:, 0] - ax
                    dyc = cur[j + 1:, 1] - ay
                    areas = 0.5 * np.abs((bx - ax) * dyc - (by - ay) * dxc)
                    k = int(np.argmin(areas))
                    if areas[k] < bval:
                        bval = areas[k]
                        bot = (i, j, j + 1 + k)
            if bot is None:
                break
            i0, j0, k0 = bot
            # single-point moves for each bottleneck vertex
            cands = [(p, d * s, None) for p in (i0, j0, k0)
                     for d in dirs for s in steps]
            # pair moves: same and opposite directions for bottleneck pairs
            pairs = [(i0, j0), (i0, k0), (j0, k0)]
            for (p, q) in pairs:
                for d in dirs:
                    for s in steps:
                        cands.append((p, d * s, (q, d * s)))       # same dir
                        cands.append((p, d * s, (q, -d * s)))      # opposite dir
            for p, dvec, pm in cands:
                if p < 3:
                    continue
                cand = cur.copy()
                cand[p] += dvec
                if pm is not None:
                    q, d2 = pm
                    if q >= 3:
                        cand[q] += d2
                cand = _project_into_triangle(cand)
                v = _min_triangle_area(cand)
                if v > cur_val + 1e-15:
                    cur, cur_val = cand, v
                    improved = True
                    break
        return cur, cur_val

    try:
        rng = np.random.default_rng(12345)
        # start from perturbed symmetric layout
        pts = fallback + rng.normal(0, 0.05, fallback.shape)
        pts[:3] = verts
        pts = _project_into_triangle(pts)
        best = pts.copy()
        best_val = _min_triangle_area(pts)
        cur_val = best_val
        T0, T1 = 0.02, 1e-5
        iters = 6000
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            cand = pts.copy()
            i = rng.integers(3, n)  # keep vertices fixed
            step = rng.normal(0, T, 2)
            cand[i] += step
            cand = _project_into_triangle(cand)
            v = _min_triangle_area(cand)
            if v >= cur_val or rng.random() < np.exp((v - cur_val) / max(T, 1e-12)):
                pts, cur_val = cand, v
                if v > best_val:
                    best, best_val = cand.copy(), v
        # targeted polish with correlated pair moves on the best found
        p1, v1 = _polish(best, rng)
        if v1 > best_val:
            best, best_val = p1, v1
        result = best if best_val > _min_triangle_area(fallback) else fallback
    except Exception:
        result = fallback

    heilbronn_triangle11._cache = result
    return result


# EVOLVE-BLOCK-END