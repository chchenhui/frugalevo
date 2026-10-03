# EVOLVE-BLOCK-START
import numpy as np
import time as _time

H = np.sqrt(3.0) / 2.0
N = 11
SEED = 20240517
TIME_BUDGET = 6.0  # seconds total for optimization


# ---------------------------------------------------------------
# Precomputed combinatorics
# ---------------------------------------------------------------

def _build_triplets():
    idx = np.array([(a, b, c) for a in range(N) for b in range(a + 1, N)
                    for c in range(b + 1, N)], dtype=np.int64)
    # membership lists: for each point, which triplets contain it
    members = [[] for _ in range(N)]
    for t, (a, b, c) in enumerate(idx):
        members[a].append(t)
        members[b].append(t)
        members[c].append(t)
    members = [np.array(m, dtype=np.int64) for m in members]
    return idx, members


_TRIP, _MEMBERS = _build_triplets()


# ---------------------------------------------------------------
# Geometry helpers (vectorized)
# ---------------------------------------------------------------

def _areas_of(pts, tri_ids):
    p = pts[_TRIP[tri_ids]]
    ax, ay = p[:, 0, 0], p[:, 0, 1]
    bx, by = p[:, 1, 0], p[:, 1, 1]
    cx, cy = p[:, 2, 0], p[:, 2, 1]
    return 0.5 * np.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def _all_areas(pts):
    return _areas_of(pts, np.arange(len(_TRIP)))


def _project(pts):
    x, y = pts[:, 0], pts[:, 1]
    w1 = x - y / np.sqrt(3.0)
    w2 = 2.0 * y / np.sqrt(3.0)
    w0 = 1.0 - w1 - w2
    w = np.stack([w0, w1, w2], axis=1)
    w = np.clip(w, 0.0, None)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w /= s
    out = np.empty_like(pts)
    out[:, 0] = w[:, 1] + 0.5 * w[:, 2]
    out[:, 1] = H * w[:, 2]
    return out


# ---------------------------------------------------------------
# Annealing engine with single / group / hinge moves
# ---------------------------------------------------------------

class _Annealer:
    def __init__(self, pts, rng, t_budget):
        self.rng = rng
        self.deadline = _time.time() + t_budget
        self.pts = _project(pts.copy())
        self.areas = _all_areas(self.pts)
        self.cur_val = float(self.areas.min())
        self.best = self.pts.copy()
        self.best_val = self.cur_val
        self.step = 0.010
        self.T = 0.0015

    def _apply_move(self, cand, group_ids, step):
        for i in group_ids:
            cand[i] += self.rng.normal(0.0, step, size=2)
        return _project(cand)

    def _try(self, group_ids, step):
        cand = self.pts.copy()
        cand = self._apply_move(cand, group_ids, step)
        # incremental area update for touched triplets
        touched = np.unique(np.concatenate([_MEMBERS[i] for i in group_ids]))
        new_areas = self.areas.copy()
        new_areas[touched] = _areas_of(cand, touched)
        v = float(new_areas.min())
        import math
        if v >= self.cur_val or self.rng.random() < math.exp((v - self.cur_val) / max(self.T, 1e-12)):
            self.pts, self.areas, self.cur_val = cand, new_areas, v
            if v > self.best_val:
                self.best, self.best_val = cand.copy(), v
            return True
        return False

    def run(self):
        it = 0
        kmin = int(np.argmin(self.areas))
        tri = _TRIP[kmin]
        while _time.time() < self.deadline:
            it += 1
            u = self.rng.random()
            if it % 200 == 0:
                # refresh smallest-triangle info periodically
                kmin = int(np.argmin(self.areas))
                tri = _TRIP[kmin]
            if u < 0.45:
                # single-point move
                i = int(self.rng.integers(N))
                self._try([i], self.step)
            elif u < 0.70:
                # group move: all vertices of the current smallest triangle
                self._try(list(tri), self.step)
            elif u < 0.85:
                # hinge move: union of vertices of the two smallest triangles
                order = np.argsort(self.areas)[:6]
                k2 = int(order[1])
                tri2 = _TRIP[k2]
                group = sorted(set(tri) | set(tri2))
                self._try(group, self.step)
            else:
                # nearby-neighbour correlated group move
                i = int(self.rng.integers(N))
                d2 = np.sum((self.pts - self.pts[i]) ** 2, axis=1)
                near = np.argsort(d2)[1:2]
                group = [i, int(near[0])]
                self._try(group, self.step)
            if it % 2500 == 0:
                self.step *= 0.85
                self.T *= 0.80
                if self.step < 1e-5:
                    break
                kmin = int(np.argmin(self.areas))
                tri = _TRIP[kmin]
        return self.best, self.best_val


# ---------------------------------------------------------------
# Deterministic hard-min polish (coordinate descent on true min)
# ---------------------------------------------------------------

def _hard_polish(pts, step, rounds):
    pts = _project(pts.copy())
    areas = _all_areas(pts)
    best = float(areas.min())
    for _ in range(rounds):
        improved = False
        kmin = int(np.argmin(areas))
        tri = list(_TRIP[kmin])
        order = tri + [i for i in range(N) if i not in tri]
        for i in order:
            for d in range(2):
                for s in (+1, -1):
                    cand = pts.copy()
                    cand[i, d] += s * step
                    cand = _project(cand)
                    touched = _MEMBERS[i]
                    na = _areas_of(cand, touched)
                    cand_areas = areas.copy()
                    cand_areas[touched] = na
                    v = float(cand_areas.min())
                    if v > best + 1e-15:
                        pts, areas, best = cand, cand_areas, v
                        improved = True
        if not improved:
            step *= 0.5
            if step < 1e-7:
                break
    return pts


# ---------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------

def _lattice_seed():
    pts = []
    rows = [4, 3, 2, 1, 1]
    nlev = len(rows)
    for r, cnt in enumerate(rows):
        y = H * (r + 0.5) / nlev
        halfw = 0.5 * (1.0 - y / H)
        xs = np.linspace(-0.5, 0.5, cnt + 2)[1:-1]
        for x in xs:
            pts.append((0.5 + x * halfw * 0.96, y))
    return _project(np.array(pts))


def _vertex_seed():
    h = H
    pts = np.array([
        (0.0, 0.0), (1.0, 0.0), (0.5, h),
        (0.4, 0.0), (0.5, h / 3), (0.1, 0.0),
        (0.9, 0.0), (0.25, h / 2), (0.75, h / 2),
        (0.5, 2 * h / 3), (0.5, h / 6),
    ], dtype=float)
    return _project(pts)


def _random_seed(rng):
    u, v = rng.random(N), rng.random(N)
    su = np.sqrt(u)
    w0, w1, w2 = 1 - su, su * (1 - v), su * v
    return np.stack([w1 + 0.5 * w2, H * w2], axis=1)


def _jittered_seed(rng, base, amp):
    return _project(base + rng.normal(0.0, amp, size=base.shape))


# ---------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------

def _optimize():
    rng = np.random.default_rng(SEED)
    t0 = _time.time()
    deadline = t0 + TIME_BUDGET

    lat = _lattice_seed()
    starts = [lat, _vertex_seed()]
    for _ in range(4):
        starts.append(_random_seed(rng))
    for _ in range(3):
        starts.append(_jittered_seed(rng, lat, 0.03))

    # quick polish each start cheaply, rank, then anneal the top ones
    scored = []
    for s in starts:
        c = _hard_polish(s, 0.004, 8)
        scored.append((float(_all_areas(c).min()), c))
    scored.sort(key=lambda t: -t[0])

    best_pts, best_val = scored[0][1], scored[0][0]
    anneal_count = 0
    for val0, s in scored[:5]:
        remaining = deadline - _time.time()
        if remaining < 0.4 or anneal_count >= 4:
            break
        budget = min(remaining * 0.6, 1.2)
        try:
            ann = _Annealer(s, rng, budget)
            cand, cv = ann.run()
            cand = _hard_polish(cand, 0.002, 40)
            v = float(_all_areas(cand).min())
            if v > best_val:
                best_pts, best_val = cand, v
        except Exception:
            continue
        anneal_count += 1

    # final long polish on the winner with whatever time remains
    remaining = deadline - _time.time()
    best_pts = _hard_polish(best_pts, 0.0015, 120)
    return _project(best_pts)


try:
    _POINTS = _optimize()
except Exception:
    try:
        _POINTS = _project(_lattice_seed())
    except Exception:
        _POINTS = np.array([[0.5 * (i % 4) / 3.0, H * (i // 4) / 3.0]
                            for i in range(N)])


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of exactly 11 points inside/on the equilateral
    triangle with vertices (0,0), (1,0), (0.5, sqrt(3)/2) maximizing the
    minimum triangle area over all point triplets (Heilbronn problem).

    Returns:
        points: np.ndarray of shape (11,2) with x,y coordinates.
    """
    return _POINTS.copy()
# EVOLVE-BLOCK-END