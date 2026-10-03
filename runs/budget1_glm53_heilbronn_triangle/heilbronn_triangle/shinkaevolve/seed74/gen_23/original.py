# EVOLVE-BLOCK-START
import numpy as np


_SQRT3 = float(np.sqrt(3.0))


def _to_xy(b):
    b = np.asarray(b, dtype=float)
    u, v = b[:, 0], b[:, 1]
    return np.column_stack([u + 0.5 * v, 0.5 * _SQRT3 * v])


def _project(b):
    b = np.clip(b, 0.0, 1.0)
    s = b.sum(axis=1)
    over = s > 1.0
    if np.any(over):
        b[over] *= (1.0 - 1e-9) / s[over, None]
    return b


def _make_triples(n):
    return np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                     for k in range(j + 1, n)])


def _areas(xy, tri):
    p = xy[tri[:, 0]]
    q = xy[tri[:, 1]]
    r = xy[tri[:, 2]]
    return 0.5 * np.abs((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                        - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))


class _Annealer:
    def __init__(self, tri, seed):
        self.tri = tri
        self.n_pts = tri.max() + 1
        self.rng = np.random.default_rng(seed)
        # precompute point->triples membership for bottleneck detection
        self.member = np.zeros((self.n_pts, len(tri)), dtype=bool)
        for c in range(3):
            self.member[tri[:, c], np.arange(len(tri))] = True

    def score(self, b):
        return float(_areas(_to_xy(b), self.tri).min())

    def bottleneck_points(self, b, m=3):
        a = _areas(_to_xy(b), self.tri)
        thresh = a.min() * 1.5 + 1e-12
        tight = a <= thresh
        cnt = self.member[:, tight].sum(axis=1)
        return np.argsort(-cnt)[:m]

    def anneal(self, b, iters=2500, T0=0.012, T1=1e-4):
        b = _project(b.copy())
        cur = self.score(b)
        best, best_b = cur, b.copy()
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            # bottleneck points get random-direction moves
            cand = b.copy()
            k = int(self.rng.integers(1, 4))  # move 1-3 points
            pts = self.bottleneck_points(b, m=k)
            for pt in pts:
                # random directions: isotropic gaussian (equivalent to any
                # random angle fan, but rotationally uniform in simplex coords)
                cand[pt] += self.rng.normal(0.0, T * 3.0, 2)
            # occasionally jitter every point slightly
            if self.rng.random() < 0.15:
                cand += self.rng.normal(0.0, T * 0.4, cand.shape)
            cand = _project(cand)
            v = self.score(cand)
            if v >= cur or self.rng.random() < np.exp((v - cur) / max(T * 1e-3, 1e-15)):
                b, cur = cand, v
                if v > best:
                    best, best_b = v, b.copy()
        return best_b, best


def _seeds():
    s = []
    # lattice-ish seeds (vertices pinned)
    s.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 0.0], [2/3, 0.0], [0.0, 1/3], [0.0, 2/3],
        [1/3, 1/3], [2/3, 1/3], [1/3, 2/3], [0.5, 1/6],
    ]))
    s.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.25, 0.25], [0.0, 0.5],
        [0.5, 0.5], [0.25, 0.0], [0.0, 0.25],
        [0.5, 0.25], [0.25, 0.5],
    ]))
    # jittered variants, deterministic
    rng = np.random.default_rng(2024)
    for base in list(s):
        j = base + rng.normal(0.0, 0.03, base.shape)
        j[:3] = base[:3]
        s.append(_project(j))
    # one free-form random seed
    r = rng.random((11, 2))
    r[:3] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    s.append(_project(r))
    return s


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    fallback_b = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [1/3, 0.0], [2/3, 0.0], [0.0, 1/3], [0.0, 2/3],
        [1/3, 1/3], [2/3, 1/3], [1/3, 2/3], [0.5, 1/6],
    ])
    try:
        tri = _make_triples(n)
        best_xy, best_val = None, -1.0
        for si, seed in enumerate(_seeds()):
            ann = _Annealer(tri, seed=777 + 13 * si)
            b, val = ann.anneal(seed)
            if val > best_val:
                best_val, best_xy = val, _to_xy(b)
        if best_xy is None or not np.all(np.isfinite(best_xy)):
            raise RuntimeError("optimization failed")
        fb_val = float(_areas(_to_xy(fallback_b), tri).min())
        if best_val < fb_val:
            best_xy = _to_xy(fallback_b)
        return np.ascontiguousarray(best_xy, dtype=float)
    except Exception:
        return np.ascontiguousarray(_to_xy(fallback_b), dtype=float)


# EVOLVE-BLOCK-END