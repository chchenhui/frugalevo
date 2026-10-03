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


def _repulsion_seed(seed, n=11, iters=400):
    """Vectorized inverse-square repulsion layout, vertices pinned."""
    rng = np.random.default_rng(seed)
    b = rng.random((n, 2)) * 0.6 + 0.05
    b[:3] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    for t in range(iters):
        d = b[:, None, :] - b[None, :, :]
        dist2 = (d ** 2).sum(axis=2) + 1e-6
        np.fill_diagonal(dist2, np.inf)
        f = (d / (dist2 ** 2)[:, :, None]).sum(axis=1)
        step = 0.02 * (0.995 ** t)
        b[3:] += step * f[3:]
        b = _project(b)
    return b


def _golden_seed(n=11):
    """Golden-ratio low-discrepancy seed, vertices pinned."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    pts = []
    for k in range(3, n):
        u = ((k * g) % 1.0) * 0.8 + 0.05
        v = ((k * g * g) % 1.0) * (0.9 - u) + 0.02
        pts.append([u, v])
    b = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]] + pts)
    return _project(b)


class _Annealer:
    def __init__(self, tri, seed):
        self.tri = tri
        self.n_pts = int(tri.max()) + 1
        self.rng = np.random.default_rng(seed)
        self.member = np.zeros((self.n_pts, len(tri)), dtype=bool)
        for c in range(3):
            self.member[tri[:, c], np.arange(len(tri))] = True

    def score(self, b):
        return float(_areas(_to_xy(b), self.tri).min())

    def bottleneck_counts(self, b, mult=1.5):
        a = _areas(_to_xy(b), self.tri)
        tight = a <= a.min() * mult + 1e-12
        return self.member[:, tight].sum(axis=1)

    def anneal(self, b, iters=3000, T0=0.015, T1=1e-4):
        b = _project(b.copy())
        cur = self.score(b)
        best, best_b = cur, b.copy()
        for it in range(iters):
            T = T0 * (T1 / T0) ** (it / iters)
            cand = b.copy()
            cnt = self.bottleneck_counts(b)
            if self.rng.random() < 0.25:
                top = np.argsort(-cnt)[:2]
                d = self.rng.normal(0.0, T * 3.0, 2)
                for pt in top:
                    cand[pt] += d
            else:
                k = int(self.rng.integers(1, 4))
                w = cnt.astype(float) + 1e-12
                chosen = []
                for _ in range(k):
                    if w.sum() <= 0:
                        break
                    idx = int(self.rng.choice(len(w), p=w / w.sum()))
                    chosen.append(idx)
                    w[idx] = 0.0
                for pt in chosen:
                    cand[pt] += self.rng.normal(0.0, T * 3.0, 2)
            if self.rng.random() < 0.15:
                cand += self.rng.normal(0.0, T * 0.4, cand.shape)
            cand = _project(cand)
            v = self.score(cand)
            if v >= cur or self.rng.random() < np.exp((v - cur) / max(T * 1e-3, 1e-15)):
                b, cur = cand, v
                if v > best:
                    best, best_b = v, b.copy()
        return best_b, best

    def grad_polish(self, b, rounds=200):
        """Gradient-ascent on binding triangles with adaptive line search."""
        best = self.score(b)
        b = _project(b.copy())
        step0 = 0.01
        for _ in range(rounds):
            improved = False
            xy = _to_xy(b)
            a = _areas(xy, self.tri)
            amin = a.min()
            tight_idx = np.where(a <= amin * 1.2 + 1e-14)[0]
            for sc in (step0, step0 * 0.3, step0 * 0.1):
                if improved:
                    break
                grad = np.zeros_like(b)
                for t in tight_idx:
                    i, j, k = self.tri[t]
                    p, q, r = xy[i], xy[j], xy[k]
                    cross = (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
                    s = np.sign(cross) if abs(cross) > 1e-18 else 1.0
                    # d(0.5*|cross|)/d(point) in xy coords, mapped to barycentric-ish (u,v)
                    # area = 0.5*s*cross ; gradients wrt p,q,r:
                    gp = np.array([-(r[1]-p[1])*s + (q[1]-p[1])*s,
                                   (r[0]-p[0])*s - (q[0]-p[0])*s]) * 0.5
                    gq = np.array([(r[1]-p[1])*s, -(r[0]-p[0])*s]) * 0.5
                    gr = np.array([-(q[1]-p[1])*s, (q[0]-p[0])*s]) * 0.5
                    # xy grad -> (u,v) grad: x = u+0.5v, y = 0.5*sqrt3*v
                    def xy2uv(g):
                        return np.array([g[0], 0.5 * g[0] + g[1] * (2.0 / _SQRT3)])
                    grad[i] += xy2uv(gp)
                    grad[j] += xy2uv(gq)
                    grad[k] += xy2uv(gr)
                gn = np.linalg.norm(grad, axis=1, keepdims=True)
                gn[gn == 0] = 1.0
                cand = _project(b + sc * grad / gn)
                v = self.score(cand)
                if v > best + 1e-14:
                    b, best = cand, v
                    improved = True
            if not improved:
                step0 *= 0.3
                if step0 < 1e-6:
                    break
            else:
                step0 = min(step0 * 1.3, 0.02)
        return b, best


def _seeds():
    s = []
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
    # jittered lattice variants (larger sigma to escape symmetry basins)
    rng = np.random.default_rng(2024)
    for base in list(s):
        j = base + rng.normal(0.0, 0.05, base.shape)
        j[:3] = base[:3]
        s.append(_project(j))
    # golden-ratio low-discrepancy seed
    s.append(_golden_seed())
    # repulsion seeds (deterministic)
    for rs in (11, 202, 303):
        s.append(_repulsion_seed(rs))
    # free-form random seeds
    for rs in (5, 6):
        r = np.random.default_rng(rs).random((11, 2))
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
        fb_val = float(_areas(_to_xy(fallback_b), tri).min())
        best_xy, best_val = None, -1.0
        for si, seed in enumerate(_seeds()):
            ann = _Annealer(tri, seed=777 + 13 * si)
            b, val = ann.anneal(seed)
            # anneal then gradient polish, then a short re-anneal restart
            b, val = ann.grad_polish(b)
            b2, val2 = ann.anneal(b, iters=800, T0=0.004, T1=1e-5)
            if val2 > val:
                b, val = b2, val2
            b, val = ann.grad_polish(b)
            if val > best_val:
                best_val, best_xy = val, _to_xy(b)
        if best_xy is None or not np.all(np.isfinite(best_xy)):
            raise RuntimeError("optimization failed")
        if best_val < fb_val:
            best_xy = _to_xy(fallback_b)
        return np.ascontiguousarray(best_xy, dtype=float)
    except Exception:
        return np.ascontiguousarray(_to_xy(fallback_b), dtype=float)


# EVOLVE-BLOCK-END