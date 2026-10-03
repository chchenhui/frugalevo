# EVOLVE-BLOCK-START
import numpy as np

N = 13


# ---------- Stage 1: fast kernel ----------

def _min_triple_area(points: np.ndarray) -> float:
    n = len(points)
    best = np.inf
    for i in range(n - 2):
        v1x = points[i + 1, 0] - points[i, 0]
        v1y = points[i + 1, 1] - points[i, 1]
        dx = points[i + 2:, 0] - points[i, 0]
        dy = points[i + 2:, 1] - points[i, 1]
        areas = 0.5 * np.abs(v1x * dy - v1y * dx)
        m = areas.min()
        if m < best:
            best = m
    return float(best)


def _hull_area(points: np.ndarray) -> float:
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _score(points: np.ndarray) -> float:
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    return _min_triple_area(points) / ha


# ---------- Stage 2: precomputed triple index table ----------

def _triples():
    idxs = []
    for i in range(N - 2):
        for j in range(i + 1, N - 1):
            for k in range(j + 1, N):
                idxs.append((i, j, k))
    return np.array(idxs)


TRIP = _triples()
I, J, K = TRIP[:, 0], TRIP[:, 1], TRIP[:, 2]


# ---------- Stage 3: candidate generation (wide funnel mouth) ----------

def _rot3_expand(base4, center=0.5):
    pts = [np.array([center, center])]
    for b in base4:
        for r in range(3):
            th = 2 * np.pi * r / 3
            c, s = np.cos(th), np.sin(th)
            dx, dy = b[0] - center, b[1] - center
            pts.append(np.array([center + c * dx - s * dy,
                                 center + s * dx + c * dy]))
    return np.array(pts)


def _gen_candidates(num_random_seeds=72):
    cands = []
    # random 3-fold symmetric bases (wide multi-start)
    for sd in range(num_random_seeds):
        rng = np.random.default_rng(1000 + sd)
        base = rng.random((4, 2)) * 0.8 + 0.1
        cands.append(_rot3_expand(base))
    # structured inits
    for radius in (0.15, 0.25, 0.35, 0.45):
        # circle + inner ring + center
        P = np.zeros((N, 2))
        for m in range(12):
            th = 2 * np.pi * m / 12
            rr = radius if m % 2 == 0 else 0.5 * radius + 0.25
            ang = th * (2.0 / 3.0)  # 3-fold twist
            P[m] = [0.5 + 0.45 * np.cos(th), 0.5 + 0.45 * np.sin(th)]
        P[12] = [0.5 + radius - 0.25, 0.5]
        cands.append(P)
    # 12-gon + center variants
    for rot in range(3):
        ang = 2 * np.pi * (np.arange(12) + rot * 0.25) / 12.0
        P = np.zeros((N, 2))
        P[:12, 0] = 0.5 + 0.45 * np.cos(ang)
        P[:12, 1] = 0.5 + 0.45 * np.sin(ang)
        P[12] = [0.5, 0.5]
        cands.append(P)
    return cands


# ---------- Stage 4: softmin gradient ascent (vectorized) ----------

def _grad_ascent(P, iters=300, alpha=8.0, step=0.004):
    best_P = P.copy()
    best_val = _score(P)
    for it in range(iters):
        ax, ay = P[I, 0], P[I, 1]
        bx, by = P[J, 0], P[J, 1]
        cx, cy = P[K, 0], P[K, 1]
        A = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        absA = np.abs(A)
        w = np.exp(-alpha * (absA - absA.min()))
        w /= w.sum()
        sgn = np.sign(A)
        # gradients of |signed area| w.r.t. each vertex
        gi = np.stack([-(by - cy), (bx - cx)], axis=1)
        gj = np.stack([(cy - ay), -(cx - ax)], axis=1)
        gk = np.stack([(ay - by), -(ax - bx)], axis=1)
        G = np.zeros_like(P)
        np.add.at(G, I, 0.5 * (w * sgn)[:, None] * gi)
        np.add.at(G, J, 0.5 * (w * sgn)[:, None] * gj)
        np.add.at(G, K, 0.5 * (w * sgn)[:, None] * gk)
        # barrier inside [eps,1-eps]^2
        eps = 0.02
        G[:, 0] += 1e-5 * (-1.0 / np.maximum(P[:, 0] - eps, 1e-3)
                           + 1.0 / np.maximum(1 - eps - P[:, 0], 1e-3))
        G[:, 1] += 1e-5 * (-1.0 / np.maximum(P[:, 1] - eps, 1e-3)
                           + 1.0 / np.maximum(1 - eps - P[:, 1], 1e-3))
        norm = np.linalg.norm(G)
        if norm < 1e-12:
            break
        P = P + step * G / norm * np.sqrt(N)
        cur = _score(P)
        if cur > best_val:
            best_val = cur
            best_P = P.copy()
    return best_P, best_val


# ---------- Stage 5: exact hill-climb polish ----------

def _polish(P, max_iters=400):
    dirs = np.array([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best = _score(P)
    step = 0.02
    it = 0
    while step > 1e-5 and it < max_iters:
        it += 1
        improved = False
        for i in range(N):
            for d in dirs:
                for s in (step, 0.5 * step):
                    trial = P.copy()
                    trial[i] += s * d
                    sc = _score(trial)
                    if sc > best + 1e-12:
                        P = trial
                        best = sc
                        improved = True
        if not improved:
            step *= 0.5
    return P, best


# ---------- Orchestrator ----------

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points in a convex region maximizing
    the smallest triangle area (Heilbronn problem, n=13).

    Wide multi-start funnel: 72+ symmetric seeds -> short gradient ascent
    -> top-10 exact hill-climb polish. Deterministic.
    """
    cands = _gen_candidates(72)

    # Stage A: quick gradient pass on every candidate (cheap, vectorized)
    scored = []
    for c, P in enumerate(cands):
        Q, v = _grad_ascent(P.copy(), iters=120)
        scored.append((v, Q))
    scored.sort(key=lambda t: -t[0])

    # Stage B: extended gradient ascent on top-20
    finalists = []
    for v, Q in scored[:20]:
        Q2, v2 = _grad_ascent(Q.copy(), iters=300)
        finalists.append((max(v, v2), Q2 if v2 >= v else Q))
    finalists.sort(key=lambda t: -t[0])

    # Stage C: exact polish on top-10
    results = []
    for v, Q in finalists[:10]:
        R, vr = _polish(Q.copy(), max_iters=250)
        results.append((vr, R))
    results.sort(key=lambda t: -t[0])

    best_val, best_P = results[0]

    # Rescale to unit hull area (keeps configuration shape)
    ha = _hull_area(best_P)
    if ha > 1e-12:
        ctr = best_P.mean(axis=0)
        best_P = (best_P - ctr) / np.sqrt(ha) + ctr

    return best_P


# EVOLVE-BLOCK-END