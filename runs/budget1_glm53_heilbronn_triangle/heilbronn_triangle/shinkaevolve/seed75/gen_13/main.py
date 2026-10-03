# EVOLVE-BLOCK-START
import numpy as np


def _min_area_and_argmin(points, idx_i, idx_j, idx_k):
    pi = points[idx_i]
    pj = points[idx_j]
    pk = points[idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross) / (np.sqrt(3) / 4.0)
    am = int(np.argmin(areas))
    return areas[am], am


def _to_bary(points, s):
    # barycentric wrt (0,0),(1,0),(0.5,s/2)
    x = points[:, 0]
    y = points[:, 1]
    c = x - y / s
    a = 2.0 * y / s
    b = 1.0 - a - c
    return np.stack([a, b, c], axis=1)


def _from_bary(bc, s):
    a, b, c = bc[:, 0], bc[:, 1], bc[:, 2]
    x = c + 0.5 * a
    y = a * s / 2.0
    return np.stack([x, y], axis=1)


def _clip_bary(bc):
    bc = np.clip(bc, 0.0, 1.0)
    t = bc.sum(axis=1, keepdims=True)
    return bc / t


def heilbronn_triangle11() -> np.ndarray:
    n = 11
    try:
        rng = np.random.default_rng(1234567)
        s = np.sqrt(3.0)

        ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing='ij')
        ii, jj, kk = ii.ravel(), jj.ravel(), kk.ravel()
        mask = (ii < jj) & (jj < kk)
        idx_i, idx_j, idx_k = ii[mask], jj[mask], kk[mask]

        def evaluate(bc):
            pts = _from_bary(bc, s)
            return _min_area_and_argmin(pts, idx_i, idx_j, idx_k)

        best_global_score = -1.0
        best_global_bc = None

        # Several deterministic warm starts: vertices + lattice variants
        starts = []
        # lattice of side 4 (15 pts) take first 11 offsets scaled
        for m in (3, 4, 5):
            lat = []
            for i in range(m + 1):
                for j in range(m + 1 - i):
                    k = m - i - j
                    lat.append([k / m, (m - i - k) / m, i / m])
            lat = np.array(lat[:11])
            if len(lat) < 11:
                lat = np.vstack([lat, np.full((11 - len(lat), 3), 1.0 / 3.0)])
            starts.append(_clip_bary(lat))
        # random starts
        for _ in range(3):
            r = rng.dirichlet(np.ones(3) * 1.5, size=n)
            starts.append(_clip_bary(r))

        for start in starts:
            bc = start.copy()
            cur, am = evaluate(bc)
            best_bc = bc.copy()
            best = cur

            # Simulated annealing with directed moves
            T0, Tend = 0.05, 1e-5
            iters = 900
            for it in range(iters):
                T = T0 * (Tend / T0) ** (it / iters)
                cand = bc.copy()
                if rng.random() < 0.7:
                    # directed: move one vertex of the worst triangle
                    w = am
                    choice = int(rng.integers(0, 3))
                    vi = idx_i[w] if choice == 0 else (idx_j[w] if choice == 1 else idx_k[w])
                    cand[vi] = _clip_bary(cand[vi] + rng.normal(0.0, T, size=3))
                else:
                    # random point move
                    vi = int(rng.integers(0, n))
                    cand[vi] = _clip_bary(cand[vi] + rng.normal(0.0, T, size=3))
                # dedupe guard
                pts_c = _from_bary(cand, s)
                pts_o = np.delete(pts_c, vi, axis=0)
                if np.min(np.linalg.norm(pts_o - pts_c[vi], axis=1)) < 1e-8:
                    continue
                val, w = evaluate(cand)
                if val >= cur - T * 1e-3 * rng.random() or val > cur:
                    bc = cand
                    cur, am = val, w
                    if val > best:
                        best = val
                        best_bc = bc.copy()

            # Final polish: greedy coordinate descent on the worst triplet
            bc = best_bc.copy()
            cur, am = evaluate(bc)
            step = 0.02
            while step > 5e-6:
                improved = False
                for _ in range(250):
                    cand = bc.copy()
                    w = am
                    choice = int(rng.integers(0, 3))
                    vi = idx_i[w] if choice == 0 else (idx_j[w] if choice == 1 else idx_k[w])
                    cand[vi] = _clip_bary(cand[vi] + rng.normal(0.0, step, size=3))
                    val, w2 = evaluate(cand)
                    if val > cur + 1e-14:
                        bc = cand
                        cur, am = val, w2
                        improved = True
                if not improved:
                    step *= 0.5

            if cur > best_global_score:
                best_global_score = cur
                best_global_bc = bc.copy()

        return _from_bary(best_global_bc, s)
    except Exception:
        # Fallback: triangular lattice
        s = np.sqrt(3.0)
        m = 4
        fb = []
        for i in range(m + 1):
            for j in range(m + 1 - i):
                k = m - i - j
                fb.append([(j + 0.5 * k) / m, (k * s / 2.0) / m])
        arr = np.zeros((11, 2))
        for idx in range(11):
            arr[idx] = fb[idx % len(fb)]
        return arr


# EVOLVE-BLOCK-END
