# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of 11 points inside/on an equilateral triangle maximizing
    the minimum triangle area over all C(11,3)=165 triplets.

    Strategy: multi-start deterministic optimization with
      (1) global coordinated perturbation phase (move all points at once),
      (2) single-point pattern search (12 directions, shrinking steps),
    evaluated with the exact minimum triangle area.

    Returns:
        points: np.ndarray of shape (11,2)
    """
    n = 11
    H = np.sqrt(3) / 2.0
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])

    def clip_to_tri(P):
        # barycentric coordinates -> clip -> renormalize
        lam3 = P[:, 1] / H
        lam2 = P[:, 0] - 0.5 * lam3
        lam1 = 1.0 - lam2 - lam3
        L = np.stack([lam1, lam2, lam3], axis=1)
        L = np.clip(L, 0.0, 1.0)
        s = L.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        L = L / s
        return L @ V

    def min_area(P):
        d = P[:, 0][:, None] * P[:, 1][None, :] - P[:, 1][:, None] * P[:, 0][None, :]
        # area of triangle (i,j,k) = 0.5*|d_ij + d_jk + d_ki|
        best = np.inf
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                dij = d[i, j]
                for k in range(j + 1, n):
                    a = 0.5 * abs(dij + d[j, k] + d[k, i])
                    if a < best:
                        best = a
        return best

    def init_config(seed):
        rng = np.random.default_rng(seed)
        lam = rng.dirichlet(np.ones(3) * 0.55, size=n)
        P = lam @ V
        # jitter to break symmetry
        P = clip_to_tri(P + rng.normal(0, 0.02, P.shape))
        return P

    def refine(P, rng, rounds=3):
        """Global coordinated moves then single-point pattern search, repeated."""
        cur = P.copy()
        cur_a = min_area(cur)
        for _ in range(rounds):
            # ---- Phase A: global coordinated perturbations ----
            improved = True
            sigma = 0.015
            tries = 0
            while improved and tries < 400:
                improved = False
                tries += 1
                for _ in range(6):
                    cand = clip_to_tri(cur + rng.normal(0, sigma, cur.shape))
                    ca = min_area(cand)
                    if ca > cur_a + 1e-12:
                        cur, cur_a = cand, ca
                        improved = True
                if not improved and sigma > 0.002:
                    sigma *= 0.5
                    improved = True  # retry with smaller sigma
            # ---- Phase B: single-point pattern search ----
            step = 0.02
            dirs = []
            for ang in np.linspace(0, 2 * np.pi, 13)[:-1]:
                dirs.append((np.cos(ang), np.sin(ang)))
            dirs = np.array(dirs)
            while step > 1e-5:
                improved_any = False
                for i in range(n):
                    for dvec in dirs:
                        cand = cur.copy()
                        cand[i] += step * dvec
                        cand = clip_to_tri(cand)
                        ca = min_area(cand)
                        if ca > cur_a + 1e-12:
                            cur, cur_a = cand, ca
                            improved_any = True
                if not improved_any:
                    step *= 0.5
        return cur, cur_a

    try:
        best_P = None
        best_a = -1.0
        for seed in range(6):
            rng = np.random.default_rng(1000 + seed)
            P0 = init_config(7 * seed + 3)
            P, a = refine(P0, rng)
            if a > best_a:
                best_a = a
                best_P = P
        return best_P

    except Exception:
        # Fallback: deterministic feasible configuration
        lam = np.zeros((n, 3))
        lam[0] = [1.0, 0.0, 0.0]
        lam[1] = [0.0, 1.0, 0.0]
        lam[2] = [0.0, 0.0, 1.0]
        k = 3
        rng = np.random.default_rng(42)
        u = rng.random((n - 3, 3))
        u = u / u.sum(axis=1, keepdims=True)
        lam[3:] = u
        return lam @ V


# EVOLVE-BLOCK-END