# EVOLVE-BLOCK-START
import numpy as np


# ---------- Stage 1: Geometry kernel (barycentric <-> Cartesian, areas) ----------

class TriangleGeometry:
    """Affine mapping from the 2-simplex (u,v) into the equilateral triangle
    with vertices A(0,0), B(1,0), C(0.5, sqrt(3)/2)."""

    def __init__(self):
        self.sqrt3 = np.sqrt(3.0)

    def to_xy(self, b):
        b = np.asarray(b, dtype=float)
        u, v = b[:, 0], b[:, 1]
        return np.column_stack([u + 0.5 * v, 0.5 * self.sqrt3 * v])

    def triples(self, n):
        idx = np.array([(i, j, k) for i in range(n) for j in range(i + 1, n)
                        for k in range(j + 1, n)])
        return idx

    def areas(self, xy, tri):
        p, q, r = xy[tri[:, 0]], xy[tri[:, 1]], xy[tri[:, 2]]
        return 0.5 * np.abs((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                            - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    def project(self, b):
        """Project barycentric coords onto the simplex u>=0, v>=0, u+v<=1."""
        b = np.clip(b, 0.0, 1.0)
        s = b.sum(axis=1)
        over = s > 1.0
        if np.any(over):
            b[over] *= (1.0 - 1e-9) / s[over, None]
        return b


# ---------- Stage 2: Deterministic maximin hill-climbing refiner ----------

class MaximinRefiner:
    def __init__(self, geom, n, tri, seed=12345, iters=400, scales=(0.02, 0.005)):
        self.geom = geom
        self.n = n
        self.tri = tri
        self.rng = np.random.default_rng(seed)
        self.iters = iters
        self.scales = scales

    def score(self, b):
        a = self.geom.areas(self.geom.to_xy(b), self.tri)
        return a.min()

    def refine(self, b):
        b = self.geom.project(b.copy())
        best = self.score(b)
        for it in range(self.iters):
            improved = False
            for scale in self.scales:
                cand = b + self.rng.normal(0.0, scale, b.shape)
                cand = self.geom.project(cand)
                val = self.score(cand)
                if val > best + 1e-12:
                    b, best = cand, val
                    improved = True
                    break
            if not improved and it > self.iters // 2:
                break  # converged early
        return b, best


# ---------- Stage 3: Curated seeds (vertices + well-spread interior) ----------

def curated_seeds():
    seeds = []
    # Seed family 1: vertices + interior lattice-like placements
    seeds.append(np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.25, 0.25], [0.0, 0.5],
        [0.5, 0.5], [0.25, 0.0], [0.0, 0.25],
        [0.5, 0.25], [0.25, 0.5],
    ]))
    # Seed family 2: perturbed uniform interior
    rng = np.random.default_rng(7)
    s = rng.random((11, 2)) * 0.8 + 0.1
    s[:3] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    seeds.append(s)
    return seeds


# ---------- Orchestrator ----------

def heilbronn_triangle11() -> np.ndarray:
    """Construct an arrangement of 11 points on or inside the equilateral
    triangle maximizing the smallest triangle area. Returns (11,2) array."""
    n = 11
    geom = TriangleGeometry()
    tri = geom.triples(n)

    best_xy, best_val = None, -1.0
    try:
        for si, seed in enumerate(curated_seeds()):
            refiner = MaximinRefiner(geom, n, tri, seed=100 + si)
            b, val = refiner.refine(seed)
            if val > best_val:
                best_val = val
                best_xy = geom.to_xy(b)
    except Exception:
        best_xy = None

    if best_xy is None or not np.all(np.isfinite(best_xy)):
        # Static fallback: always feasible
        b = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.25, 0.25], [0.0, 0.5],
            [0.5, 0.5], [0.25, 0.0], [0.0, 0.25],
            [0.5, 0.25], [0.25, 0.5],
        ])
        best_xy = geom.to_xy(b)

    return np.ascontiguousarray(best_xy, dtype=float)


# EVOLVE-BLOCK-END