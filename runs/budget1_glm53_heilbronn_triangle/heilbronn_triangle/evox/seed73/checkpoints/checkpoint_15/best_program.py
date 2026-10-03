# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

_S3 = np.sqrt(3.0) / 2.0
_IDX = np.array(list(itertools.combinations(range(11), 3)))
_CACHE = None


def _areas_all(P):
    """Areas of all C(11,3)=165 triplets for one configuration."""
    a = P[_IDX[:, 0]]
    b = P[_IDX[:, 1]]
    c = P[_IDX[:, 2]]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def _signed_area(P, i, j, k):
    """Signed area of triplet (i, j, k)."""
    a, b, c = P[i], P[j], P[k]
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1])
                  - (b[1] - a[1]) * (c[0] - a[0]))


def _grad_step(P, A, h, deadline):
    """Direct gradient ascent on the bottleneck triplet's area.

    Computes the analytic gradient of the (signed) bottleneck triangle area
    with respect to its three vertices, tries a small backtracking/advancing
    line search along the ascent direction (projected back into the
    triangle), and accepts any strict improvement of the global min area.
    Returns (P, A, success).
    """
    ar = _areas_all(P)
    t = int(np.argmin(ar))
    i, j, k = _IDX[t]
    S = _signed_area(P, i, j, k)
    if abs(S) < 1e-15:
        return P, A, False
    sgn = 1.0 if S > 0 else -1.0
    a, b, c = P[i], P[j], P[k]
    # gradients of signed area wrt each vertex (times sgn to ascend |S|)
    ga = sgn * 0.5 * np.array([b[1] - c[1], c[0] - b[0]])
    gb = sgn * 0.5 * np.array([c[1] - a[1], a[0] - c[0]])
    gc = sgn * 0.5 * np.array([a[1] - b[1], b[0] - a[0]])
    for hh in (h, 2 * h, 4 * h, 0.5 * h):
        Q = P.copy()
        Q[i] += hh * ga
        Q[j] += hh * gb
        Q[k] += hh * gc
        Q = _project_inside(Q)
        A2 = float(_areas_all(Q).min())
        if A2 > A + 1e-15:
            return Q, A2, True
    return P, A, False


def _areas_batch(PB):
    """Vectorized min-area over a batch of B configurations (B,11,2)."""
    a = PB[:, _IDX[:, 0], :]
    b = PB[:, _IDX[:, 1], :]
    c = PB[:, _IDX[:, 2], :]
    ar = 0.5 * np.abs((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                      - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    return ar.min(axis=1)


def _project_inside(P):
    """Project points back into the unit equilateral triangle."""
    Q = np.array(P, dtype=float, copy=True)
    Q[..., 0] = np.clip(Q[..., 0], 0.0, 1.0)
    Q[..., 1] = np.clip(Q[..., 1], 0.0,
                        np.sqrt(3) * np.minimum(Q[..., 0], 1.0 - Q[..., 0]))
    return Q


def _initial_configs():
    """Deterministic structured seeds: lattices, boundary ring, jittered."""
    s = _S3
    cfgs = []
    # Triangular lattice rows (rows of 1,2,3,... points), several densities
    for denom in (4, 5, 6):
        P = []
        for r in range(denom + 1):
            y = s * r / denom
            m = r + 1
            for j in range(m):
                x = 0.5 + (j - (m - 1) / 2.0) / denom
                P.append((x, y))
        cfgs.append(_project_inside(np.array(P[:11])))
    # Boundary ring: 11 points arclength-uniform around perimeter
    ts = np.linspace(0.0, 3.0, 12)[:-1]
    pts = []
    for t in ts:
        if t < 1.0:
            pts.append((t, 0.0))
        elif t < 2.0:
            u = t - 1.0
            pts.append((1.0 - 0.5 * u, s * u))
        else:
            u = t - 2.0
            pts.append((0.5 * (1.0 - u), s * (1.0 - u)))
    cfgs.append(np.array(pts))
    # Jittered random, fixed seed (more restarts for wider exploration)
    rng = np.random.default_rng(12345)
    for _ in range(8):
        x = rng.random(11)
        y = rng.random(11) * s
        cfgs.append(_project_inside(np.stack([x, y], axis=1)))
    # Slightly-perturbed lattice seeds (good basins near structured optima)
    rng2 = np.random.default_rng(999)
    for base in list(cfgs[:3]):
        for _ in range(3):
            cfgs.append(_project_inside(base + rng2.normal(0.0, 0.02, base.shape)))
    return cfgs


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct 11 points inside the unit equilateral triangle maximizing the
    smallest triangle area.

    Approach: multi-start local search combining (a) exact gradient ascent
    on the bottleneck triplet's area (analytic gradient + line search over
    the three bottleneck vertices, projected into the triangle) and (b) a
    vectorized random-batch fallback that jitters the two worst triplets to
    escape constrained/degenerate local optima. Fully deterministic
    (fixed seeds), time-budgeted, result cached across calls.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    n = 11
    rng = np.random.default_rng(20240711)
    deadline = time.time() + 240.0

    best_P, best_A = None, -1.0

    for P0 in _initial_configs():
        if time.time() > deadline:
            break
        P = _project_inside(np.asarray(P0, dtype=float)[:n])
        A = _areas_all(P).min()
        h = 0.05          # gradient line-search scale
        step = 0.05       # random-batch scale
        stall = 0
        while time.time() < deadline:
            # Phase 1: gradient ascent on the bottleneck (cheap, directed)
            P, A, ok = _grad_step(P, A, h, deadline)
            if ok:
                h = min(h * 1.5, 0.2)
                stall = 0
                continue
            # Phase 2: random batch on two worst triplets
            ar = _areas_all(P)
            order = np.argsort(ar)
            tri = _IDX[int(order[0])]
            tri2 = _IDX[int(order[1])]
            B = 256
            cands = np.repeat(P[None, :, :], B, axis=0)
            noise = rng.normal(0.0, step, (B, 3, 2))
            cands[:, tri[0], :] += noise[:, 0, :]
            cands[:, tri[1], :] += noise[:, 1, :]
            cands[:, tri[2], :] += noise[:, 2, :]
            half = B // 2
            noise2 = rng.normal(0.0, step, (half, 3, 2))
            cands[:half, tri2[0], :] += noise2[:, 0, :]
            cands[:half, tri2[1], :] += noise2[:, 1, :]
            cands[:half, tri2[2], :] += noise2[:, 2, :]
            if stall > 3:  # occasional full perturbation to escape
                cands += rng.normal(0.0, step * 0.5, (B, n, 2))
            cands = _project_inside(cands.reshape(-1, 2)).reshape(B, n, 2)
            vals = _areas_batch(cands)
            j = int(np.argmax(vals))
            if vals[j] > A + 1e-15:
                P, A = cands[j], float(vals[j])
                stall = 0
            else:
                stall += 1
                step *= 0.7
                h *= 0.5
                if step < 1e-7 and h < 1e-9:
                    break                # converged for this start
                if stall > 6:
                    step = 0.02
                    h = 0.02
                    stall = 0
        if A > best_A:
            best_A, best_P = A, P

    _CACHE = best_P
    return best_P


# EVOLVE-BLOCK-END
