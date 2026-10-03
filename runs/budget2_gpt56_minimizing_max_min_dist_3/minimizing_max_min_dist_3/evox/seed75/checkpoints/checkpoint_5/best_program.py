# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Numerically improve a 14-point diameter-one packing from a symmetric seed.

    Two points are fixed at opposite ends of a unit diameter.  SLSQP then
    maximizes a common lower bound on all squared pairwise distances while
    constraining every pairwise distance to remain at most one.
    """
    from scipy.optimize import minimize

    # The previous two-ring construction is an effective feasible starting
    # point.  Rotate and translate it so that its poles are (0,0,0) and
    # (1,0,0), fixing translation, rotation, and scale degeneracies.
    beta = 2.0 * (1.0 - np.cos(np.pi / 6.0))
    h2 = (1.0 - beta) / (5.0 - beta)
    h = np.sqrt(h2)
    r = np.sqrt(1.0 - h2)
    a = np.arange(6, dtype=float) * (np.pi / 3.0)
    seed = np.vstack((
        np.column_stack((r * np.cos(a), r * np.sin(a), np.full(6, h))),
        np.column_stack((r * np.cos(a + np.pi / 6.0),
                         r * np.sin(a + np.pi / 6.0), np.full(6, -h))),
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
    ))
    seed = np.vstack((
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        np.column_stack(((seed[:12, 2] + 1.0) / 2.0,
                         seed[:12, 0] / 2.0, seed[:12, 1] / 2.0)),
    ))

    ii, jj = np.triu_indices(14, 1)

    def unpack(x):
        return np.vstack(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                          x[:-1].reshape(12, 3)))

    def distances2(x):
        p = unpack(x)
        d = p[ii] - p[jj]
        return np.einsum("ij,ij->i", d, d)

    def constraints(x):
        d2 = distances2(x)
        t = x[-1]
        return np.concatenate((d2 - t, 1.0 - d2))

    d0 = distances2(np.r_[seed[2:].ravel(), 0.0])
    x0 = np.r_[seed[2:].ravel(), d0.min()]
    best = seed.copy()
    best_ratio = d0.min() / d0.max()

    # Small deterministic perturbations allow escape from the highly
    # symmetric local optimum while retaining reproducible output.
    rng = np.random.default_rng(140314)
    starts = [x0]
    for _ in range(4):
        trial = x0.copy()
        trial[:-1] += rng.normal(scale=0.018, size=36)
        starts.append(trial)

    for start in starts:
        result = minimize(
            lambda x: -x[-1], start, method="SLSQP",
            bounds=[(None, None)] * 36 + [(0.0, 1.0)],
            constraints={"type": "ineq", "fun": constraints},
            options={"maxiter": 600, "ftol": 1e-11, "disp": False},
        )
        candidate = unpack(result.x)
        delta = candidate[ii] - candidate[jj]
        d2 = np.einsum("ij,ij->i", delta, delta)
        ratio = d2.min() / d2.max()
        if np.isfinite(ratio) and ratio > best_ratio:
            best_ratio = ratio
            best = candidate

    return best.astype(float)


# EVOLVE-BLOCK-END
