# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Build fourteen points in R^3 maximizing minimum squared distance divided
    by maximum squared distance.  The returned configuration is centered and
    finite; its particular scale is immaterial to the objective.
    """
    n, dim = 14, 3
    iu, ju = np.triu_indices(n, 1)
    rng = np.random.default_rng(918273645)

    def dsq(p: np.ndarray) -> np.ndarray:
        z = p[iu] - p[ju]
        return np.einsum("ij,ij->i", z, z)

    def normalize(p: np.ndarray) -> np.ndarray:
        p = p - p.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(p * p))
        if scale > 1.0e-15:
            p = p / scale
        return p

    def ratio(p: np.ndarray) -> float:
        q = dsq(p)
        return float(q.min() / q.max())

    candidates = []

    # Search for several contact graphs.  Soft minimum and maximum log
    # distances remove the scale direction while retaining broad forces early.
    for restart in range(10):
        p = normalize(rng.normal(size=(n, dim)))
        m = np.zeros_like(p)
        v = np.zeros_like(p)

        for step in range(4600):
            frac = step / 4599.0
            beta = 8.0 + 520.0 * frac * frac

            delta = p[iu] - p[ju]
            q = np.einsum("ij,ij->i", delta, delta)
            ld = 0.5 * np.log(q + 1.0e-16)

            lo = -beta * ld
            lo -= lo.max()
            wm = np.exp(lo)
            wm /= wm.sum()

            hi = beta * ld
            hi -= hi.max()
            wx = np.exp(hi)
            wx /= wx.sum()

            # Gradient of softmin(log d)-softmax(log d).
            pw = (wm - wx)[:, None] * delta / (q[:, None] + 1.0e-16)
            g = np.zeros_like(p)
            np.add.at(g, iu, pw)
            np.add.at(g, ju, -pw)

            m = 0.89 * m + 0.11 * g
            v = 0.995 * v + 0.005 * g * g
            lr = 0.026 * (0.12 + 0.88 * (1.0 - frac))
            p = normalize(p + lr * m / (np.sqrt(v) + 1.0e-9))

            # A few late tiny kicks help exchange nearly-degenerate contacts.
            if step > 3500 and step % 250 == 0:
                trial = normalize(p + rng.normal(size=p.shape) * 0.0012)
                if ratio(trial) > ratio(p):
                    p = trial

        candidates.append((ratio(p), p.copy()))

    candidates.sort(key=lambda z: z[0], reverse=True)
    best_ratio, best = candidates[0][0], candidates[0][1].copy()

    # Direct feasibility formulation: with shortest distance fixed at one,
    # minimizing D is exactly the desired max-min/diameter problem.
    try:
        from scipy.optimize import minimize

        def constrained_polish(seed: np.ndarray) -> np.ndarray:
            seed = normalize(seed)
            qseed = dsq(seed)
            seed = seed / np.sqrt(qseed.min())
            d0 = float(dsq(seed).max()) * (1.0 + 1.0e-11)
            x0 = np.concatenate((seed.ravel(), [d0]))
            pairs = len(iu)

            def objective(x):
                return x[-1]

            def objective_jac(x):
                out = np.zeros_like(x)
                out[-1] = 1.0
                return out

            def ineq(x):
                p = x[:-1].reshape(n, dim)
                q = dsq(p)
                return np.concatenate((q - 1.0, x[-1] - q))

            def ineq_jac(x):
                p = x[:-1].reshape(n, dim)
                delta = p[iu] - p[ju]
                out = np.zeros((2 * pairs, n * dim + 1))
                for k in range(pairs):
                    a, b = iu[k], ju[k]
                    gg = 2.0 * delta[k]
                    out[k, 3 * a:3 * a + 3] = gg
                    out[k, 3 * b:3 * b + 3] = -gg
                    out[pairs + k, 3 * a:3 * a + 3] = -gg
                    out[pairs + k, 3 * b:3 * b + 3] = gg
                    out[pairs + k, -1] = 1.0
                return out

            def centroid(x):
                return x[:-1].reshape(n, dim).sum(axis=0)

            def centroid_jac(x):
                out = np.zeros((dim, n * dim + 1))
                for k in range(n):
                    out[:, 3 * k:3 * k + 3] = np.eye(dim)
                return out

            result = minimize(
                objective,
                x0,
                jac=objective_jac,
                method="SLSQP",
                constraints=(
                    {"type": "ineq", "fun": ineq, "jac": ineq_jac},
                    {"type": "eq", "fun": centroid, "jac": centroid_jac},
                ),
                options={"maxiter": 1100, "ftol": 5.0e-14, "disp": False},
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                return normalize(result.x[:-1].reshape(n, dim))
            return seed

        polished = []
        for _, seed in candidates[:3]:
            p = constrained_polish(seed)
            polished.append(p)
            r = ratio(p)
            if r > best_ratio:
                best_ratio, best = r, p.copy()

        # Exact-objective accepted moves.  This deliberately does not
        # differentiate through min/max: every accepted state improves the
        # precise metric used by the evaluator.
        for p0 in polished + [best]:
            p = p0.copy()
            current = ratio(p)
            scale = 1.8e-3

            for block in range(15):
                accepted = 0
                for _ in range(90):
                    q = dsq(p)
                    qlo, qhi = q.min(), q.max()

                    # A sharp active-contact force supplies a useful tangent
                    # direction, while random motion permits contact exchange.
                    beta = 900.0
                    wlo = np.exp(-beta * (q - qlo))
                    whi = np.exp(-beta * (qhi - q))
                    wlo /= wlo.sum()
                    whi /= whi.sum()
                    delta = p[iu] - p[ju]
                    force = np.zeros_like(p)
                    pair = (wlo - whi)[:, None] * delta
                    np.add.at(force, iu, pair)
                    np.add.at(force, ju, -pair)

                    fnorm = np.sqrt(np.mean(force * force))
                    if fnorm > 1.0e-14:
                        force /= fnorm
                    noise = rng.normal(size=(n, dim))
                    trial = normalize(p + scale * (0.55 * force + noise))
                    value = ratio(trial)
                    if value > current + 2.0e-14:
                        p, current = trial, value
                        accepted += 1

                scale *= 1.18 if accepted else 0.52
                if scale < 2.0e-7:
                    break

            if current > best_ratio:
                best_ratio, best = current, p.copy()

    except Exception:
        pass

    return np.asarray(normalize(best), dtype=float)


# EVOLVE-BLOCK-END