# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 points in R^3 maximizing the exact squared
    minimum-distance / diameter ratio.

    A smooth continuation discovers several candidate contact graphs, an
    explicit constrained diameter solve sharpens them, and an exact-ratio
    batched hill polish finishes on the evaluator's nonsmooth objective.
    """
    n, dim = 14, 3
    iu, ju = np.triu_indices(n, 1)
    pairs = len(iu)
    rng = np.random.default_rng(91374261)

    def pair_dsq(p):
        q = p[iu] - p[ju]
        return np.einsum("ij,ij->i", q, q)

    def normalize(p):
        p = np.asarray(p, dtype=float).copy()
        p -= p.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(p * p))
        if not np.isfinite(scale) or scale < 1.0e-14:
            return p
        return p / scale

    def exact_ratio(p):
        d2 = pair_dsq(p)
        return float(d2.min() / d2.max())

    def batched_exact_ratio(x):
        # x has shape (batch, n, 3)
        delta = x[:, iu, :] - x[:, ju, :]
        d2 = np.einsum("bki,bki->bk", delta, delta)
        return d2.min(axis=1) / d2.max(axis=1)

    candidates = []

    # Smooth log-distance continuation, solved in separate short phases.
    # This has scale-free gradients and does not require diameter projection
    # during each line-search evaluation.
    try:
        from scipy.optimize import minimize

        def smooth_value_and_gradient(x, beta):
            p = x.reshape(n, dim)
            delta = p[iu] - p[ju]
            d2 = np.einsum("ij,ij->i", delta, delta) + 1.0e-18
            ld = 0.5 * np.log(d2)

            a = -beta * ld
            a -= a.max()
            wnear = np.exp(a)
            wnear /= wnear.sum()

            b = beta * ld
            b -= b.max()
            wfar = np.exp(b)
            wfar /= wfar.sum()

            # Maximize softmin(log d) - softmax(log d).
            weight = wnear - wfar
            gpair = weight[:, None] * delta / d2[:, None]
            g = np.zeros((n, dim))
            np.add.at(g, iu, gpair)
            np.add.at(g, ju, -gpair)

            value = (
                -np.log(np.exp(a).sum()) / beta + ld.max()
                - (np.log(np.exp(b).sum()) / beta + ld.max())
            )
            # The algebraically equivalent expression above has harmless
            # constants omitted; only its gradient matters to L-BFGS.
            return -value, -g.ravel()

        for restart in range(10):
            p = normalize(rng.normal(size=(n, dim)))

            for beta, maxiter in (
                (7.0, 80),
                (22.0, 100),
                (70.0, 120),
                (210.0, 140),
                (620.0, 160),
            ):
                def objective(x, b=beta):
                    return smooth_value_and_gradient(x, b)

                result = minimize(
                    objective,
                    p.ravel(),
                    jac=True,
                    method="L-BFGS-B",
                    options={
                        "maxiter": maxiter,
                        "ftol": 1.0e-14,
                        "gtol": 2.0e-10,
                        "maxls": 28,
                    },
                )
                if result.x is not None and np.all(np.isfinite(result.x)):
                    p = normalize(result.x.reshape(n, dim))

            candidates.append((exact_ratio(p), p))

    except Exception:
        # Deterministic NumPy fallback if SciPy is unavailable.
        for restart in range(12):
            p = normalize(rng.normal(size=(n, dim)))
            for step in range(4200):
                beta = 10.0 + 550.0 * (step / 4199.0) ** 1.6
                delta = p[iu] - p[ju]
                d2 = np.einsum("ij,ij->i", delta, delta)
                z = -beta * d2
                z -= z.max()
                w = np.exp(z)
                w /= w.sum()
                force = (2.0 * beta * w)[:, None] * delta
                grad = np.zeros_like(p)
                np.add.at(grad, iu, force)
                np.add.at(grad, ju, -force)
                gn = np.sqrt(np.mean(grad * grad))
                if gn > 1.0e-14:
                    p = normalize(p + (0.020 * (1.0 - step / 5000.0)) * grad / gn)
            candidates.append((exact_ratio(p), p))

    candidates.sort(key=lambda z: z[0], reverse=True)
    best_ratio, best = candidates[0][0], candidates[0][1].copy()

    # Solve the exact diameter problem from several distinct continuation
    # basins.  Scaling makes the smallest separation equal to one.
    try:
        from scipy.optimize import minimize

        def polish_constraints(seed):
            seed = seed / np.sqrt(pair_dsq(seed).min())
            q0 = float(pair_dsq(seed).max()) * (1.0 + 1.0e-11)
            x0 = np.concatenate((seed.ravel(), [q0]))

            def fun(x):
                return x[-1]

            def jac_fun(x):
                g = np.zeros_like(x)
                g[-1] = 1.0
                return g

            def con(x):
                p = x[:-1].reshape(n, dim)
                d2 = pair_dsq(p)
                return np.concatenate((d2 - 1.0, x[-1] - d2))

            def jac_con(x):
                p = x[:-1].reshape(n, dim)
                delta = p[iu] - p[ju]
                out = np.zeros((2 * pairs, n * dim + 1))
                for k in range(pairs):
                    a, b = iu[k], ju[k]
                    g = 2.0 * delta[k]
                    out[k, 3 * a:3 * a + 3] = g
                    out[k, 3 * b:3 * b + 3] = -g
                    out[pairs + k, 3 * a:3 * a + 3] = -g
                    out[pairs + k, 3 * b:3 * b + 3] = g
                    out[pairs + k, -1] = 1.0
                return out

            def center(x):
                return x[:-1].reshape(n, dim).sum(axis=0)

            def jac_center(x):
                out = np.zeros((dim, n * dim + 1))
                for k in range(n):
                    out[:, 3 * k:3 * k + 3] = np.eye(dim)
                return out

            return minimize(
                fun,
                x0,
                jac=jac_fun,
                method="SLSQP",
                constraints=(
                    {"type": "ineq", "fun": con, "jac": jac_con},
                    {"type": "eq", "fun": center, "jac": jac_center},
                ),
                options={"maxiter": 750, "ftol": 8.0e-14, "disp": False},
            )

        refined = []
        for _, seed in candidates[:4]:
            result = polish_constraints(seed)
            if result.x is not None and np.all(np.isfinite(result.x)):
                p = normalize(result.x[:-1].reshape(n, dim))
                refined.append((exact_ratio(p), p))

        candidates.extend(refined)
        candidates.sort(key=lambda z: z[0], reverse=True)
        best_ratio, best = candidates[0][0], candidates[0][1].copy()

    except Exception:
        pass

    # Exact-objective basin polish.  Each round tests a population of tangent
    # perturbations and accepts only a strict improvement in dmin^2 / dmax^2.
    # This directly targets the evaluator metric and handles changing active
    # contacts without differentiating through min/max.
    for _, initial in candidates[:3]:
        p = normalize(initial)
        score = exact_ratio(p)
        sigma = 0.020
        batch = 56

        for _ in range(300):
            noise = rng.normal(size=(batch, n, dim))
            noise -= noise.mean(axis=1, keepdims=True)

            # Remove the radial scale direction before re-normalization.
            radial = np.sum(noise * p[None, :, :], axis=(1, 2))
            denom = np.sum(p * p)
            noise -= (radial / denom)[:, None, None] * p[None, :, :]

            trial = p[None, :, :] + sigma * noise
            trial -= trial.mean(axis=1, keepdims=True)
            rms = np.sqrt(np.mean(trial * trial, axis=(1, 2)))
            trial /= rms[:, None, None]

            values = batched_exact_ratio(trial)
            k = int(np.argmax(values))
            if values[k] > score + 1.0e-15:
                p = trial[k].copy()
                score = float(values[k])
                sigma = min(0.030, sigma * 1.13)
            else:
                sigma *= 0.72
                if sigma < 2.0e-8:
                    break

        if score > best_ratio:
            best_ratio, best = score, p.copy()

    return np.asarray(normalize(best), dtype=float)


# EVOLVE-BLOCK-END