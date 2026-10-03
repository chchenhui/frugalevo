# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen reproducible points in R^3.

    The search is unrestricted: structured seeds merely diversify the
    contact graphs supplied to a smooth packing continuation.  Final scoring
    and final constrained polishing use the exact squared min-distance /
    squared diameter objective used by the evaluator.
    """
    n, d = 14, 3
    pi, pj = np.triu_indices(n, 1)
    m_pairs = pi.size
    rng = np.random.default_rng(20250314017)

    incidence = np.zeros((n, m_pairs), dtype=np.float64)
    columns = np.arange(m_pairs)
    incidence[pi, columns] = 1.0
    incidence[pj, columns] = -1.0

    def normalize_batch(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).copy()
        x -= x.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(x * x, axis=(1, 2)))
        scale = np.maximum(scale, 1.0e-15)
        return x / scale[:, None, None]

    def normalize(x: np.ndarray) -> np.ndarray:
        return normalize_batch(x[None, :, :])[0]

    def batched_scores(x: np.ndarray) -> np.ndarray:
        delta = x[:, pi] - x[:, pj]
        dsq = np.einsum("bpd,bpd->bp", delta, delta)
        return dsq.min(axis=1) / dsq.max(axis=1)

    def score(x: np.ndarray) -> float:
        delta = x[pi] - x[pj]
        dsq = np.einsum("pd,pd->p", delta, delta)
        return float(dsq.min() / dsq.max())

    def icosa_seed() -> np.ndarray:
        """Twelve icosahedral vertices plus two independently placed points."""
        phi = (1.0 + np.sqrt(5.0)) * 0.5
        verts = []
        for a in (-1.0, 1.0):
            for b in (-phi, phi):
                verts.append((0.0, a, b))
                verts.append((a, b, 0.0))
                verts.append((b, 0.0, a))
        base = np.asarray(verts, dtype=np.float64)
        base /= np.linalg.norm(base[0])

        u = rng.normal(size=(2, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        radii = np.array([0.72 + 0.28 * rng.random(), 0.72 + 0.28 * rng.random()])
        points = np.vstack((base, radii[:, None] * u))
        points += 0.030 * rng.normal(size=points.shape)
        return normalize(points)

    def ring_seed() -> np.ndarray:
        """Three staggered rings and two non-antipodal cap points."""
        points = []
        heights = (-0.48, 0.0, 0.48)
        counts = (4, 5, 3)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        for level, count in zip(heights, counts):
            radius = np.sqrt(max(0.12, 1.0 - level * level))
            angles = phase + rng.uniform(-0.22, 0.22) + 2.0 * np.pi * np.arange(count) / count
            for angle in angles:
                points.append((radius * np.cos(angle), radius * np.sin(angle), level))
            phase += np.pi / count

        caps = rng.normal(size=(2, 3))
        caps[:, 2] += np.array([1.35, -1.35])
        caps /= np.linalg.norm(caps, axis=1, keepdims=True)
        points = np.vstack((np.asarray(points), caps))
        points += 0.035 * rng.normal(size=points.shape)
        return normalize(points)

    # A diverse unrestricted population.  The structured starts are seeds,
    # not constraints; all subsequent coordinates evolve independently.
    population = []
    for _ in range(18):
        population.append(normalize(rng.normal(size=(n, d))))
    for _ in range(8):
        population.append(ring_seed())
    for _ in range(8):
        population.append(icosa_seed())
    population = np.asarray(population, dtype=np.float64)

    def optimize_population(
        x: np.ndarray,
        steps: int,
        beta0: float,
        beta1: float,
        lr0: float,
        lr1: float,
    ) -> np.ndarray:
        """Batched Adam ascent on softmin(log distance)-softmax(log distance)."""
        first = np.zeros_like(x)
        second = np.zeros_like(x)
        age = 0

        for step in range(steps):
            t = step / max(steps - 1, 1)
            beta = beta0 + (beta1 - beta0) * t
            learning = lr0 + (lr1 - lr0) * t

            delta = x[:, pi] - x[:, pj]
            dsq = np.einsum("bpd,bpd->bp", delta, delta) + 1.0e-15
            logd = 0.5 * np.log(dsq)

            lo = -beta * logd
            lo -= lo.max(axis=1, keepdims=True)
            near = np.exp(lo)
            near /= near.sum(axis=1, keepdims=True)

            hi = beta * logd
            hi -= hi.max(axis=1, keepdims=True)
            far = np.exp(hi)
            far /= far.sum(axis=1, keepdims=True)

            pair_force = (near - far)[:, :, None] * delta / dsq[:, :, None]
            gradient = np.einsum("np,bpd->bnd", incidence, pair_force)

            first = 0.90 * first + 0.10 * gradient
            second = 0.995 * second + 0.005 * gradient * gradient
            age += 1
            first_hat = first / (1.0 - 0.90 ** age)
            second_hat = second / (1.0 - 0.995 ** age)

            x += learning * first_hat / (np.sqrt(second_hat) + 1.0e-8)
            x = normalize_batch(x)

        return x

    population = optimize_population(
        population, 3300, 5.0, 115.0, 0.034, 0.0075
    )

    # Diversify the strongest contact geometries without discarding their
    # exact copies.  This often changes one marginal diameter contact.
    values = batched_scores(population)
    elite = population[np.argsort(values)[-10:]]
    descendants = []
    for parent in elite:
        descendants.append(parent)
        descendants.append(normalize(parent + 0.010 * rng.normal(size=(n, d))))
        descendants.append(normalize(parent + 0.027 * rng.normal(size=(n, d))))
        descendants.append(normalize(parent + 0.050 * rng.normal(size=(n, d))))
    population = np.asarray(descendants, dtype=np.float64)

    population = optimize_population(
        population, 3000, 65.0, 330.0, 0.014, 0.0022
    )

    values = batched_scores(population)
    order = np.argsort(values)[::-1]
    best = population[order[0]].copy()
    best_value = float(values[order[0]])

    # Exact epigraph solve after scaling d_min^2 to one:
    # minimize Q subject to 1 <= ||xi-xj||^2 <= Q.
    try:
        from scipy.optimize import minimize

        center_jac = np.zeros((d, n * d + 1), dtype=np.float64)
        for k in range(n):
            center_jac[:, d * k:d * k + d] = np.eye(d)

        def constrained_polish(seed: np.ndarray) -> np.ndarray:
            seed = normalize(seed)
            delta = seed[pi] - seed[pj]
            dsq = np.einsum("pd,pd->p", delta, delta)
            seed /= np.sqrt(dsq.min())

            delta = seed[pi] - seed[pj]
            dsq = np.einsum("pd,pd->p", delta, delta)
            x0 = np.concatenate((seed.ravel(), [float(dsq.max() * (1.0 + 1.0e-10))]))

            def con(x: np.ndarray) -> np.ndarray:
                p = x[:-1].reshape(n, d)
                v = p[pi] - p[pj]
                q = np.einsum("pd,pd->p", v, v)
                return np.concatenate((q - 1.0, x[-1] - q))

            def jac_con(x: np.ndarray) -> np.ndarray:
                p = x[:-1].reshape(n, d)
                v = p[pi] - p[pj]
                jac = np.zeros((2 * m_pairs, n * d + 1), dtype=np.float64)
                rows = np.arange(m_pairs)
                for coordinate in range(d):
                    g = 2.0 * v[:, coordinate]
                    jac[rows, d * pi + coordinate] = g
                    jac[rows, d * pj + coordinate] = -g
                    jac[m_pairs + rows, d * pi + coordinate] = -g
                    jac[m_pairs + rows, d * pj + coordinate] = g
                jac[m_pairs + rows, -1] = 1.0
                return jac

            result = minimize(
                lambda x: x[-1],
                x0,
                jac=lambda x: np.r_[np.zeros(n * d), 1.0],
                method="SLSQP",
                constraints=(
                    {"type": "ineq", "fun": con, "jac": jac_con},
                    {
                        "type": "eq",
                        "fun": lambda x: x[:-1].reshape(n, d).sum(axis=0),
                        "jac": lambda x: center_jac,
                    },
                ),
                options={"maxiter": 1000, "ftol": 5.0e-14, "disp": False},
            )

            if result.x is not None and np.all(np.isfinite(result.x)):
                return normalize(result.x[:-1].reshape(n, d))
            return normalize(seed)

        # Multiple basins are important because the exact problem is
        # nonsmooth whenever a shortest or farthest contact changes.
        for index in order[:7]:
            candidate = constrained_polish(population[index])
            candidate_value = score(candidate)
            if np.isfinite(candidate_value) and candidate_value > best_value:
                best = candidate
                best_value = candidate_value

        # One final reinitialized exact solve removes tiny accumulated
        # feasibility slack from the selected best contact graph.
        candidate = constrained_polish(best)
        candidate_value = score(candidate)
        if np.isfinite(candidate_value) and candidate_value > best_value:
            best = candidate
            best_value = candidate_value

    except Exception:
        pass

    return np.asarray(normalize(best), dtype=np.float64)


# EVOLVE-BLOCK-END