# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Deterministically construct a 14 point Euclidean diameter packing.

    A population-based smooth feasibility search is used to generate
    asymmetric candidates, followed by constrained active-set polishing of
    the exact formulation:
        maximize t
        subject to t <= ||x_i-x_j||^2 <= 1.
    """
    n = 14
    pi, pj = np.triu_indices(n, 1)
    pairs = len(pi)
    rng = np.random.default_rng(7301941)

    def dist2(x: np.ndarray) -> np.ndarray:
        d = x[pi] - x[pj]
        return np.einsum("ij,ij->i", d, d)

    def value(x: np.ndarray) -> float:
        q = dist2(x)
        return float(q.min() / q.max())

    def diameter_normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        x -= x[0]
        q = dist2(x)
        mx = float(q.max())
        if mx > 0.0 and np.isfinite(mx):
            x /= np.sqrt(mx)
        return x

    def rms_normalize(x: np.ndarray) -> np.ndarray:
        x = x - x.mean(axis=0, keepdims=True)
        s = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(s, 1e-12)

    def anneal(seed: np.ndarray, steps: int = 520) -> np.ndarray:
        """
        Smooth log-diameter force flow.  Low-distance pairs receive
        repulsion and high-distance pairs receive attraction.  RMS
        normalization removes the irrelevant scale direction.
        """
        x = rms_normalize(seed)
        velocity = np.zeros_like(x)

        for it in range(steps):
            u = it / max(steps - 1, 1)
            sharp = 5.0 + 115.0 * u * u

            d = x[pi] - x[pj]
            q = np.einsum("ij,ij->i", d, d)
            q = np.maximum(q, 1e-10)
            z = np.log(q)

            a = -sharp * z
            a -= a.max()
            low = np.exp(a)
            low /= low.sum()

            b = sharp * z
            b -= b.max()
            high = np.exp(b)
            high /= high.sum()

            # Gradient of soft-min(log q) - soft-max(log q).
            force_pair = 2.0 * (low - high)[:, None] * d / q[:, None]
            force = np.zeros_like(x)
            np.add.at(force, pi, force_pair)
            np.add.at(force, pj, -force_pair)

            force -= force.mean(axis=0, keepdims=True)
            velocity = 0.67 * velocity + force
            velocity -= velocity.mean(axis=0, keepdims=True)

            limit = 3.0
            vn = np.linalg.norm(velocity, axis=1, keepdims=True)
            velocity *= np.minimum(1.0, limit / np.maximum(vn, 1e-12))

            step = 0.030 * (1.0 - 0.72 * u)
            x = rms_normalize(x + step * velocity)

        return diameter_normalize(x)

    # Several geometrically distinct deterministic seeds.
    cube = np.array(
        [[a, b, c]
         for a in (-1.0, 1.0)
         for b in (-1.0, 1.0)
         for c in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.vstack((np.eye(3), -np.eye(3))) * np.sqrt(3.0)
    seeds = [np.vstack((cube, axes))]

    # Twisted two-ring configurations are useful asymmetric alternatives.
    for twist in (0.0, np.pi / 14.0, np.pi / 7.0):
        theta = 2.0 * np.pi * np.arange(7) / 7.0
        ring1 = np.column_stack((np.cos(theta), np.sin(theta), np.full(7, 0.52)))
        ring2 = np.column_stack((
            np.cos(theta + twist),
            np.sin(theta + twist),
            np.full(7, -0.52),
        ))
        seeds.append(np.vstack((ring1, ring2)))

    # Random full-dimensional seeds deliberately permit non-antipodal optima.
    for _ in range(26):
        seeds.append(rng.normal(size=(n, 3)))

    # Include some line-packings but perturb them so that antipodality is not
    # imposed during the subsequent full point optimization.
    for _ in range(10):
        u = rng.normal(size=(7, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        x = np.vstack((u, -u))
        x += 0.075 * rng.normal(size=x.shape)
        seeds.append(x)

    screened = []
    for seed in seeds:
        candidate = anneal(seed)
        screened.append((value(candidate), candidate))

    screened.sort(key=lambda item: item[0], reverse=True)

    # Keep candidates from separate rank positions, rather than only nearly
    # identical variants of the currently best basin.
    selected = []
    for _, candidate in screened:
        if not selected:
            selected.append(candidate)
            continue
        separation = min(np.mean((candidate - old) ** 2) for old in selected)
        if separation > 2.0e-4 or len(selected) < 4:
            selected.append(candidate)
        if len(selected) == 11:
            break

    best = diameter_normalize(screened[0][1])
    best_score = value(best)

    try:
        from scipy.optimize import minimize

        def unpack(y: np.ndarray) -> np.ndarray:
            return np.vstack((np.zeros((1, 3)), y[:-1].reshape(n - 1, 3)))

        def constraints(y: np.ndarray) -> np.ndarray:
            q = dist2(unpack(y))
            return np.concatenate((q - y[-1], 1.0 - q))

        def jacobian(y: np.ndarray) -> np.ndarray:
            x = unpack(y)
            d = 2.0 * (x[pi] - x[pj])
            jac = np.zeros((2 * pairs, 3 * (n - 1) + 1), dtype=float)

            mask_i = pi != 0
            rows_i = np.nonzero(mask_i)[0]
            cols_i = 3 * (pi[mask_i] - 1)
            for coordinate in range(3):
                jac[rows_i, cols_i + coordinate] = d[mask_i, coordinate]
                jac[pairs + rows_i, cols_i + coordinate] = -d[mask_i, coordinate]

            mask_j = pj != 0
            rows_j = np.nonzero(mask_j)[0]
            cols_j = 3 * (pj[mask_j] - 1)
            for coordinate in range(3):
                jac[rows_j, cols_j + coordinate] = -d[mask_j, coordinate]
                jac[pairs + rows_j, cols_j + coordinate] = d[mask_j, coordinate]

            jac[:pairs, -1] = -1.0
            return jac

        objective_gradient = np.zeros(3 * (n - 1) + 1)
        objective_gradient[-1] = -1.0
        bounds = [(-1.05, 1.05)] * (3 * (n - 1)) + [(0.0, 1.0)]
        cons = {"type": "ineq", "fun": constraints, "jac": jacobian}

        # Exact constrained polish of each globally screened candidate.
        for candidate in selected:
            candidate = diameter_normalize(candidate)
            q = dist2(candidate)
            initial_t = 0.992 * float(q.min())
            y0 = np.concatenate((candidate[1:].ravel(), [initial_t]))

            result = minimize(
                lambda y: -y[-1],
                y0,
                jac=lambda y: objective_gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter": 700, "ftol": 2.0e-12, "disp": False},
            )

            if result.x is None or not np.all(np.isfinite(result.x)):
                continue

            candidate = diameter_normalize(unpack(result.x))
            q = dist2(candidate)
            if q.min() <= 0.0:
                continue
            candidate_score = float(q.min() / q.max())
            if candidate_score > best_score:
                best_score = candidate_score
                best = candidate

    except Exception:
        pass

    return np.asarray(diameter_normalize(best), dtype=float)


# EVOLVE-BLOCK-END