import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Annealed spherical construction followed by exact diameter-constrained polish."""
    n = 14
    rng = np.random.default_rng(20260912)
    ii, jj = np.triu_indices(n, 1)
    m = len(ii)

    index = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / n
    golden = np.pi * (3.0 - np.sqrt(5.0))
    seed = np.column_stack((
        np.sqrt(1.0 - z * z) * np.cos(golden * index),
        np.sqrt(1.0 - z * z) * np.sin(golden * index),
        z,
    ))

    def distances2(x):
        d = x[ii] - x[jj]
        return np.einsum("ij,ij->i", d, d)

    def ratio(x):
        q = distances2(x)
        return float(q.min() / q.max())

    best = seed.copy()
    best_value = ratio(best)

    # Explore unrestricted radial configurations with smooth Riesz-energy
    # continuation, retaining the best state for the exact polishing stage.
    for restart in range(8):
        if restart == 0:
            points = seed.copy()
        else:
            points = rng.normal(size=(n, 3))
        points = points.astype(float, copy=True)
        points -= points.mean(axis=0, keepdims=True)
        diameter = np.sqrt(distances2(points).max())
        if not np.isfinite(diameter) or diameter <= 0.0:
            continue
        points /= diameter

        # Increasing powers progressively replace smooth repulsion by a
        # max-min separation objective while preserving unrestricted radii.
        for power in (2, 4, 8, 16):
            epsilon = 2.0e-3 / power
            for iteration in range(180):
                delta = points[ii] - points[jj]
                q = np.einsum("ij,ij->i", delta, delta)
                weight = -2.0 * power * (q + epsilon) ** (-power - 1)
                gradient = np.zeros_like(points)
                np.add.at(gradient, ii, weight[:, None] * delta)
                np.add.at(gradient, jj, -weight[:, None] * delta)

                scale = np.max(np.linalg.norm(gradient, axis=1))
                if not np.isfinite(scale) or scale <= 1.e-14:
                    break
                learning_rate = 0.075 * (1.0 - 0.55 *
                                         iteration / 179.0)
                points -= learning_rate * gradient / scale
                points -= points.mean(axis=0, keepdims=True)
                diameter = np.sqrt(distances2(points).max())
                if not np.isfinite(diameter) or diameter <= 0.0:
                    break
                points /= diameter

        value = ratio(points)
        if np.isfinite(value) and value > best_value:
            best, best_value = points.copy(), value

    # Directly optimize the evaluator-visible min/diameter objective by
    # fixing squared diameter to one and maximizing the common lower bound.
    try:
        from scipy.optimize import minimize

        def normalize(x):
            x = np.asarray(x, dtype=float).reshape(n, 3).copy()
            x -= x.mean(axis=0, keepdims=True)
            qmax = distances2(x).max()
            if not np.isfinite(qmax) or qmax <= 0.0:
                return None
            return x / np.sqrt(qmax)

        def constraints(v):
            x = v[:-1].reshape(n, 3)
            t = v[-1]
            q = distances2(x)
            return np.concatenate((q - t, 1.0 - q))

        def constraint_jacobian(v):
            x = v[:-1].reshape(n, 3)
            delta = x[ii] - x[jj]
            jac = np.zeros((2 * m, 3 * n + 1), dtype=float)
            rows = np.arange(m)
            for coord in range(3):
                jac[rows, 3 * ii + coord] = 2.0 * delta[:, coord]
                jac[rows, 3 * jj + coord] = -2.0 * delta[:, coord]
            jac[:m, -1] = -1.0
            jac[m:, :-1] = -jac[:m, :-1]
            return jac

        # Adaptive six-vertex contact-star patch.  The six shortest and six
        # longest pairs identify a local packing defect and its diameter
        # boundary; only the resulting six vertices are made movable.
        qbest = distances2(best)
        low = np.argsort(qbest)[:6]
        high = np.argsort(qbest)[-6:]
        incidence = np.zeros(n, dtype=float)
        for edges, weight in ((low, 1.0), (high, 1.0)):
            incidence += np.bincount(
                np.r_[ii[edges], jj[edges]], minlength=n
            )
        centers = list(np.argsort(-incidence))
        active_endpoint = int(np.argmax(
            np.bincount(np.r_[ii[high], jj[high]], minlength=n)
        ))
        centers = [centers[0],
                   next((x for x in centers if x != centers[0]), centers[0]),
                   active_endpoint]

        patch_best = best.copy()
        patch_value = best_value
        for center in centers:
            lower_neighbors = []
            upper_neighbors = []
            for e in low:
                if ii[e] == center:
                    lower_neighbors.append((qbest[e], int(jj[e])))
                elif jj[e] == center:
                    lower_neighbors.append((qbest[e], int(ii[e])))
            for e in high:
                if ii[e] == center:
                    upper_neighbors.append((-qbest[e], int(jj[e])))
                elif jj[e] == center:
                    upper_neighbors.append((-qbest[e], int(ii[e])))
            chosen = [center]
            for _, vertex in sorted(lower_neighbors):
                if vertex not in chosen:
                    chosen.append(vertex)
                if len(chosen) == 4:
                    break
            for _, vertex in sorted(upper_neighbors):
                if vertex not in chosen:
                    chosen.append(vertex)
                if len(chosen) == 6:
                    break
            for vertex in range(n):
                if len(chosen) == 6:
                    break
                if vertex not in chosen:
                    chosen.append(vertex)
            movable = np.asarray(chosen[:6], dtype=int)
            fixed = np.asarray([x for x in range(n) if x not in movable],
                               dtype=int)

            def unpack_patch(v):
                x = best.copy()
                x[movable] = v[:-1].reshape(6, 3)
                return x

            def patch_constraints(v):
                q = distances2(unpack_patch(v))
                return np.concatenate((q - v[-1], 1.0 - q))

            q0 = distances2(best)
            contact_edges = np.r_[low, high]
            jac = np.zeros((len(contact_edges), 18), dtype=float)
            for r, e in enumerate(contact_edges):
                if ii[e] in movable:
                    a = int(np.where(movable == ii[e])[0][0])
                    jac[r, 3*a:3*a+3] = 2.0 * (best[ii[e]] - best[jj[e]])
                if jj[e] in movable:
                    a = int(np.where(movable == jj[e])[0][0])
                    jac[r, 3*a:3*a+3] = 2.0 * (best[jj[e]] - best[ii[e]])
            _, _, vh = np.linalg.svd(jac, full_matrices=False)
            displacement = vh[-1].reshape(6, 3)
            patch_starts = [
                np.concatenate((best[movable].ravel(), [float(q0.min())])),
                np.concatenate(((best[movable] + 0.006 * displacement).ravel(),
                                [float(q0.min())])),
            ]
            for v0 in patch_starts:
                result = minimize(
                    lambda v: -v[-1], v0, method="SLSQP",
                    constraints={"type": "ineq", "fun": patch_constraints},
                    bounds=[(None, None)] * 18 + [(0.0, 1.0)],
                    options={"maxiter": 220, "ftol": 1.e-11, "disp": False},
                )
                if result.x is not None and np.all(np.isfinite(result.x)):
                    candidate = normalize(unpack_patch(result.x))
                    if candidate is not None and ratio(candidate) > patch_value:
                        patch_best, patch_value = candidate, ratio(candidate)

        # One final full-coordinate polish retains the patch's global benefit.
        q = distances2(patch_best)
        v0 = np.concatenate((patch_best.ravel(), [float(q.min())]))
        result = minimize(
            lambda v: -v[-1], v0, method="SLSQP",
            jac=lambda v: np.r_[np.zeros(3 * n), -1.0],
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraint_jacobian},
            bounds=[(None, None)] * (3 * n) + [(0.0, 1.0)],
            options={"maxiter": 180, "ftol": 1.e-11, "disp": False},
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            candidate = normalize(result.x[:-1])
            if candidate is not None and ratio(candidate) > best_value:
                best, best_value = candidate, ratio(candidate)
        if patch_value > best_value:
            best, best_value = patch_best, patch_value
    except Exception:
        pass

    return np.asarray(best, dtype=float)