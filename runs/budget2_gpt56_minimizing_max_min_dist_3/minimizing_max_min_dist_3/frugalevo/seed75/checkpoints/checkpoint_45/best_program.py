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

    # Explore twelve deterministic two-shell starts: six
    # cube--octahedral shells and six truncated-octahedral shells.
    structured_starts = []

    cube = np.asarray([
        (sx, sy, sz)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ], dtype=float)

    axial = np.asarray([
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ], dtype=float)

    face_pairs = np.asarray([
        (1.0, 1.0, 0.0), (-1.0, -1.0, 0.0),
        (1.0, 0.0, 1.0), (-1.0, 0.0, -1.0),
        (0.0, 1.0, 1.0), (0.0, -1.0, -1.0),
    ], dtype=float)
    face_pairs /= np.linalg.norm(face_pairs, axis=1, keepdims=True)

    for shell_ratio in (1.35, 1.50, 1.65, 1.80, 1.95, 2.10):
        structured_starts.append(
            np.vstack((cube, shell_ratio * axial)).astype(float)
        )

    alternating = np.asarray(
        [1.06, 0.94, 1.06, 0.94, 1.06, 0.94], dtype=float
    )
    for variant in range(6):
        perm = np.roll(np.arange(3), variant % 3)
        signs = 1.0 if variant < 3 else -1.0
        faces = face_pairs[:, perm] * signs
        faces = faces.copy()
        structured_starts.append(
            np.vstack((cube, alternating[:, None] * faces)).astype(float)
        )

    # Distinct, centered release directions prevent the continuation from
    # remaining trapped in the highly symmetric shell subspace.
    release = rng.normal(size=(n, 3)).astype(float)
    release -= release.mean(axis=0, keepdims=True)
    release_norm = np.linalg.norm(release, axis=1, keepdims=True)
    release /= np.maximum(release_norm, 1.0e-12)

    # Increasing powers progressively replace smooth repulsion by a
    # max-min separation objective while preserving unrestricted radii.
    for restart, initial in enumerate(structured_starts):
        points = np.asarray(initial, dtype=float).copy()
        points -= points.mean(axis=0, keepdims=True)
        diameter = np.sqrt(distances2(points).max())
        if not np.isfinite(diameter) or diameter <= 0.0:
            continue
        points /= diameter

        # Increasing powers progressively replace smooth repulsion by a
        # max-min separation objective while preserving unrestricted radii.
        for power in (2, 4, 8, 16):
            epsilon = 2.0e-3 / power
            for iteration in range(120):
                if power == 8:
                    amplitude = 0.015 * (1.0 - iteration / 119.0)
                    points += amplitude * release
                    points -= points.mean(axis=0, keepdims=True)
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
                                         iteration / 119.0)
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

        """Use three Delaunay cavities as boundary-conditioned local patches."""
        from scipy.spatial import Delaunay, QhullError

        qbest = distances2(best)
        low_edge = int(np.argmin(qbest))
        high_edge = int(np.argmax(qbest))
        low_endpoints = (int(ii[low_edge]), int(jj[low_edge]))
        high_endpoints = (int(ii[high_edge]), int(jj[high_edge]))

        try:
            triangulation = Delaunay(np.asarray(best, dtype=float))
            simplices = np.asarray(triangulation.simplices, dtype=int)
        except (QhullError, ValueError, RuntimeError):
            simplices = np.empty((0, 4), dtype=int)

        patch_best = best.copy()
        patch_value = best_value
        if len(simplices):
            candidates = []
            for endpoint in low_endpoints:
                candidates.append(endpoint)
            upper_count = np.bincount(
                np.r_[ii[np.argsort(qbest)[-6:]],
                      jj[np.argsort(qbest)[-6:]]], minlength=n)
            candidates.append(int(high_endpoints[
                int(np.argmax(upper_count[list(high_endpoints)]))
            ]))

            seen = set()
            for center in candidates:
                if center in seen:
                    continue
                seen.add(center)
                incident = simplices[np.any(simplices == center, axis=1)]
                if not len(incident):
                    continue
                union = np.unique(incident.ravel())
                if len(union) < 3:
                    continue
                if len(union) > 7:
                    distances_from_center = np.linalg.norm(
                        best[union] - best[center], axis=1)
                    union = union[np.argsort(distances_from_center)[:7]]
                movable = np.asarray(union, dtype=int)
                k = len(movable)
                if k < 3 or k > 7:
                    continue

                def unpack_cavity(v):
                    x = best.copy()
                    x[movable] = np.asarray(
                        v[:-1], dtype=float).reshape(k, 3)
                    return x

                def cavity_constraints(v):
                    q = distances2(unpack_cavity(v))
                    return np.concatenate((q - v[-1], 1.0 - q))

                q0 = distances2(best)
                v0 = np.concatenate((
                    np.asarray(best[movable], dtype=float).ravel(),
                    [float(q0.min())]))
                result = minimize(
                    lambda v: -v[-1], v0, method="SLSQP",
                    constraints={"type": "ineq", "fun": cavity_constraints},
                    bounds=[(None, None)] * (3 * k) + [(0.0, 1.0)],
                    options={"maxiter": 180, "ftol": 1.e-11,
                             "disp": False},
                )
                if result.x is not None and np.all(np.isfinite(result.x)):
                    candidate = normalize(unpack_cavity(result.x))
                    if candidate is not None and ratio(candidate) > patch_value:
                        patch_best = np.asarray(candidate, dtype=float).copy()
                        patch_value = ratio(patch_best)

        # Refine in the centered rank-three Gram face before recovering
        # coordinates.  This permits coherent changes to several contact
        # distances which are awkward to obtain by direct coordinate motion.
        def gram_project(g):
            """Project a symmetric Gram matrix onto the centered PSD rank-three face."""
            h = np.eye(n) - np.ones((n, n), dtype=float) / n
            g = h @ ((g + g.T) * 0.5) @ h
            eigenvalues, eigenvectors = np.linalg.eigh(g)
            keep = np.argsort(eigenvalues)[-3:]
            values = np.maximum(eigenvalues[keep], 0.0)
            g = (eigenvectors[:, keep] * values) @ eigenvectors[:, keep].T
            return h @ ((g + g.T) * 0.5) @ h

        def gram_coordinates(g):
            """Recover centered three-dimensional coordinates and normalize diameter."""
            g = gram_project(g)
            eigenvalues, eigenvectors = np.linalg.eigh(g)
            keep = np.argsort(eigenvalues)[-3:]
            values = np.maximum(eigenvalues[keep], 0.0)
            x = eigenvectors[:, keep] * np.sqrt(values)
            x -= x.mean(axis=0, keepdims=True)
            qmax = distances2(x).max()
            if not np.isfinite(qmax) or qmax <= 0.0:
                return None
            return x / np.sqrt(qmax)

        base = np.asarray(patch_best, dtype=float)
        base -= base.mean(axis=0, keepdims=True)
        gram_base = base @ base.T
        gram_starts = [gram_base]
        for sign in (-1.0, 1.0):
            perturbation = np.outer(
                base[:, 0] * base[:, 1],
                base[:, 0] * base[:, 1])
            gram_starts.append(gram_project(
                gram_base + sign * 0.035 * perturbation))

        gram_candidates = []
        for start in gram_starts:
            gram = gram_project(start)
            for major in range(90):
                diagonal = np.diag(gram)
                qgram = diagonal[:, None] + diagonal[None, :] - 2.0 * gram
                upper = qgram[ii, jj]
                lower = float(upper.min())
                upper_bound = float(upper.max())
                if not np.isfinite(lower + upper_bound):
                    break

                # A subgradient of the active lower bound, opposed by a
                # barrier-like subgradient for distances at the diameter.
                low = np.flatnonzero(upper <= lower + 2.0e-5)
                high = np.flatnonzero(upper >= upper_bound - 2.0e-5)
                direction = np.zeros((n, n), dtype=float)
                for edge in low:
                    a, b = int(ii[edge]), int(jj[edge])
                    direction[a, a] += 1.0
                    direction[b, b] += 1.0
                    direction[a, b] -= 1.0
                    direction[b, a] -= 1.0
                for edge in high:
                    a, b = int(ii[edge]), int(jj[edge])
                    direction[a, a] -= 0.42
                    direction[b, b] -= 0.42
                    direction[a, b] += 0.42
                    direction[b, a] += 0.42
                direction = (direction + direction.T) * 0.5
                direction -= direction.mean(axis=0, keepdims=True)
                direction -= direction.mean(axis=1, keepdims=True)
                step = 0.018 * (1.0 - 0.55 * major / 89.0)
                gram = gram_project(gram + step * direction)
                recovered = gram_coordinates(gram)
                if recovered is not None:
                    gram_candidates.append(recovered)

        # Recover each Gram candidate with one bounded diameter-constrained
        # coordinate polish, retaining the best evaluator-visible ratio.
        for candidate0 in gram_candidates:
            q0 = distances2(candidate0)
            v0 = np.concatenate((candidate0.ravel(), [float(q0.min())]))
            result = minimize(
                lambda v: -v[-1], v0, method="SLSQP",
                jac=lambda v: np.r_[np.zeros(3 * n), -1.0],
                constraints={"type": "ineq", "fun": constraints,
                             "jac": constraint_jacobian},
                bounds=[(None, None)] * (3 * n) + [(0.0, 1.0)],
                options={"maxiter": 140, "ftol": 1.e-11, "disp": False},
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