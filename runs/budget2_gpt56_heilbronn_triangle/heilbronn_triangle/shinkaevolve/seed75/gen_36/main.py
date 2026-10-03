# EVOLVE-BLOCK-START
import numpy as np


_SQRT3 = float(np.sqrt(3.0))
_HEIGHT = _SQRT3 * 0.5


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the unit equilateral triangle.

    The three corners are retained explicitly.  The remaining points are
    optimized in barycentric coordinates by repeatedly replacing the point
    involved in the most active small-area constraints.
    """
    try:
        rng = np.random.default_rng(11031987)

        # All 165 triples of eleven points.
        triples = np.array(
            [(i, j, k)
             for i in range(11)
             for j in range(i + 1, 11)
             for k in range(j + 1, 11)],
            dtype=np.intp,
        )

        # First three points are the triangle's corners.  A point with
        # barycentric variables (b,c) has Cartesian coordinates
        # (b + c/2, c*sqrt(3)/2).
        corners = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, _HEIGHT]], dtype=float
        )

        def project_simplex(bc):
            """Keep b,c,a=1-b-c positive, preserving batched inputs."""
            z = np.asarray(bc, dtype=float).copy()
            # Boundary points are feasible and are frequently present in
            # extremal Heilbronn configurations.  In particular, do not
            # perturb an SLSQP boundary solution back into the interior.
            eps = 0.0
            z = np.maximum(z, eps)
            s = z[..., 0] + z[..., 1]
            cap = 1.0 - eps
            mask = s > cap
            if np.any(mask):
                z[mask] *= (cap / s[mask])[..., None]
            return z

        def cartesian(bc):
            p = np.empty((bc.shape[0], 2), dtype=float)
            p[:, 0] = bc[:, 0] + 0.5 * bc[:, 1]
            p[:, 1] = _HEIGHT * bc[:, 1]
            return p

        def areas_for_batch(batch_bc):
            """
            Return normalized areas for a batch of 8-point barycentric states.
            Normalization is by the area of the containing equilateral triangle.
            """
            m = batch_bc.shape[0]
            pts = np.empty((m, 11, 2), dtype=float)
            pts[:, :3, :] = corners
            pts[:, 3:, :] = cartesian(batch_bc.reshape(-1, 2)).reshape(m, 8, 2)

            a = pts[:, triples[:, 0], :]
            b = pts[:, triples[:, 1], :]
            c = pts[:, triples[:, 2], :]
            cross = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
            cross -= (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

            # (cross / 2) divided by (sqrt(3) / 4).
            return np.abs(cross) * (2.0 / _SQRT3)

        def tail_score(areas, temperature):
            """
            Stable soft-min.  Unlike a raw minimum, this also improves several
            nearly active constraints simultaneously.
            """
            lo = np.min(areas, axis=1)
            return lo - temperature * np.log(
                np.sum(np.exp(-(areas - lo[:, None]) / temperature), axis=1)
            )

        best_bc = None
        best_minimum = -np.inf
        # Preserve good states from separate orientation chambers.  A layout
        # that is not the best before nonlinear polishing can still be the
        # best epigraph solution once its determinant signs are fixed.
        archive = []

        # Independent deterministic starts make escaping combinatorial local
        # arrangements substantially more reliable than a single hill climb.
        restarts = 7
        iterations = 1850
        proposals = 34

        for restart in range(restarts):
            # Uniform simplex samples, followed by a mild repulsive selection.
            pool = rng.dirichlet((1.15, 1.15, 1.15), size=160)[:, 1:]
            chosen = [pool[rng.integers(len(pool))]]

            # Select remaining initial points far from already selected ones.
            while len(chosen) < 8:
                old = np.asarray(chosen)
                delta = pool[:, None, :] - old[None, :, :]
                dist = np.sum(delta * delta, axis=2)
                idx = int(np.argmax(np.min(dist, axis=1)))
                chosen.append(pool[idx])

            state = project_simplex(np.asarray(chosen))
            current_areas = areas_for_batch(state[None, :, :])[0]
            current_min = float(np.min(current_areas))
            restart_best = state.copy()
            restart_bestimum = current_min

            for step in range(iterations):
                frac = step / float(iterations - 1)
                # Late iterations are fine local corrections.
                sigma = 0.115 * (1.0 - frac) ** 1.7 + 0.0035
                temp = 0.0055 * (1.0 - frac) + 0.00055

                # Weight the points appearing in the most restrictive triples.
                cutoff = current_min + 0.0075 + 0.008 * (1.0 - frac)
                active = current_areas <= cutoff
                weights = np.zeros(8, dtype=float)
                active_triples = triples[active]
                if len(active_triples):
                    for point_index in range(3, 11):
                        weights[point_index - 3] = np.count_nonzero(
                            active_triples == point_index
                        )

                if weights.sum() <= 0.0:
                    moving = int(rng.integers(8))
                else:
                    # A little deterministic cycling prevents one point from
                    # monopolizing the update when several constraints tie.
                    moving = int(np.argmax(weights + 1.0e-5 * rng.random(8)))

                candidates = np.repeat(state[None, :, :], proposals, axis=0)

                # Most proposals are local; a few early candidates are global
                # simplex samples, allowing exchanges between distinct patterns.
                noise = rng.normal(0.0, sigma, size=(proposals, 2))
                candidates[:, moving, :] += noise

                if step < iterations // 2:
                    global_count = 5
                    candidates[:global_count, moving, :] = rng.dirichlet(
                        (1.1, 1.1, 1.1), size=global_count
                    )[:, 1:]

                # Preserve the incumbent as one candidate.
                candidates[-1, moving, :] = state[moving]
                candidates[:, moving, :] = project_simplex(
                    candidates[:, moving, :]
                )

                # The final bottleneck is normally a coupled collection of
                # near-zero-slack determinants.  Repair those constraints by
                # moving every participating free point simultaneously rather
                # than relying exclusively on coordinate-wise changes.
                if step >= (3 * iterations) // 4 and len(active_triples):
                    involved = np.unique(active_triples[active_triples >= 3]) - 3
                    if len(involved):
                        repair_count = 10
                        scales = np.linspace(0.28, 1.35, repair_count) * sigma
                        repairs = np.repeat(
                            state[None, :, :], repair_count, axis=0
                        )
                        repairs[:, involved, :] += (
                            rng.normal(
                                0.0, 1.0, size=(repair_count, len(involved), 2)
                            )
                            * scales[:, None, None]
                        )
                        candidates[:repair_count] = project_simplex(repairs)

                candidate_areas = areas_for_batch(candidates)
                scores = tail_score(candidate_areas, temp)
                winner = int(np.argmax(scores))

                # The incumbent is in the batch, so this update is monotone
                # for the current smooth lower-tail objective.
                state = candidates[winner]
                current_areas = candidate_areas[winner]
                current_min = float(np.min(current_areas))

                if current_min > restart_bestimum:
                    restart_bestimum = current_min
                    restart_best = state.copy()

                if current_min > best_minimum:
                    best_minimum = current_min
                    best_bc = state.copy()

            archive.append((restart_bestimum, restart_best))

        if best_bc is None or not np.all(np.isfinite(best_bc)):
            raise RuntimeError("search did not produce a feasible simplex state")

        # With the orientation pattern fixed by the discrete search, the
        # remaining local problem is a conventional smooth maximin program.
        # This is particularly effective after joint repair has exposed the
        # small set of genuinely active determinant constraints.
        try:
            from scipy.optimize import minimize

            simplex_corners = np.array(
                ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float
            )

            def signed_determinants(tail):
                uv = np.vstack((simplex_corners, tail))
                q = uv[triples]
                return (
                    (q[:, 1, 0] - q[:, 0, 0]) * (q[:, 2, 1] - q[:, 0, 1])
                    - (q[:, 1, 1] - q[:, 0, 1]) * (q[:, 2, 0] - q[:, 0, 0])
                )

            # Polish several archived chambers rather than betting on the
            # pre-polish winner alone.  Restricting this to four states keeps
            # the expensive nonlinear stage bounded and deterministic.
            archive.sort(key=lambda item: item[0], reverse=True)
            for seed_minimum, seed_bc in archive[:4]:
                signed0 = signed_determinants(seed_bc)
                orientations = np.sign(signed0)
                orientations[orientations == 0.0] = 1.0

                def polish_constraints(w):
                    tail = w[:-1].reshape(8, 2)
                    return np.concatenate((
                        orientations * signed_determinants(tail) - w[-1],
                        1.0 - tail.sum(axis=1),
                    ))

                polished = minimize(
                    lambda w: -w[-1],
                    np.r_[seed_bc.ravel(), seed_minimum],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * 16 + [(0.0, 1.0)],
                    constraints={"type": "ineq", "fun": polish_constraints},
                    options={"maxiter": 1800, "ftol": 2.0e-12, "disp": False},
                )
                if polished.success and np.all(np.isfinite(polished.x)):
                    candidate = project_simplex(polished.x[:-1].reshape(8, 2))
                    candidate_minimum = float(
                        np.min(areas_for_batch(candidate[None, :, :])[0])
                    )
                    if candidate_minimum > best_minimum + 1.0e-11:
                        best_bc = candidate
                        best_minimum = candidate_minimum
        except Exception:
            pass

        result = np.empty((11, 2), dtype=float)
        result[:3] = corners
        result[3:] = cartesian(best_bc)
        return result

    except Exception:
        # Feasible deterministic fallback, used only if an unexpected numerical
        # failure occurs during the optimization search.
        return np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, _HEIGHT],
                [0.20, 0.12],
                [0.50, 0.10],
                [0.80, 0.12],
                [0.14, 0.34],
                [0.50, 0.31],
                [0.86, 0.34],
                [0.27, 0.57],
                [0.73, 0.57],
            ],
            dtype=float,
        )


# EVOLVE-BLOCK-END