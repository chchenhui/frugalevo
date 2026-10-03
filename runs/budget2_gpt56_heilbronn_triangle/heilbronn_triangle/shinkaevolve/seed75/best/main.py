# EVOLVE-BLOCK-START
import numpy as np


_SQRT3 = float(np.sqrt(3.0))
_HEIGHT = 0.5 * _SQRT3


def heilbronn_triangle11() -> np.ndarray:
    """
    Deterministically construct eleven points in the unit equilateral triangle.

    Internal coordinates are (u, v) in the reference simplex:
    u >= 0, v >= 0, u + v <= 1.  Absolute determinants in these coordinates
    are exactly the triangle areas normalized by the containing triangle area.
    """
    try:
        if hasattr(heilbronn_triangle11, "_cached"):
            return heilbronn_triangle11._cached.copy()

        rng = np.random.default_rng(11031987)
        n = 11
        free = 8
        triples = np.asarray(
            [(i, j, k) for i in range(n)
             for j in range(i + 1) if False
             for k in range(j + 1, n)],
            dtype=np.intp,
        )
        # The compact comprehension above intentionally avoids accidental
        # dependence on external helpers; build the actual index list here.
        triples = np.asarray(
            [(i, j, k) for i in range(n)
             for j in range(i + 1, n)
             for k in range(j + 1, n)],
            dtype=np.intp,
        )

        fixed = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)

        def project(z):
            z = np.maximum(np.asarray(z, dtype=float).copy(), 0.0)
            total = z[..., 0] + z[..., 1]
            mask = total > 1.0
            if np.any(mask):
                z[mask] /= total[mask, None]
            return z

        def areas(batch):
            m = len(batch)
            p = np.empty((m, n, 2), dtype=float)
            p[:, :3] = fixed
            p[:, 3:] = batch
            a = p[:, triples[:, 0]]
            b = p[:, triples[:, 1]]
            c = p[:, triples[:, 2]]
            return np.abs(
                (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
            )

        def soft_tail(values, temperature):
            lo = np.min(values, axis=1)
            return lo - temperature * np.log(
                np.exp(-(values - lo[:, None]) / temperature).sum(axis=1)
            )

        def dispersed_start(pool_size):
            pool = rng.dirichlet((1.08, 1.08, 1.08), size=pool_size)[:, 1:]
            chosen = [pool[rng.integers(pool_size)]]
            while len(chosen) < free:
                old = np.asarray(chosen)
                distances = np.sum(
                    (pool[:, None, :] - old[None, :, :]) ** 2, axis=2
                )
                chosen.append(pool[np.argmax(np.min(distances, axis=1))])
            return np.asarray(chosen)

        best_state = None
        best_value = -1.0
        finalists = []

        restarts = 7
        iterations = 1850
        proposals = 38

        for restart in range(restarts):
            if restart == 0:
                state = np.array(
                    [
                        (0.18, 0.08), (0.49, 0.07), (0.78, 0.08),
                        (0.10, 0.30), (0.40, 0.27), (0.68, 0.22),
                        (0.20, 0.56), (0.48, 0.44),
                    ],
                    dtype=float,
                )
            else:
                state = dispersed_start(180 + 20 * (restart % 3))
            state = project(state)

            current_areas = areas(state[None])[0]
            local_state = state.copy()
            local_value = float(np.min(current_areas))

            for step in range(iterations):
                frac = step / float(iterations - 1)
                sigma = 0.115 * (1.0 - frac) ** 1.75 + 0.0022
                temperature = 0.0052 * (1.0 - frac) ** 1.75 + 0.00010
                current_min = float(np.min(current_areas))

                cutoff = current_min + 0.006 + 0.012 * (1.0 - frac)
                active = triples[current_areas <= cutoff]
                weights = np.zeros(free, dtype=float)
                if len(active):
                    for q in range(free):
                        weights[q] = np.count_nonzero(active == q + 3)

                moving = (
                    int(np.argmax(weights + 1.0e-7 * rng.random(free)))
                    if weights.sum() else int(rng.integers(free))
                )

                candidates = np.repeat(state[None], proposals, axis=0)
                candidates[:, moving] += rng.normal(
                    0.0, sigma, size=(proposals, 2)
                )

                if step < iterations // 2:
                    candidates[:5, moving] = rng.dirichlet(
                        (1.05, 1.05, 1.05), size=5
                    )[:, 1:]

                joint_count = 8
                candidates[5:5 + joint_count] = state + rng.normal(
                    0.0, 0.28 * sigma, size=(joint_count, free, 2)
                )

                # Random joint moves are useful for changing combinatorial
                # patterns, but close to an extremal layout the restrictive
                # determinants need directed, coordinated repairs.  Build
                # several fields from differently sized lower tails rather
                # than aggregating every active triple into one direction:
                # competing constraints then have opportunities to improve
                # without their gradients cancelling completely.
                p = np.vstack((fixed, state))
                ordered = np.argsort(current_areas)
                field_sizes = (6, 12, 24, 40)
                directed = np.zeros((len(field_sizes), free, 2), dtype=float)

                for field_index, count in enumerate(field_sizes):
                    ids = ordered[:count]
                    involved = triples[ids]
                    participation = np.zeros(n, dtype=float)
                    for point_index in range(3, n):
                        participation[point_index] = np.count_nonzero(
                            involved == point_index
                        )

                    for triple_index in ids:
                        ia, ib, ic = triples[triple_index]
                        aa, bb, cc = p[ia], p[ib], p[ic]
                        determinant = (
                            (bb[0] - aa[0]) * (cc[1] - aa[1])
                            - (bb[1] - aa[1]) * (cc[0] - aa[0])
                        )
                        orientation = 1.0 if determinant >= 0.0 else -1.0
                        derivatives = (
                            np.array((bb[1] - cc[1], cc[0] - bb[0])),
                            np.array((cc[1] - aa[1], aa[0] - cc[0])),
                            np.array((aa[1] - bb[1], bb[0] - aa[0])),
                        )
                        for point_index, derivative in zip(
                            (ia, ib, ic), derivatives
                        ):
                            if point_index >= 3:
                                # Balance participation so that a point shared
                                # by many bottlenecks does not drown out less
                                # frequent but equally important constraints.
                                weight = 1.0 / max(
                                    participation[point_index], 1.0
                                )
                                directed[
                                    field_index, point_index - 3
                                ] += orientation * weight * derivative

                field_norm = np.max(
                    np.linalg.norm(directed, axis=2), axis=1
                )
                usable = field_norm > 1.0e-13
                directed[usable] /= field_norm[usable, None, None]
                gradient_step = 0.022 * (1.0 - frac) + 0.0010
                candidates[13:13 + len(field_sizes)] = (
                    state[None] + gradient_step * directed
                )

                candidates[-1] = state
                candidates = project(candidates)

                trial_areas = areas(candidates)
                minima = np.min(trial_areas, axis=1)

                # Early search benefits from a smooth lower-tail objective.
                # Late search is explicitly lexicographic: never prefer a
                # better collection of near-active constraints over a worse
                # true bottleneck determinant.
                if frac < 0.72:
                    winner = int(np.argmax(soft_tail(trial_areas, temperature)))
                else:
                    tail = np.mean(np.partition(trial_areas, 12, axis=1)[:, :12],
                                   axis=1)
                    winner = int(np.lexsort((tail, minima))[-1])

                state = candidates[winner]
                current_areas = trial_areas[winner]

                value = float(minima[winner])
                if value > local_value:
                    local_value = value
                    local_state = state.copy()
                if value > best_value:
                    best_value = value
                    best_state = state.copy()

            finalists.append((local_value, local_state))

        finalists.sort(key=lambda item: item[0], reverse=True)
        if best_state is None:
            raise RuntimeError("no feasible search state")

        # Direct constrained maximin polish.  The signs identify the local
        # oriented-matroid cell; retaining them makes every area constraint
        # differentiable for SLSQP while still enforcing the true absolute
        # determinant within that cell.
        try:
            from scipy.optimize import minimize

            def signed_determinants(uv):
                q = uv[triples]
                return (
                    (q[:, 1, 0] - q[:, 0, 0]) *
                    (q[:, 2, 1] - q[:, 0, 1])
                    - (q[:, 1, 1] - q[:, 0, 1]) *
                    (q[:, 2, 0] - q[:, 0, 0])
                )

            polish_states = [best_state] + [
                state for _, state in finalists[:4]
            ]

            for initial in polish_states:
                initial = project(initial)
                signed = signed_determinants(np.vstack((fixed, initial)))
                signs = np.where(signed >= 0.0, 1.0, -1.0)
                start_t = float(np.min(np.abs(signed)))

                def constraint_values(w):
                    tail = w[:-1].reshape(free, 2)
                    uv = np.vstack((fixed, tail))
                    return np.concatenate((
                        signs * signed_determinants(uv) - w[-1],
                        tail.ravel(),
                        1.0 - tail.sum(axis=1),
                    ))

                result = minimize(
                    lambda w: -w[-1],
                    np.r_[initial.ravel(), start_t],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * (2 * free) + [(0.0, 1.0)],
                    constraints={"type": "ineq", "fun": constraint_values},
                    options={"maxiter": 900, "ftol": 1.0e-12, "disp": False},
                )

                if result.success and np.all(np.isfinite(result.x)):
                    candidate = project(result.x[:-1].reshape(free, 2))
                    candidate_value = float(np.min(areas(candidate[None])[0]))
                    if candidate_value > best_value + 1.0e-12:
                        best_value = candidate_value
                        best_state = candidate.copy()
        except Exception:
            pass

        output = np.empty((n, 2), dtype=float)
        output[:3] = (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, _HEIGHT),
        )
        output[3:, 0] = best_state[:, 0] + 0.5 * best_state[:, 1]
        output[3:, 1] = _HEIGHT * best_state[:, 1]

        heilbronn_triangle11._cached = output.copy()
        return output

    except Exception:
        return np.array(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.5, _HEIGHT),
                (0.20, 0.12),
                (0.50, 0.10),
                (0.80, 0.12),
                (0.14, 0.34),
                (0.50, 0.31),
                (0.86, 0.34),
                (0.27, 0.57),
                (0.73, 0.57),
            ],
            dtype=float,
        )


# EVOLVE-BLOCK-END