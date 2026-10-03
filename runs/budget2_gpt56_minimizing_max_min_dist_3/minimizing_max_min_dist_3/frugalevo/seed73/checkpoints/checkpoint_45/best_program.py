import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Soft spherical search followed by one exact fixed-diameter epigraph polish."""
    n = 14
    rng = np.random.default_rng(20250314)
    pi, pj = np.triu_indices(n, 1)
    m = len(pi)

    k = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / n
    theta = np.pi * (3.0 - np.sqrt(5.0)) * k
    base = np.column_stack((
        np.sqrt(1.0 - z * z) * np.cos(theta),
        np.sqrt(1.0 - z * z) * np.sin(theta),
        z,
    ))

    def ratio_of(p):
        q = p[pi] - p[pj]
        ds2 = np.einsum("ij,ij->i", q, q)
        if not np.all(np.isfinite(ds2)) or ds2.max() <= 0.0:
            return -np.inf
        return float(ds2.min() / ds2.max())

    best = base.copy()
    best_ratio = ratio_of(best)

    # Deterministic FCC-window tournament.  Unlike the spherical starts, these
    # seeds contain genuine interior tetrahedral and octahedral voids.
    lattice = np.asarray([
        (a, b, c)
        for a in range(-3, 4)
        for b in range(-3, 4)
        for c in range(-3, 4)
        if (a + b + c) % 2 == 0
    ], dtype=float)
    lattice -= lattice.mean(axis=0, keepdims=True)
    lattice /= np.linalg.norm(lattice, axis=1).max()

    seed_sites = np.argsort(
        np.einsum("ij,ij->i", lattice, lattice), kind="stable"
    )[-18:][::-1]
    seeds = []
    for start in seed_sites:
        selected = [int(start)]
        while len(selected) < n:
            unused = np.asarray(
                [i for i in range(len(lattice)) if i not in selected],
                dtype=int,
            )
            chosen = unused[np.argmax(np.min(
                np.linalg.norm(
                    lattice[unused, None, :] - lattice[selected][None, :, :],
                    axis=2,
                ),
                axis=1,
            ))]
            selected.append(int(chosen))
        points = lattice[np.asarray(selected, dtype=int)].copy()
        points -= points.mean(axis=0, keepdims=True)
        diameter = np.linalg.norm(
            points[:, None, :] - points[None, :, :], axis=2
        ).max()
        if diameter > 0.0:
            points /= diameter
        seeds.append(points)

    def release(points, start, stop):
        """Run a bounded FCC inverse-power release and normalize its diameter."""
        points = np.asarray(points, dtype=float).copy()
        for iteration in range(start, stop):
            delta = points[pi] - points[pj]
            dist2 = np.einsum("ij,ij->i", delta, delta)
            if not np.all(np.isfinite(dist2)):
                return points
            dist2 = np.maximum(dist2, 1e-8)

            power = 6.0 + 6.0 * iteration / 899.0
            weights = dist2 ** (-(power + 2.0) / 2.0)
            weights /= np.maximum(weights.sum(), 1e-300)

            gradient = np.zeros_like(points)
            contribution = weights[:, None] * delta
            np.add.at(gradient, pi, contribution)
            np.add.at(gradient, pj, -contribution)

            step = 0.018 * (1.0 - 0.65 * iteration / 899.0)
            points += step * gradient
            points -= points.mean(axis=0, keepdims=True)

            pair_delta = points[pi] - points[pj]
            diameter = np.sqrt(np.max(
                np.einsum("ij,ij->i", pair_delta, pair_delta)
            ))
            if not np.isfinite(diameter) or diameter <= 1e-12:
                return points
            points /= diameter
        return points

    # Screen all crystallographic windows, then fully release only the three
    # best FCC contact topologies.  Each seed receives at most 900 iterations.
    screened = []
    for seed in seeds:
        trial = release(seed, 0, 300)
        value = ratio_of(trial)
        if np.isfinite(value):
            screened.append((value, trial.copy()))

    screened.sort(key=lambda item: item[0], reverse=True)
    released = []
    for _, trial in screened[:3]:
        polished = release(trial, 300, 900)
        value = ratio_of(polished)
        if np.isfinite(value):
            released.append((value, polished.copy()))

    released.sort(key=lambda item: item[0], reverse=True)
    for value, points in released:
        if value > best_ratio:
            best_ratio = value
            best = points.copy()

    # Alternating-contact-block-repack: first move the seven vertices covering
    # the shortest contacts while holding the complementary seven fixed.
    def alternating_block_repack(points):
        """Optimize alternating seven-vertex contact-cover blocks with SLSQP."""
        try:
            from scipy.optimize import minimize

            diameter2 = float(np.max(
                np.einsum("ij,ij->i", points[pi] - points[pj],
                          points[pi] - points[pj])
            ))
            for _ in range(4):
                d = points[pi] - points[pj]
                q = np.einsum("ij,ij->i", d, d)
                order = np.argsort(q, kind="stable")
                chosen = []
                covered = np.zeros(m, dtype=bool)
                for edge in order:
                    a, b = int(pi[edge]), int(pj[edge])
                    if a not in chosen or b not in chosen:
                        if a not in chosen:
                            chosen.append(a)
                        if len(chosen) >= 7:
                            break
                        if b not in chosen:
                            chosen.append(b)
                    if len(chosen) >= 7:
                        break
                chosen = np.asarray(sorted(set(chosen)), dtype=int)
                if chosen.size < 7:
                    chosen = np.asarray(
                        sorted(set(chosen.tolist() +
                                   [i for i in range(n) if i not in chosen][:7-chosen.size])),
                        dtype=int)
                for block in (chosen, np.asarray(
                        [i for i in range(n) if i not in set(chosen)], dtype=int)):
                    fixed = np.asarray([i for i in range(n) if i not in set(block)],
                                       dtype=int)
                    bsz = block.size
                    local = points[block].copy()
                    q0 = np.einsum(
                        "ij,ij->i", points[pi] - points[pj],
                        points[pi] - points[pj])
                    x0 = np.r_[local.ravel(), float(q0.min()) * (1.0 - 1e-9)]

                    def fun(x):
                        """Maximize the epigraph minimum squared distance."""
                        return -x[-1]

                    def con(x):
                        """Return global lower and fixed-diameter inequalities."""
                        trial = points.copy()
                        trial[block] = x[:-1].reshape(bsz, 3)
                        dd = trial[pi] - trial[pj]
                        qq = np.einsum("ij,ij->i", dd, dd)
                        return np.r_[qq - x[-1], diameter2 - qq]

                    result = minimize(
                        fun, x0, method="SLSQP",
                        bounds=[(None, None)] * (3 * bsz) + [(0.0, diameter2)],
                        constraints={"type": "ineq", "fun": con},
                        options={"maxiter": 140, "ftol": 1e-10, "disp": False},
                    )
                    if np.all(np.isfinite(result.x)):
                        trial = points.copy()
                        trial[block] = result.x[:-1].reshape(bsz, 3)
                        if ratio_of(trial) > ratio_of(points):
                            points = trial
                            diameter2 = float(np.max(
                                np.einsum("ij,ij->i",
                                          points[pi] - points[pj],
                                          points[pi] - points[pj])
                            ))
            return points
        except Exception:
            return points

    best = alternating_block_repack(best)

    # Directly optimize the measured maximin formulation at the incumbent
    # diameter.  This can exploit radial freedom omitted by sphere updates.
    """Polish the incumbent with fixed-diameter epigraph SLSQP in a rigid gauge."""
    try:
        from scipy.optimize import minimize

        delta = best[pi] - best[pj]
        ds2 = np.einsum("ij,ij->i", delta, delta)
        diameter2 = float(ds2.max())

        def gauge(p):
            """Center points and impose a deterministic six-degree rigid gauge."""
            p = np.asarray(p, dtype=float).copy()
            p -= p.mean(axis=0, keepdims=True)
            e1_norm = np.linalg.norm(p[0])
            if not np.isfinite(e1_norm) or e1_norm <= 1e-14:
                return p
            e1 = p[0] / e1_norm
            v = p[1] - np.dot(p[1], e1) * e1
            e2_norm = np.linalg.norm(v)
            if not np.isfinite(e2_norm) or e2_norm <= 1e-14:
                return p
            e2 = v / e2_norm
            e3 = np.cross(e1, e2)
            return p @ np.column_stack((e1, e2, e3))

        # The extra starts are deterministic rigidly rotated gauges.  Gauging
        # each one removes translation and the three rotational nullspaces.
        # Enumerate deterministic active-diameter branches.  The incumbent's
        # nearly longest pairs are used as explicit equality constraints so
        # SLSQP starts on distinct diameter faces rather than only rotated
        # copies of the same unconstrained epigraph problem.
        near = np.flatnonzero(ds2 >= diameter2 * (1.0 - 2.0e-5))
        if near.size == 0:
            near = np.asarray([int(np.argmax(ds2))], dtype=int)
        near = near[np.argsort(near, kind="stable")]

        branch_sets = []
        for edge in near:
            branch_sets.append((int(edge),))
            if len(branch_sets) >= 12:
                break
        if len(branch_sets) < 12:
            for a in range(near.size):
                for b in range(a + 1, near.size):
                    branch_sets.append((int(near[a]), int(near[b])))
                    if len(branch_sets) >= 12:
                        break
                if len(branch_sets) >= 12:
                    break

        rotations = (
            np.eye(3),
            np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]),
            np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]),
        )

        def objective(x):
            return -x[-1]

        def objective_jac(x):
            g = np.zeros_like(x)
            g[-1] = -1.0
            return g

        def constraints(x):
            p = x[:-1].reshape(n, 3)
            d = p[pi] - p[pj]
            q = np.einsum("ij,ij->i", d, d)
            lower = q - x[-1]
            upper = diameter2 - q
            if active_branch:
                # Fix the selected longest-pair branch at the incumbent
                # squared diameter while preserving every global inequality.
                active = q[np.asarray(active_branch, dtype=int)] - diameter2
                return np.r_[lower, upper, active]
            return np.r_[lower, upper]

        def constraints_jac(x):
            p = x[:-1].reshape(n, 3)
            d = p[pi] - p[pj]
            jac = np.zeros((2 * m, 3 * n + 1))
            rows = np.arange(m)
            for axis in range(3):
                jac[rows, 3 * pi + axis] = 2.0 * d[:, axis]
                jac[rows, 3 * pj + axis] = -2.0 * d[:, axis]
                jac[m + rows, 3 * pi + axis] = -2.0 * d[:, axis]
                jac[m + rows, 3 * pj + axis] = 2.0 * d[:, axis]
            jac[:m, -1] = -1.0
            return jac

        def gauge_constraint(x):
            p = x[:-1].reshape(n, 3)
            return np.array((p[:, 0].mean(), p[:, 1].mean(), p[:, 2].mean(),
                             p[0, 1], p[0, 2], p[1, 2]))

        def gauge_jac(x):
            jac = np.zeros((6, 3 * n + 1))
            jac[0, 0:3 * n:3] = 1.0 / n
            jac[1, 1:3 * n:3] = 1.0 / n
            jac[2, 2:3 * n:3] = 1.0 / n
            jac[3, 1] = 1.0
            jac[4, 2] = 1.0
            jac[5, 5] = 1.0
            return jac

        for branch_index, active_branch in enumerate(branch_sets):
            rotation = rotations[branch_index % len(rotations)]
            shaped = gauge(best @ rotation.T)
            d = shaped[pi] - shaped[pj]
            q0 = np.einsum("ij,ij->i", d, d)
            # Start strictly inside every lower-distance inequality.  The
            # tiny deterministic slack avoids SLSQP treating an initially
            # active epigraph constraint as an infeasible roundoff artifact.
            qmin0 = float(q0.min())
            x0 = np.r_[shaped.ravel(), qmin0 * (1.0 - 1e-10)]
            result = minimize(
                objective,
                x0,
                jac=objective_jac,
                method="SLSQP",
                bounds=[(None, None)] * (3 * n) + [(0.0, diameter2)],
                constraints=[
                    {"type": "ineq", "fun": constraints,
                     "jac": constraints_jac},
                    {"type": "eq", "fun": gauge_constraint,
                     "jac": gauge_jac},
                ],
                options={"maxiter": 180, "ftol": 1e-11, "disp": False},
            )
            candidate = result.x[:-1].reshape(n, 3)
            candidate_ratio = ratio_of(candidate)
            if np.isfinite(candidate_ratio) and candidate_ratio > best_ratio:
                best_ratio = candidate_ratio
                best = candidate.copy()
    except Exception:
        pass

    return np.asarray(best, dtype=float)