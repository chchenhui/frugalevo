# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """Find a strong 14-point code, then directly polish its true epigraph
    objective with SLSQP under all pairwise squared-distance constraints."""
    a = 1.0 / np.sqrt(3.0)
    directions = np.array(
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
         [a, a, a], [a, a, -a], [a, -a, a], [-a, a, a]]
    )
    base = np.vstack((directions, -directions))
    ii, jj = np.triu_indices(14, 1)
    # Signed pair-to-vertex incidence matrix.  This replaces the Python loop
    # used to accumulate pair forces and makes larger deterministic batches
    # affordable.
    incidence = np.zeros((14, len(ii)))
    incidence[ii, np.arange(len(ii))] = 1.0
    incidence[jj, np.arange(len(ii))] = -1.0
    rng = np.random.default_rng(20260912)

    def ratio(y):
        d2 = np.sum((y[ii] - y[jj]) ** 2, axis=1)
        return np.sqrt(d2.min() / d2.max())

    best = base.copy()
    best_ratio = ratio(best)

    # Keep local perturbations of the good antipodal seed, but also include
    # independent spherical starts.  The latter can enter asymmetric basins
    # which cannot be reached easily from a small perturbation of the code.
    # A wider population improves coverage of the many nearly equivalent
    # nonsmooth basins while retaining a substantial set of local code seeds.
    x = np.repeat(base[None], 192, axis=0)
    x[1:96] += 0.20 * rng.normal(size=x[1:96].shape)
    x[96:] = rng.normal(size=x[96:].shape)
    x[96:] /= np.linalg.norm(x[96:], axis=2)[:, :, None]
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt(np.mean(np.sum(x * x, axis=2), axis=1))[:, None, None]

    # The final sharp stages are important: low powers find a broadly even
    # configuration, whereas high powers resolve the actual active shortest
    # and diameter pairs.
    # Very sharp final continuations are a cheap local polishing phase.  They
    # make the soft objective nearly identical to the evaluator's active
    # shortest-pair / diameter-pair ratio after the lower-power stages have
    # already found well distributed configurations.
    for power, steps in (
        (10, 1200), (20, 1500), (40, 1800), (80, 2200),
        (120, 1800), (240, 2200), (480, 2400), (960, 2600),
        # This is effectively an active-constraint polishing phase: at this
        # sharpness, only genuinely shortest and diameter pairs contribute.
        (1920, 3200),
    ):
        for step in range(steps):
            delta = x[:, ii] - x[:, jj]
            d2 = np.sum(delta * delta, axis=2)
            logd = 0.5 * np.log(np.maximum(d2, 1e-30))

            # Soft extrema give stable gradients before progressively focusing
            # on the actual shortest and longest pair distances.
            near = np.exp(-power * (logd - logd.min(axis=1, keepdims=True)))
            near /= near.sum(axis=1, keepdims=True)
            far = np.exp(power * (logd - logd.max(axis=1, keepdims=True)))
            far /= far.sum(axis=1, keepdims=True)
            force = ((near - far) / np.maximum(d2, 1e-30))[:, :, None] * delta

            grad = np.einsum("bpc,np->bnc", force, incidence, optimize=True)

            grad -= grad.mean(axis=1, keepdims=True)
            grad -= x * (
                np.sum(grad * x, axis=(1, 2)) /
                np.sum(x * x, axis=(1, 2))
            )[:, None, None]
            norm = np.sqrt(np.sum(grad * grad, axis=(1, 2)))
            # Constant normalized steps keep oscillating once only a few
            # active pairs remain.  Decay within each continuation stage,
            # with especially small steps for the nearly nonsmooth stages.
            q = step / max(steps - 1, 1)
            # High-power soft extrema have sparse, rapidly changing active
            # sets, so they require substantially smaller polishing moves.
            if power <= 80:
                initial_step = 0.030
                floor = 0.00035
            elif power <= 240:
                initial_step = 0.010
                floor = 0.00035
            elif power <= 960:
                initial_step = 0.0035
                floor = 0.00035
            else:
                # Active sets switch abruptly at the final temperature, so
                # use a genuinely diminishing subgradient-style step.
                initial_step = 0.0012
                floor = 0.00005
            eta = initial_step * (1.0 - 0.75 * q) + floor
            x += eta * grad / np.maximum(norm[:, None, None], 1e-30)
            x -= x.mean(axis=1, keepdims=True)
            x /= np.sqrt(np.mean(np.sum(x * x, axis=2), axis=1))[:, None, None]

            d2 = np.sum((x[:, ii] - x[:, jj]) ** 2, axis=2)
            values = np.sqrt(d2.min(axis=1) / d2.max(axis=1))
            k = int(np.argmax(values))
            if values[k] > best_ratio:
                best_ratio = values[k]
                best = x[k].copy()

    # Directly optimize the evaluator's actual squared ratio.  First remove
    # Euclidean gauge freedom: a diameter endpoint is placed at the origin,
    # the other endpoint lies on the positive x axis, and a third point is
    # placed in the xy plane.  Only the remaining shape coordinates and the
    # epigraph value t are optimization variables.
    d2best = np.sum((best[ii] - best[jj]) ** 2, axis=1)
    diameter_edge = int(np.argmax(d2best))
    p0, p1 = int(ii[diameter_edge]), int(jj[diameter_edge])
    q = best - best[p0]
    e1 = q[p1] / np.linalg.norm(q[p1])
    candidates = [k for k in range(14) if k != p0 and k != p1]
    p2 = max(candidates, key=lambda k: np.linalg.norm(
        q[k] - np.dot(q[k], e1) * e1))
    v2 = q[p2] - np.dot(q[p2], e1) * e1
    e2 = v2 / np.linalg.norm(v2)
    e3 = np.cross(e1, e2)
    q = q @ np.column_stack((e1, e2, e3))
    q /= np.sqrt(np.max(np.sum((q[ii] - q[jj]) ** 2, axis=1)))

    # Relabeling makes the fixed coordinates especially simple.  Point zero
    # is fixed, point one has only an x coordinate, and point two has x,y.
    order = np.array([p0, p1, p2] +
                     [k for k in range(14) if k not in (p0, p1, p2)])
    q = q[order]
    free = [(1, 0), (2, 0), (2, 1)]
    free += [(k, c) for k in range(3, 14) for c in range(3)]
    free = np.asarray(free, dtype=int)
    nvar = len(free)

    def unpack(z):
        y = np.zeros((14, 3))
        y[free[:, 0], free[:, 1]] = z[:nvar]
        return y

    start_t = np.min(np.sum((q[ii] - q[jj]) ** 2, axis=1)) - 1e-11
    z0 = np.concatenate((q[free[:, 0], free[:, 1]], [start_t]))

    def objective(z):
        return -z[-1]

    def objective_jac(z):
        g = np.zeros(nvar + 1)
        g[-1] = -1.0
        return g

    # The first 91 inequalities are d_ij^2 >= t and the second 91 are
    # d_ij^2 <= 1.  Their exact Jacobian avoids differencing noise at the
    # many active contacts of this nonsmooth packing problem.
    def constraints(z):
        y = unpack(z)
        dd = y[ii] - y[jj]
        ds = np.sum(dd * dd, axis=1)
        return np.concatenate((ds - z[-1], 1.0 - ds))

    def constraints_jac(z):
        y = unpack(z)
        dd = y[ii] - y[jj]
        jac = np.zeros((2 * len(ii), nvar + 1))
        for col, (point, coord) in enumerate(free):
            deriv = 2.0 * dd[:, coord] * (
                (ii == point).astype(float) - (jj == point).astype(float))
            jac[:len(ii), col] = deriv
            jac[len(ii):, col] = -deriv
        jac[:len(ii), -1] = -1.0
        return jac

    try:
        polished = minimize(
            objective, z0, jac=objective_jac, method="SLSQP",
            constraints={"type": "ineq", "fun": constraints,
                         "jac": constraints_jac},
            options={"maxiter": 3000, "ftol": 1e-13, "disp": False},
        )
        if np.all(np.isfinite(polished.x)):
            candidate_gauged = unpack(polished.x)
            # Restore original ordering, then measure the real objective.
            candidate = np.empty_like(candidate_gauged)
            candidate[order] = candidate_gauged
            cd2 = np.sum((candidate[ii] - candidate[jj]) ** 2, axis=1)
            cmax = np.max(cd2)
            if cmax > 0.0:
                candidate /= np.sqrt(cmax)
                cd2 /= cmax
                candidate_ratio = np.sqrt(np.min(cd2))
                # SLSQP can report a line-search warning despite returning a
                # useful feasible iterate, so test geometry rather than status.
                if np.max(cd2) <= 1.0 + 2e-7 and candidate_ratio > best_ratio:
                    best = candidate
                    best_ratio = candidate_ratio
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        pass

    return best


# EVOLVE-BLOCK-END
