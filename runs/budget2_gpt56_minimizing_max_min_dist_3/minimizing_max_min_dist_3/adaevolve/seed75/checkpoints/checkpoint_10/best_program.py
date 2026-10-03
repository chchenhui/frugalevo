# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Optimize 14 unconstrained 3D points from antipodal symmetric seeds.

    A batched deterministic smooth min-distance/max-distance optimization is
    used.  Translation and uniform scale are removed after each update, since
    both are irrelevant to the distance ratio.
    """
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
    x = np.repeat(base[None], 72, axis=0)
    x[1:48] += 0.20 * rng.normal(size=x[1:48].shape)
    x[48:] = rng.normal(size=x[48:].shape)
    x[48:] /= np.linalg.norm(x[48:], axis=2)[:, :, None]
    x -= x.mean(axis=1, keepdims=True)
    x /= np.sqrt(np.mean(np.sum(x * x, axis=2), axis=1))[:, None, None]

    # The final sharp stages are important: low powers find a broadly even
    # configuration, whereas high powers resolve the actual active shortest
    # and diameter pairs.
    for power, steps in (
        (10, 1200), (20, 1500), (40, 1800), (80, 2200),
        (120, 1800), (240, 2200),
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
            initial_step = 0.030 if power <= 80 else 0.010
            eta = initial_step * (1.0 - 0.70 * q) + 0.001
            x += eta * grad / np.maximum(norm[:, None, None], 1e-30)
            x -= x.mean(axis=1, keepdims=True)
            x /= np.sqrt(np.mean(np.sum(x * x, axis=2), axis=1))[:, None, None]

            d2 = np.sum((x[:, ii] - x[:, jj]) ** 2, axis=2)
            values = np.sqrt(d2.min(axis=1) / d2.max(axis=1))
            k = int(np.argmax(values))
            if values[k] > best_ratio:
                best_ratio = values[k]
                best = x[k].copy()

    return best


# EVOLVE-BLOCK-END
