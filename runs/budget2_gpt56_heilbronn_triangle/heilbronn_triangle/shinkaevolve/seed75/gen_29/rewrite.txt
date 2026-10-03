# EVOLVE-BLOCK-START
import numpy as np


_N = 11
_FREE = 8
_HEIGHT = np.sqrt(3.0) * 0.5

# Affine simplex coordinates: (u, v) -> (u + v/2, height*v).
_CORNERS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    dtype=np.float64,
)

_TRIPLES = np.array(
    [(i, j, k)
     for i in range(_N - 2)
     for j in range(i + 1, _N - 1)
     for k in range(j + 1, _N)],
    dtype=np.intp,
)


def _project(x: np.ndarray) -> np.ndarray:
    """Fold arbitrary planar coordinates deterministically into the simplex."""
    y = np.abs(np.asarray(x, dtype=np.float64)).copy()
    s = y[..., 0] + y[..., 1]
    mask = s > 1.0
    if np.any(mask):
        y[mask] /= s[mask, None]
    return y


def _areas(states: np.ndarray) -> np.ndarray:
    """All normalized triangle areas for a batch of eight-free-point states."""
    m = states.shape[0]
    pts = np.empty((m, _N, 2), dtype=np.float64)
    pts[:, :3] = _CORNERS
    pts[:, 3:] = states

    a = pts[:, _TRIPLES[:, 0]]
    b = pts[:, _TRIPLES[:, 1]]
    c = pts[:, _TRIPLES[:, 2]]
    return np.abs(
        (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
        - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    )


def _quality(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Primary value is the actual bottleneck determinant.  The secondary value
    balances its active set without ever outweighing a meaningful bottleneck
    improvement.
    """
    ar = _areas(states)
    low = np.partition(ar, 15, axis=1)[:, :16]
    minimum = low[:, 0]
    # Tail value is only a tie breaker, but remains useful to swarm movement.
    value = minimum + 1.0e-5 * np.mean(low[:, 1:], axis=1)
    return minimum, value


def _better(new_min, new_aux, old_min, old_aux):
    return (new_min > old_min + 1.0e-13) | (
        (np.abs(new_min - old_min) <= 1.0e-13) & (new_aux > old_aux)
    )


def _initial_population(rng: np.random.Generator, size: int) -> np.ndarray:
    """Use several combinatorially different support patterns."""
    pop = rng.dirichlet((1.0, 1.0, 1.0), size=(size, _FREE))[:, :, 1:]

    # A separated triangular pattern provides substantially better starts than
    # entirely uniform samples.
    lattice = np.array(
        [
            [0.125, 0.070], [0.405, 0.055], [0.735, 0.070],
            [0.065, 0.330], [0.350, 0.245], [0.655, 0.230],
            [0.135, 0.605], [0.410, 0.430],
        ],
        dtype=np.float64,
    )
    count = size * 3 // 5
    pop[:count] = _project(
        lattice[None] + rng.normal(0.0, 0.105, size=(count, _FREE, 2))
    )

    # Explicit boundary-rich starts explore a qualitatively separate family.
    edge_count = size // 6
    for q in range(edge_count):
        z = rng.dirichlet((1.2, 1.2, 1.2), size=_FREE)[:, 1:]
        for j in range(4):
            t = rng.uniform(0.08, 0.92)
            edge = (j + q) % 3
            if edge == 0:
                z[j] = (t, 0.0)
            elif edge == 1:
                z[j] = (0.0, t)
            else:
                z[j] = (t, 1.0 - t)
        pop[count + q] = z
    return pop


def _swarm_search(rng: np.random.Generator) -> np.ndarray:
    """
    Particle swarm optimization on complete configurations.

    Unlike pointwise annealing, each particle carries a 16-dimensional
    velocity, so it can cross bottlenecks requiring coupled motion of several
    active points.
    """
    particles = 224
    iterations = 690

    x = _initial_population(rng, particles)
    velocity = rng.normal(0.0, 0.025, size=x.shape)

    pmin, paux = _quality(x)
    personal = x.copy()

    g = int(np.argmax(pmin))
    global_best = x[g].copy()
    global_min = float(pmin[g])
    global_aux = float(paux[g])

    for it in range(iterations):
        f = it / float(iterations - 1)
        inertia = 0.78 - 0.43 * f
        cognitive = 1.32 - 0.26 * f
        social = 0.72 + 1.22 * f

        r1 = rng.random((particles, _FREE, 1))
        r2 = rng.random((particles, _FREE, 1))
        velocity = (
            inertia * velocity
            + cognitive * r1 * (personal - x)
            + social * r2 * (global_best[None] - x)
        )

        # Early isotropic kicks retain diverse triangular order types.
        kick = 0.019 * (1.0 - f) ** 1.5 + 0.00035
        velocity += rng.normal(0.0, kick, size=velocity.shape)
        x = _project(x + velocity)

        cur_min, cur_aux = _quality(x)
        take = _better(cur_min, cur_aux, pmin, paux)
        personal[take] = x[take]
        pmin[take] = cur_min[take]
        paux[take] = cur_aux[take]

        idx = int(np.argmax(pmin))
        if (
            pmin[idx] > global_min + 1.0e-13
            or (
                abs(pmin[idx] - global_min) <= 1.0e-13
                and paux[idx] > global_aux
            )
        ):
            global_best = personal[idx].copy()
            global_min = float(pmin[idx])
            global_aux = float(paux[idx])

        # Rejuvenate persistently weak particles around both the archive and a
        # broad simplex distribution.  This prevents a single swarm collapse.
        if it % 46 == 45 and it < 570:
            worst = np.argsort(pmin)[:particles // 9]
            noise = rng.normal(0.0, 0.080 * (1.0 - f) + 0.014,
                               size=(len(worst), _FREE, 2))
            x[worst] = _project(global_best[None] + noise)
            velocity[worst] = rng.normal(0.0, 0.035, size=velocity[worst].shape)

    return global_best


def _active_refine(rng: np.random.Generator, state: np.ndarray) -> np.ndarray:
    """
    Deterministic batched pattern search on points occurring in the currently
    smallest constraints.  Late acceptance is genuinely lexicographic.
    """
    best = state.copy()
    best_min, best_aux = _quality(best[None])
    best_min = float(best_min[0])
    best_aux = float(best_aux[0])

    directions = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
         [0.707, 0.707], [-0.707, 0.707],
         [0.707, -0.707], [-0.707, -0.707]],
        dtype=np.float64,
    )

    rounds = 480
    batch = 144
    for it in range(rounds):
        frac = it / float(rounds - 1)
        scale = 0.025 * (1.0 - frac) ** 1.85 + 0.00016

        ar = _areas(best[None])[0]
        active = np.argpartition(ar, 19)[:20]
        involved = np.unique(_TRIPLES[active].ravel())
        movable = involved[involved >= 3] - 3
        if len(movable) == 0:
            movable = np.arange(_FREE)

        candidates = np.broadcast_to(best, (batch, _FREE, 2)).copy()

        # Half directional coordinate trials, half coupled active-set trials.
        for row in range(batch):
            p = int(movable[row % len(movable)])
            if row < 64:
                d = directions[(row // len(movable)) % len(directions)]
                candidates[row, p] += scale * d
            else:
                p2 = int(movable[rng.integers(len(movable))])
                candidates[row, p] += rng.normal(0.0, scale, 2)
                if p2 != p:
                    candidates[row, p2] += rng.normal(0.0, 0.58 * scale, 2)
                if row % 7 == 0:
                    selected = rng.choice(movable, min(3, len(movable)), replace=False)
                    candidates[row, selected] += rng.normal(
                        0.0, 0.24 * scale, size=(len(selected), 2)
                    )

        candidates = _project(candidates)
        cm, ca = _quality(candidates)

        # Exact minimum is the dominant final ranking criterion.
        order = np.lexsort((ca, cm))
        winner = int(order[-1])
        if (
            cm[winner] > best_min + 1.0e-13
            or (
                abs(cm[winner] - best_min) <= 1.0e-13
                and ca[winner] > best_aux
            )
        ):
            best = candidates[winner].copy()
            best_min = float(cm[winner])
            best_aux = float(ca[winner])

    return best


def _fallback() -> np.ndarray:
    affine = np.vstack((
        _CORNERS,
        np.array(
            [[0.125, 0.070], [0.405, 0.055], [0.735, 0.070],
             [0.065, 0.330], [0.350, 0.245], [0.655, 0.230],
             [0.135, 0.605], [0.410, 0.430]],
            dtype=np.float64,
        ),
    ))
    result = np.empty_like(affine)
    result[:, 0] = affine[:, 0] + 0.5 * affine[:, 1]
    result[:, 1] = _HEIGHT * affine[:, 1]
    return result


def heilbronn_triangle11() -> np.ndarray:
    """Return eleven deterministic points in the prescribed equilateral triangle."""
    try:
        rng = np.random.default_rng(11031991)

        # Independent swarms explore distinct ordering families; only their
        # best archive is sent to the more expensive active-set polish.
        seeds = [_swarm_search(rng) for _ in range(3)]
        seed_array = np.asarray(seeds)
        values, aux = _quality(seed_array)
        pick = int(np.lexsort((aux, values))[-1])
        free = _active_refine(rng, seed_array[pick])

        affine = np.vstack((_CORNERS, free))
        result = np.empty_like(affine)
        result[:, 0] = affine[:, 0] + 0.5 * affine[:, 1]
        result[:, 1] = _HEIGHT * affine[:, 1]

        if result.shape != (11, 2) or not np.all(np.isfinite(result)):
            return _fallback()
        return result
    except Exception:
        return _fallback()


# EVOLVE-BLOCK-END