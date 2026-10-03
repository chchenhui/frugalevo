# EVOLVE-BLOCK-START
import numpy as np


_N = 14
_II, _JJ = np.triu_indices(_N, 1)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(x * x, axis=1)))
    return x / max(float(rms), 1.0e-15)


def _normalize_batch(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(x * x, axis=2), axis=1))
    return x / np.maximum(rms, 1.0e-15)[:, None, None]


def _score(x: np.ndarray) -> float:
    d = x[_II] - x[_JJ]
    q = np.einsum("ij,ij->i", d, d)
    return float(q.min() / q.max())


def _batch_score(x: np.ndarray) -> np.ndarray:
    d = x[:, _II] - x[:, _JJ]
    q = np.einsum("bpj,bpj->bp", d, d)
    return q.min(axis=1) / q.max(axis=1)


def _active_direction(x: np.ndarray, fraction: float) -> np.ndarray:
    """
    Rank-based contact balancing direction.

    Unlike a log-sum-exp surrogate, only explicitly selected short and long
    contact classes contribute.  Their triangular rank profiles avoid abrupt
    changes when several nearly tied distances exchange active status.
    """
    delta = x[_II] - x[_JJ]
    q = np.einsum("ij,ij->i", delta, delta)
    count = len(q)

    contacts = max(4, min(24, int(round(fraction * count))))
    short_order = np.argsort(q)[:contacts]
    long_order = np.argsort(q)[-contacts:]

    weights = np.zeros(count, dtype=float)
    profile = np.linspace(1.0, 0.15, contacts)

    # Repel short edges in a scale-aware manner and pull in long edges.
    weights[short_order] += profile / np.maximum(q[short_order], 1.0e-14)
    weights[long_order] -= profile / np.maximum(q[long_order], 1.0e-14)

    force = 2.0 * weights[:, None] * delta
    direction = np.zeros_like(x)
    np.add.at(direction, _II, force)
    np.add.at(direction, _JJ, -force)

    direction -= direction.mean(axis=0, keepdims=True)
    radial = np.sum(direction * x) / max(np.sum(x * x), 1.0e-15)
    direction -= radial * x

    norm = np.sqrt(np.mean(np.sum(direction * direction, axis=1)))
    if norm > 1.0e-14:
        direction /= norm
    return direction


def _layer_seed(radius: float, height: float, twist: float, pole: float) -> np.ndarray:
    a = np.arange(6, dtype=float) * (np.pi / 3.0)
    lower = np.column_stack((
        radius * np.cos(a),
        radius * np.sin(a),
        -height * np.ones(6),
    ))
    upper = np.column_stack((
        radius * np.cos(a + twist),
        radius * np.sin(a + twist),
        height * np.ones(6),
    ))
    return np.vstack((lower, upper, [[0.0, 0.0, -pole], [0.0, 0.0, pole]]))


def _seeds(rng: np.random.Generator) -> list:
    bases = [
        _layer_seed(0.946, 0.405, np.pi / 6.0, 1.000),
        _layer_seed(0.920, 0.385, np.pi / 6.0, 0.980),
        _layer_seed(0.975, 0.430, np.pi / 6.0, 1.030),
        _layer_seed(0.950, 0.415, 0.455, 1.000),
        _layer_seed(0.955, 0.400, 0.585, 1.000),
    ]

    cube = np.array(
        [[a, b, c] for a in (-1.0, 1.0)
         for b in (-1.0, 1.0)
         for c in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.array([
        [1.55, 0.0, 0.0], [-1.55, 0.0, 0.0],
        [0.0, 1.67, 0.0], [0.0, -1.67, 0.0],
        [0.0, 0.0, 1.60], [0.0, 0.0, -1.60],
    ])
    bases.append(np.vstack((cube, axes)))

    phi = 0.5 * (1.0 + np.sqrt(5.0))
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        [0, 0, 1.75], [0, 0, -1.75],
    ], dtype=float)
    bases.append(ico)

    result = []
    for base in bases:
        result.append(_normalize(base + 0.018 * rng.standard_normal(base.shape)))
    for _ in range(5):
        result.append(_normalize(rng.standard_normal((_N, 3))))
    return result


def _search(seed: np.ndarray, rng: np.random.Generator, iterations: int) -> np.ndarray:
    x = _normalize(seed)
    value = _score(x)
    best = x.copy()
    best_value = value

    velocity = np.zeros_like(x)
    scales = np.array([1.0, 0.55, 0.25, 0.10], dtype=float)
    failures = 0

    for step in range(iterations):
        t = step / max(iterations - 1, 1)

        # Broad contact classes first, increasingly sparse active sets later.
        fraction = 0.27 - 0.205 * (t ** 0.72)
        direction = _active_direction(x, fraction)

        momentum = 0.34 if failures < 4 else 0.08
        velocity = momentum * velocity + (1.0 - momentum) * direction
        vnorm = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
        if vnorm > 1.0e-14:
            velocity /= vnorm

        step_size = 0.050 * (1.0 - 0.84 * t) + 0.00055
        trials = x[None, :, :] + step_size * scales[:, None, None] * velocity

        # Controlled transverse alternatives are only used during the broad
        # active-set stage, where active-contact rearrangements are useful.
        if step % 31 == 0 and t < 0.72:
            noise = rng.standard_normal((2, _N, 3))
            noise -= noise.mean(axis=1, keepdims=True)
            noise -= (
                np.sum(noise * x[None, :, :], axis=(1, 2))
                / np.sum(x * x)
            )[:, None, None] * x[None, :, :]
            norm = np.sqrt(np.mean(np.sum(noise * noise, axis=2), axis=1))
            noise /= np.maximum(norm, 1.0e-14)[:, None, None]
            trials = np.concatenate((
                trials,
                x[None, :, :] + 0.16 * step_size * noise,
            ))

        trials = _normalize_batch(trials)
        values = _batch_score(trials)
        choice = int(np.argmax(values))

        if values[choice] > value + 1.0e-15:
            x = trials[choice]
            value = float(values[choice])
            failures = 0
            if value > best_value:
                best_value = value
                best = x.copy()
        else:
            failures += 1
            velocity *= 0.35

            # An exact-best restart changes the active contact assignment
            # without accepting any deterioration.
            if failures >= 14:
                x = best.copy()
                value = best_value
                velocity.fill(0.0)
                failures = 0

    return best


def min_max_dist_dim3_14() -> np.ndarray:
    """Return fourteen finite three-dimensional points."""
    rng = np.random.default_rng(918274)

    screened = []
    for seed in _seeds(rng):
        candidate = _search(seed, rng, 520)
        screened.append((_score(candidate), candidate))

    screened.sort(key=lambda item: item[0], reverse=True)

    best_value = -np.inf
    best_points = None
    for _, seed in screened[:6]:
        candidate = _search(seed, rng, 1180)
        value = _score(candidate)
        if value > best_value:
            best_value = value
            best_points = candidate

    # A final exact-monotone sparse-contact pass.
    final = _search(best_points, rng, 620)
    if _score(final) > best_value:
        best_points = final

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END