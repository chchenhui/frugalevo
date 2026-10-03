# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct fourteen antipodal points from seven optimized unoriented
    directions.  If u_i are unit directions, the returned point set is
    {u_i, -u_i}.  Its diameter is exactly 2, and minimizing the maximum
    absolute line inner product maximizes the required distance ratio.
    """
    rng = np.random.default_rng(784231907)
    lines = 7
    population = 160

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.maximum(norms, 1.0e-14)

    def coherence(batch: np.ndarray) -> np.ndarray:
        gram = np.matmul(batch, np.swapaxes(batch, -1, -2))
        gram = np.abs(gram)
        diagonal = np.arange(lines)
        gram[..., diagonal, diagonal] = 0.0
        return np.max(gram, axis=(-2, -1))

    def make_icosahedral_seed() -> np.ndarray:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        base = np.array(
            [
                [0.0, 1.0, phi],
                [0.0, 1.0, -phi],
                [1.0, phi, 0.0],
                [1.0, -phi, 0.0],
                [phi, 0.0, 1.0],
                [phi, 0.0, -1.0],
            ],
            dtype=float,
        )
        base = normalize_rows(base)
        extra = normalize_rows(np.array([[1.0, 1.0, 1.0]], dtype=float))
        return np.vstack((base, extra))

    # Diverse population of line arrangements.
    directions = normalize_rows(rng.normal(size=(population, lines, 3)))
    seed = make_icosahedral_seed()
    for k in range(32):
        directions[k] = normalize_rows(seed + 0.18 * rng.normal(size=(lines, 3)))

    best_index = int(np.argmin(coherence(directions)))
    best = directions[best_index].copy()
    best_value = float(coherence(best[None, ...])[0])

    # Evolutionary projected contact relaxation.  The low-temperature phases
    # approximate the true maximum coherence while elite cloning provides
    # basin changes unavailable to a single deterministic descent.
    phases = [
        (7.0, 180, 0.105, 0.22),
        (15.0, 220, 0.074, 0.16),
        (34.0, 280, 0.050, 0.105),
        (80.0, 340, 0.031, 0.060),
        (190.0, 420, 0.018, 0.030),
        (480.0, 480, 0.010, 0.012),
    ]

    velocity = np.zeros_like(directions)

    for beta, iterations, initial_step, mutation_scale in phases:
        for iteration in range(iterations):
            dots = np.matmul(directions, np.swapaxes(directions, 1, 2))
            absolute = np.abs(dots)

            diagonal = np.arange(lines)
            absolute[:, diagonal, diagonal] = -np.inf

            logits = beta * absolute
            logits -= np.max(logits, axis=(1, 2), keepdims=True)
            weights = np.exp(logits)
            weights[:, diagonal, diagonal] = 0.0
            weights /= np.sum(weights, axis=(1, 2), keepdims=True)

            # Gradient of a smooth maximum of |u_i dot u_j|.
            signed = weights * np.sign(dots)
            gradient = np.matmul(signed, directions)
            gradient -= (
                np.sum(gradient * directions, axis=2, keepdims=True) * directions
            )

            velocity = 0.72 * velocity - gradient
            speed = np.linalg.norm(velocity, axis=2, keepdims=True)
            velocity *= np.minimum(1.0, 1.7 / np.maximum(speed, 1.0e-14))

            fraction = iteration / max(iterations - 1, 1)
            step = initial_step * (1.0 - 0.72 * fraction)
            directions = normalize_rows(directions + step * velocity)

            # Periodically perform elite-based evolutionary replacement.
            if iteration > 0 and iteration % 55 == 0:
                values = coherence(directions)
                order = np.argsort(values)
                elite_count = 24
                replace_count = population // 3
                parents = directions[order[:elite_count]]
                parent_ids = rng.integers(0, elite_count, size=replace_count)

                children = parents[parent_ids] + mutation_scale * (
                    0.45 + 0.55 * (1.0 - fraction)
                ) * rng.normal(size=(replace_count, lines, 3))
                children = normalize_rows(children)

                worst = order[-replace_count:]
                directions[worst] = children
                velocity[worst] = 0.0

            if iteration % 12 == 0 or iteration == iterations - 1:
                values = coherence(directions)
                index = int(np.argmin(values))
                if values[index] < best_value:
                    best_value = float(values[index])
                    best = directions[index].copy()

    # Exact nonsmooth local polishing.  The accepted moves are judged only by
    # actual coherence, avoiding any final discrepancy from soft objectives.
    x = best.copy()
    current = best_value
    sigma = 0.060

    for trial in range(26000):
        candidate = x.copy()
        line = int(rng.integers(lines))

        # Tangent perturbation on the unit sphere.
        perturbation = rng.normal(size=3)
        perturbation -= np.dot(perturbation, candidate[line]) * candidate[line]
        candidate[line] += sigma * perturbation
        candidate[line] /= max(np.linalg.norm(candidate[line]), 1.0e-14)

        value = float(coherence(candidate[None, ...])[0])
        if value < current:
            x = candidate
            current = value
            if current < best_value:
                best_value = current
                best = x.copy()

        if (trial + 1) % 2600 == 0:
            sigma *= 0.62

    points = np.vstack((best, -best))
    return np.asarray(points, dtype=float)


# EVOLVE-BLOCK-END
