# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Return fourteen deterministic points in R^3.

    The staged search uses a portfolio of structured and random seeds,
    inexpensive survivor screening, long continuation of elite basins, and
    exact-ratio evolutionary polishing.  Translation and uniform scale are
    removed after every update.
    """
    n = 14
    pi, pj = np.triu_indices(n, 1)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).copy()
        x -= x.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(scale, 1.0e-14)

    def ratio(x: np.ndarray) -> float:
        d = x[pi] - x[pj]
        q = np.einsum("ij,ij->i", d, d)
        return float(q.min() / q.max())

    def batch_ratio(x: np.ndarray) -> np.ndarray:
        d = x[:, pi, :] - x[:, pj, :]
        q = np.einsum("bij,bij->bi", d, d)
        return q.min(axis=1) / q.max(axis=1)

    def soft_force(x: np.ndarray, beta: float) -> np.ndarray:
        d = x[pi] - x[pj]
        q = np.maximum(np.einsum("ij,ij->i", d, d), 1.0e-15)
        logd = 0.5 * np.log(q)

        low = np.exp(-beta * (logd - logd.min()))
        high = np.exp(beta * (logd - logd.max()))
        low /= low.sum()
        high /= high.sum()

        pair_force = (low - high)[:, None] * d / q[:, None]
        force = np.zeros_like(x)
        np.add.at(force, pi, pair_force)
        np.add.at(force, pj, -pair_force)
        force -= force.mean(axis=0, keepdims=True)
        return force

    def continuation(seed: np.ndarray, steps: int, phase: str) -> np.ndarray:
        """
        Smooth log-distance continuation with exact-objective checkpoints.
        A rollback mechanism keeps sharp surrogate phases from losing a good
        nonsmooth contact graph.
        """
        x = normalize(seed)
        best = x.copy()
        best_score = ratio(x)
        velocity = np.zeros_like(x)
        step_scale = 1.0
        losses = 0

        for it in range(steps):
            t = it / max(steps - 1, 1)

            if phase == "screen":
                beta = 4.0 + 125.0 * t ** 1.55
                step = 0.048 * (1.0 - 0.48 * t)
                momentum = 0.82
                stride = 10
            elif phase == "search":
                beta = 7.0 + 315.0 * t ** 1.45
                step = 0.040 * (1.0 - 0.62 * t)
                momentum = 0.83
                stride = 8
            else:
                beta = 42.0 + 690.0 * t * t
                step = 0.0175 * (1.0 - 0.68 * t)
                momentum = 0.76
                stride = 6

            g = soft_force(x, beta)
            gnorm = np.sqrt(np.mean(np.sum(g * g, axis=1)))
            if gnorm > 1.0e-15:
                g /= gnorm

            velocity = momentum * velocity + (1.0 - momentum) * g
            x = normalize(x + step_scale * step * velocity)

            if it % stride == stride - 1 or it == steps - 1:
                current = ratio(x)
                if current > best_score:
                    best_score = current
                    best = x.copy()
                    losses = 0
                    step_scale = min(1.0, step_scale * 1.035)
                elif current < best_score * 0.9985:
                    losses += 1
                    if losses >= 3:
                        x = best.copy()
                        velocity.fill(0.0)
                        step_scale = max(0.38, step_scale * 0.64)
                        losses = 0
                else:
                    losses = max(losses - 1, 0)

        return best

    def layered(radius: float, height: float, twist: float,
                pole: float, noise: float, seed: int) -> np.ndarray:
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
        x = np.vstack((lower, upper, [[0.0, 0.0, -pole],
                                      [0.0, 0.0, pole]]))
        if noise:
            x += noise * np.random.default_rng(seed).standard_normal((n, 3))
        return x

    def antipodal_seed(seed: int, noise: float) -> np.ndarray:
        """
        Seven unoriented directions expanded to fourteen antipodal points.
        This supplies a qualitatively different high-diameter starting graph.
        """
        rng = np.random.default_rng(seed)
        directions = rng.standard_normal((7, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        x = np.vstack((directions, -directions))
        if noise:
            x += noise * rng.standard_normal((n, 3))
        return x

    def exact_mutation_polish(start: np.ndarray) -> np.ndarray:
        """
        Batched, deterministic hill climbing evaluated only with the exact
        squared-distance ratio used by the evaluator.
        """
        rng = np.random.default_rng(902107)
        incumbent = normalize(start)
        incumbent_score = ratio(incumbent)

        for sigma, rounds, population in (
            (0.030, 10, 8),
            (0.016, 16, 8),
            (0.008, 22, 8),
            (0.0035, 28, 8),
            (0.0014, 28, 8),
        ):
            for _ in range(rounds):
                candidates = incumbent[None, :, :] + sigma * rng.standard_normal(
                    (population, n, 3)
                )
                candidates -= candidates.mean(axis=1, keepdims=True)
                scales = np.sqrt(np.mean(np.sum(candidates * candidates, axis=2),
                                        axis=1))
                candidates /= scales[:, None, None]

                values = batch_ratio(candidates)
                winner = int(np.argmax(values))
                if values[winner] > incumbent_score:
                    incumbent = candidates[winner].copy()
                    incumbent_score = float(values[winner])

        return incumbent

    # ---- Portfolio construction ------------------------------------------------
    seeds = []
    layer_specs = (
        (1.00, 0.43, np.pi / 6.0, 1.30),
        (1.00, 0.49, np.pi / 6.0, 1.36),
        (1.06, 0.46, 0.46, 1.37),
        (0.95, 0.53, 0.57, 1.34),
        (1.08, 0.39, 0.50, 1.29),
        (0.99, 0.58, 0.62, 1.43),
        (1.02, 0.45, 0.52, 1.35),
        (0.97, 0.50, 0.49, 1.38),
    )
    for k, spec in enumerate(layer_specs):
        seeds.append(layered(*spec, noise=0.018, seed=3101 + 71 * k))

    cube = np.array(
        [[a, b, c] for a in (-1.0, 1.0)
                   for b in (-1.0, 1.0)
                   for c in (-1.0, 1.0)],
        dtype=float,
    )
    for scales in ((1.50, 1.66, 1.82), (1.73, 1.49, 1.64)):
        axis = np.diag(scales)
        seeds.append(np.vstack((cube, axis, -axis)))

    for seed in (711, 1931, 8191):
        seeds.append(antipodal_seed(seed, 0.025))

    for seed in (271828, 314159):
        seeds.append(np.random.default_rng(seed).standard_normal((n, 3)))

    # ---- Stage 1: inexpensive broad screening ---------------------------------
    screened = []
    for seed in seeds:
        candidate = continuation(seed, 520, "screen")
        screened.append((ratio(candidate), candidate))

    screened.sort(key=lambda item: item[0], reverse=True)

    # ---- Stage 2: spend the main budget only on survivors ---------------------
    elite = []
    for _, candidate in screened[:7]:
        refined = continuation(candidate, 1080, "search")
        elite.append((ratio(refined), refined))

    elite.sort(key=lambda item: item[0], reverse=True)
    best_value, best = elite[0]

    # Perturb multiple elite contact graphs rather than only the winner.
    restart_rng = np.random.default_rng(481516)
    for _, parent in elite[:3]:
        for amplitude in (0.018, 0.040):
            trial = parent + amplitude * restart_rng.standard_normal((n, 3))
            trial = continuation(trial, 560, "search")
            value = ratio(trial)
            if value > best_value:
                best_value, best = value, trial

    # ---- Stage 3: sharp contact balancing and exact local evolution -----------
    sharp = continuation(best, 1450, "polish")
    sharp_value = ratio(sharp)
    if sharp_value > best_value:
        best_value, best = sharp_value, sharp

    evolved = exact_mutation_polish(best)
    if ratio(evolved) > best_value:
        best = evolved

    return np.asarray(normalize(best), dtype=float)


# EVOLVE-BLOCK-END