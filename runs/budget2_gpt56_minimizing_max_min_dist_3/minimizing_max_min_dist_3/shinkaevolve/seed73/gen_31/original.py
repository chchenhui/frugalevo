# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Construct a reproducible high-quality diameter-normalized 14-point packing."""
    n, d = 14, 3
    pair_i, pair_j = np.triu_indices(n, 1)
    pair_count = pair_i.size
    rng = np.random.default_rng(42014)

    incidence = np.zeros((n, pair_count), dtype=np.float64)
    pair_index = np.arange(pair_count)
    incidence[pair_i, pair_index] = 1.0
    incidence[pair_j, pair_index] = -1.0

    def normalize(points: np.ndarray) -> np.ndarray:
        points = points - points.mean(axis=0, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(points * points, axis=1)))
        return points / max(rms, 1.0e-15)

    def score(points: np.ndarray) -> float:
        delta = points[pair_i] - points[pair_j]
        squared = np.einsum("pd,pd->p", delta, delta)
        return float(squared.min() / squared.max())

    def normalize_rows(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def line_seed() -> np.ndarray:
        """Generate a diverse near-Grassmannian antipodal seed."""
        m = 7
        upper = np.triu_indices(m, 1)
        vectors = normalize_rows(rng.normal(size=(m, d)))

        for beta in (5.0, 16.0, 50.0, 120.0):
            for _ in range(100):
                dots = vectors @ vectors.T
                values = dots[upper]
                weights = np.exp(beta * (np.abs(values) - np.max(np.abs(values))))
                weights /= weights.sum()

                signed = weights * np.sign(values)
                coupling = np.zeros((m, m))
                coupling[upper] = signed
                coupling[(upper[1], upper[0])] = signed

                gradient = coupling @ vectors
                gradient -= (
                    np.sum(gradient * vectors, axis=1, keepdims=True) * vectors
                )
                vectors = normalize_rows(vectors - 0.085 * gradient)

        seed = np.vstack((vectors, -vectors))
        seed += 0.012 * rng.normal(size=seed.shape)
        return normalize(seed)

    best_points = None
    best_score = -np.inf

    def record(points: np.ndarray) -> None:
        nonlocal best_points, best_score
        value = score(points)
        if value > best_score:
            best_score = value
            best_points = points.copy()

    def optimize(points: np.ndarray, iterations: int,
                 beta_start: float, beta_end: float,
                 learning_start: float, learning_end: float) -> np.ndarray:
        """Adam ascent on softmin(log distance) minus softmax(log distance)."""
        nonlocal best_points, best_score

        first = np.zeros_like(points)
        second = np.zeros_like(points)

        for step in range(iterations):
            fraction = step / max(iterations - 1, 1)
            beta = beta_start + (beta_end - beta_start) * fraction
            learning = learning_start + (learning_end - learning_start) * fraction

            delta = points[pair_i] - points[pair_j]
            squared = np.einsum("pd,pd->p", delta, delta) + 1.0e-15
            log_distance = 0.5 * np.log(squared)

            low_logits = -beta * log_distance
            low_weights = np.exp(low_logits - low_logits.max())
            low_weights /= low_weights.sum()

            high_logits = beta * log_distance
            high_weights = np.exp(high_logits - high_logits.max())
            high_weights /= high_weights.sum()

            forces = (
                (low_weights - high_weights)[:, None] * delta / squared[:, None]
            )
            gradient = incidence @ forces

            first = 0.90 * first + 0.10 * gradient
            second = 0.995 * second + 0.005 * gradient * gradient
            t = step + 1
            points += learning * (first / (1.0 - 0.90 ** t)) / (
                np.sqrt(second / (1.0 - 0.995 ** t)) + 1.0e-8
            )
            points = normalize(points)

            if step % 80 == 0 or step == iterations - 1:
                record(points)

        return points

    # Fast unrestricted exploration, supplemented by line-packing basins.
    for _ in range(12):
        points = normalize(rng.normal(size=(n, d)))
        optimize(points, 1750, 5.0, 95.0, 0.038, 0.009)

    for _ in range(5):
        optimize(line_seed(), 1900, 8.0, 150.0, 0.032, 0.006)

    # Controlled local descendants of the strongest exact configuration.
    anchors = best_points.copy()
    for noise, steps in ((0.025, 1200), (0.012, 1600), (0.005, 1800)):
        candidate = normalize(anchors + noise * rng.normal(size=(n, d)))
        optimize(candidate, steps, 55.0, 250.0, 0.015, 0.0025)
        anchors = best_points.copy()

    # Exact-score backtracking polish with alternating sharp soft contact sets.
    polished = best_points.copy()
    polished_score = best_score

    for iteration in range(1000):
        delta = polished[pair_i] - polished[pair_j]
        squared = np.einsum("pd,pd->p", delta, delta) + 1.0e-15
        log_distance = 0.5 * np.log(squared)
        beta = (90.0, 160.0, 280.0, 450.0)[iteration % 4]

        low = np.exp(-beta * (log_distance - log_distance.min()))
        low /= low.sum()
        high = np.exp(beta * (log_distance - log_distance.max()))
        high /= high.sum()

        forces = (low - high)[:, None] * delta / squared[:, None]
        direction = incidence @ forces
        direction -= direction.mean(axis=0, keepdims=True)

        norm = np.sqrt(np.mean(np.sum(direction * direction, axis=1)))
        if norm < 1.0e-14:
            continue
        direction /= norm

        step_size = 0.028
        for _ in range(8):
            candidate = normalize(polished + step_size * direction)
            candidate_score = score(candidate)
            if candidate_score > polished_score + 1.0e-14:
                polished = candidate
                polished_score = candidate_score
                break
            step_size *= 0.5

    if polished_score > best_score:
        best_points = polished
        best_score = polished_score

    # Directly minimize the squared diameter with the minimum squared
    # separation fixed at one.  This is the exact max-min formulation after
    # scale normalization, and is especially effective once the preceding
    # continuation has identified a good contact graph.
    try:
        from scipy.optimize import minimize

        def constrained_polish(seed: np.ndarray) -> np.ndarray:
            seed = seed / np.sqrt(
                np.min(
                    np.einsum(
                        "pd,pd->p",
                        seed[pair_i] - seed[pair_j],
                        seed[pair_i] - seed[pair_j],
                    )
                )
            )
            initial_d2 = np.einsum(
                "pd,pd->p",
                seed[pair_i] - seed[pair_j],
                seed[pair_i] - seed[pair_j],
            )
            x0 = np.concatenate((seed.ravel(), [initial_d2.max() * (1.0 + 1.0e-10)]))

            def constraints(x: np.ndarray) -> np.ndarray:
                p = x[:-1].reshape(n, d)
                delta = p[pair_i] - p[pair_j]
                d2 = np.einsum("pd,pd->p", delta, delta)
                return np.concatenate((d2 - 1.0, x[-1] - d2))

            def constraint_jacobian(x: np.ndarray) -> np.ndarray:
                p = x[:-1].reshape(n, d)
                delta = p[pair_i] - p[pair_j]
                jac = np.zeros((2 * pair_count, n * d + 1))
                rows = np.arange(pair_count)
                for coordinate in range(d):
                    derivative = 2.0 * delta[:, coordinate]
                    jac[rows, d * pair_i + coordinate] = derivative
                    jac[rows, d * pair_j + coordinate] = -derivative
                    jac[pair_count + rows, d * pair_i + coordinate] = -derivative
                    jac[pair_count + rows, d * pair_j + coordinate] = derivative
                jac[pair_count + rows, -1] = 1.0
                return jac

            def centered(x: np.ndarray) -> np.ndarray:
                return x[:-1].reshape(n, d).sum(axis=0)

            center_jacobian = np.zeros((d, n * d + 1))
            for point in range(n):
                center_jacobian[:, d * point:d * point + d] = np.eye(d)

            result = minimize(
                lambda x: x[-1],
                x0,
                jac=lambda x: np.r_[np.zeros(n * d), 1.0],
                method="SLSQP",
                constraints=(
                    {"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
                    {"type": "eq", "fun": centered, "jac": lambda x: center_jacobian},
                ),
                options={"maxiter": 700, "ftol": 2.0e-13, "disp": False},
            )
            return result.x[:-1].reshape(n, d) if result.success else seed

        # Rescaling after pass one makes the next solve start with an exact
        # lower-bound contact set rather than accumulated numerical slack.
        constrained = constrained_polish(best_points)
        constrained = constrained / np.sqrt(
            np.min(
                np.einsum(
                    "pd,pd->p",
                    constrained[pair_i] - constrained[pair_j],
                    constrained[pair_i] - constrained[pair_j],
                )
            )
        )
        constrained = constrained_polish(constrained)
        constrained = normalize(constrained)
        constrained_score = score(constrained)
        if np.isfinite(constrained_score) and constrained_score > best_score:
            best_points = constrained
            best_score = constrained_score
    except Exception:
        pass

    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END