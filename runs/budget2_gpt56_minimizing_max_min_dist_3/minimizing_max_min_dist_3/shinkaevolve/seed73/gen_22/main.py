# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Build 14 reproducible points in R^3 by optimizing a population of
    diameter-normalized packing candidates.

    The search works in two stages:
      1. broad population exploration with a smooth min/max-distance objective;
      2. selection, diversification, and sharper contact polishing.

    Translation and scale are removed after every update because they are
    invariances of the objective.
    """
    n_points = 14
    dimension = 3
    population_size = 36
    rng = np.random.default_rng(2024051701)

    upper_i, upper_j = np.triu_indices(n_points, 1)
    pair_count = upper_i.size

    # Signed point/pair incidence matrix.  It converts pair forces into
    # point gradients without Python-level pair accumulation.
    incidence = np.zeros((n_points, pair_count), dtype=np.float64)
    incidence[upper_i, np.arange(pair_count)] = 1.0
    incidence[upper_j, np.arange(pair_count)] = -1.0

    def normalize(configurations: np.ndarray) -> np.ndarray:
        configurations = configurations - configurations.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(configurations * configurations, axis=(1, 2)))
        return configurations / rms[:, None, None]

    def pairwise_squared(configurations: np.ndarray) -> np.ndarray:
        differences = configurations[:, upper_i] - configurations[:, upper_j]
        return np.einsum("bpd,bpd->bp", differences, differences)

    def exact_scores(configurations: np.ndarray) -> np.ndarray:
        squared = pairwise_squared(configurations)
        return squared.min(axis=1) / squared.max(axis=1)

    def record_best(
        configurations: np.ndarray,
        best_points: np.ndarray,
        best_score: float,
    ) -> tuple[np.ndarray, float]:
        scores = exact_scores(configurations)
        winner = int(np.argmax(scores))
        if scores[winner] > best_score:
            return configurations[winner].copy(), float(scores[winner])
        return best_points, best_score

    def optimize_population(
        configurations: np.ndarray,
        steps: int,
        beta_start: float,
        beta_end: float,
        learning_start: float,
        learning_end: float,
        best_points: np.ndarray,
        best_score: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Adam ascent on softmin(log distance) - softmax(log distance).

        As beta rises, the soft objective converges toward the actual
        closest-pair versus farthest-pair ratio.
        """
        first_moment = np.zeros_like(configurations)
        second_moment = np.zeros_like(configurations)

        # Moment resets make the optimizer re-adapt when the contact measure
        # changes from a broad average to a sparse minimax contact set.
        reset_levels = (50.0, 120.0, 220.0)
        reset_index = 0
        while reset_index < len(reset_levels) and reset_levels[reset_index] <= beta_start:
            reset_index += 1
        moment_age = 0

        for step in range(steps):
            fraction = step / max(steps - 1, 1)
            beta = beta_start + (beta_end - beta_start) * fraction
            learning_rate = learning_start + (learning_end - learning_start) * fraction

            if reset_index < len(reset_levels) and beta >= reset_levels[reset_index]:
                first_moment.fill(0.0)
                second_moment.fill(0.0)
                moment_age = 0
                reset_index += 1

            if beta >= 220.0:
                learning_rate *= 0.48
                second_decay = 0.9995
            elif beta >= 120.0:
                learning_rate *= 0.62
                second_decay = 0.9985
            elif beta >= 50.0:
                learning_rate *= 0.82
                second_decay = 0.997
            else:
                second_decay = 0.995

            differences = configurations[:, upper_i] - configurations[:, upper_j]
            squared = np.einsum("bpd,bpd->bp", differences, differences)
            log_distance = 0.5 * np.log(squared + 1.0e-15)

            low_logits = -beta * log_distance
            low_logits -= low_logits.max(axis=1, keepdims=True)
            low_weights = np.exp(low_logits)
            low_weights /= low_weights.sum(axis=1, keepdims=True)

            high_logits = beta * log_distance
            high_logits -= high_logits.max(axis=1, keepdims=True)
            high_weights = np.exp(high_logits)
            high_weights /= high_weights.sum(axis=1, keepdims=True)

            pair_forces = (
                (low_weights - high_weights)[:, :, None]
                * differences
                / (squared[:, :, None] + 1.0e-15)
            )
            gradient = np.einsum("np,bpd->bnd", incidence, pair_forces)

            first_moment = 0.90 * first_moment + 0.10 * gradient
            second_moment = (
                second_decay * second_moment
                + (1.0 - second_decay) * gradient * gradient
            )

            moment_age += 1
            bias1 = 1.0 - 0.90 ** moment_age
            bias2 = 1.0 - second_decay ** moment_age
            configurations += (
                learning_rate
                * (first_moment / bias1)
                / (np.sqrt(second_moment / bias2) + 1.0e-8)
            )
            configurations = normalize(configurations)

            # Preserve strong exact candidates rather than trusting only
            # the annealed surrogate at the end of a stage.
            if step % 200 == 199 or step == steps - 1:
                best_points, best_score = record_best(
                    configurations, best_points, best_score
                )

        return configurations, best_points, best_score

    # Stage 1: broad independent exploration.
    population = normalize(rng.normal(size=(population_size, n_points, dimension)))
    best_points = population[0].copy()
    best_score = -np.inf

    population, best_points, best_score = optimize_population(
        population,
        steps=4400,
        beta_start=6.0,
        beta_end=95.0,
        learning_start=0.030,
        learning_end=0.010,
        best_points=best_points,
        best_score=best_score,
    )

    # Stage 2 input: retain elite geometries and make several controlled
    # descendants of each, retaining exact copies of the elite as anchors.
    scores = exact_scores(population)
    elite_count = 9
    elite = population[np.argsort(scores)[-elite_count:]]

    descendants = []
    for parent in elite:
        descendants.append(parent)
        descendants.append(normalize((parent + 0.018 * rng.normal(size=parent.shape))[None])[0])
        descendants.append(normalize((parent + 0.036 * rng.normal(size=parent.shape))[None])[0])
        descendants.append(normalize((parent + 0.060 * rng.normal(size=parent.shape))[None])[0])

    population = np.asarray(descendants, dtype=np.float64)
    population, best_points, best_score = optimize_population(
        population,
        steps=6500,
        beta_start=70.0,
        beta_end=280.0,
        learning_start=0.014,
        learning_end=0.0025,
        best_points=best_points,
        best_score=best_score,
    )

    # Final exact comparison includes all polished candidates.
    best_points, best_score = record_best(population, best_points, best_score)

    # Monotone exact-score polish of the strongest configuration.  Sharp
    # alternating soft contact sets supply useful directions, while
    # backtracking accepts only improvements to the true objective.
    polished = best_points.copy()
    polished_score = best_score
    for iteration in range(900):
        differences = polished[upper_i] - polished[upper_j]
        squared = np.einsum("pd,pd->p", differences, differences) + 1.0e-15
        log_distance = 0.5 * np.log(squared)
        beta = (110.0, 180.0, 300.0, 480.0)[iteration % 4]

        low = np.exp(-beta * (log_distance - log_distance.min()))
        low /= low.sum()
        high = np.exp(beta * (log_distance - log_distance.max()))
        high /= high.sum()

        forces = (low - high)[:, None] * differences / squared[:, None]
        direction = incidence @ forces
        direction -= direction.mean(axis=0, keepdims=True)
        direction_norm = np.sqrt(np.mean(direction * direction))
        if direction_norm < 1.0e-14:
            continue

        direction /= direction_norm
        step_size = 0.022
        for _ in range(8):
            candidate = normalize((polished + step_size * direction)[None])[0]
            candidate_squared = np.einsum(
                "pd,pd->p",
                candidate[upper_i] - candidate[upper_j],
                candidate[upper_i] - candidate[upper_j],
            )
            candidate_score = float(candidate_squared.min() / candidate_squared.max())
            if candidate_score > polished_score + 1.0e-14:
                polished = candidate
                polished_score = candidate_score
                break
            step_size *= 0.5

    if polished_score > best_score:
        best_points = polished
    return np.asarray(best_points, dtype=np.float64)


# EVOLVE-BLOCK-END