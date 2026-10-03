# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 reproducible points in R^3 with a large minimum-distance to
    diameter ratio.  Translation and uniform scale are normalized away because
    they do not affect the objective.
    """
    n, d = 14, 3
    rng = np.random.default_rng(20240517)
    iu, ju = np.triu_indices(n, 1)

    def normalize(points: np.ndarray) -> np.ndarray:
        points -= points.mean(axis=0)
        points /= np.sqrt(np.mean(points * points))
        return points

    def exact_ratio(points: np.ndarray) -> float:
        delta = points[iu] - points[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(dsq.min() / dsq.max())

    def soft_gradient(points: np.ndarray, beta: float) -> np.ndarray:
        delta = points[iu] - points[ju]
        dsq = np.einsum("ij,ij->i", delta, delta)
        log_dist = 0.5 * np.log(dsq + 1.0e-15)

        low = -beta * log_dist
        low -= low.max()
        w_min = np.exp(low)
        w_min /= w_min.sum()

        high = beta * log_dist
        high -= high.max()
        w_max = np.exp(high)
        w_max /= w_max.sum()

        pair_gradient = (
            (w_min - w_max)[:, None] * delta / (dsq[:, None] + 1.0e-15)
        )
        gradient = np.zeros_like(points)
        np.add.at(gradient, iu, pair_gradient)
        np.add.at(gradient, ju, -pair_gradient)
        return gradient

    best_points = None
    best_ratio = -np.inf

    for restart in range(12):
        points = normalize(rng.normal(size=(n, d)))
        first = np.zeros_like(points)
        second = np.zeros_like(points)

        # Broad smooth search establishes a good global contact graph.
        for step in range(8000):
            beta = 5.0 + 115.0 * step / 7999.0
            gradient = soft_gradient(points, beta)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            learning_rate = 0.028 * (0.10 + 0.90 * (1.0 - step / 8000.0))
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            normalize(points)

        # Sharper smoothing resolves the final nearest/farthest contacts.
        first.fill(0.0)
        second.fill(0.0)
        for step in range(2200):
            beta = 120.0 + 480.0 * step / 2199.0
            gradient = soft_gradient(points, beta)

            first = 0.9 * first + 0.1 * gradient
            second = 0.999 * second + 0.001 * gradient * gradient
            mhat = first / (1.0 - 0.9 ** (step + 1))
            vhat = second / (1.0 - 0.999 ** (step + 1))

            learning_rate = 0.006 * (1.0 - 0.55 * step / 2200.0)
            points += learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)
            normalize(points)

        # Finish with exact-objective acceptance, but retain the sharp
        # soft-contact force rather than using only isotropic random kicks.
        # Tangent noise and single-point proposals let active contact graphs
        # exchange while every accepted move improves the true hard ratio.
        current_ratio = exact_ratio(points)
        proposal_scale = 0.010
        stalled_batches = 0

        for polish_step in range(650):
            direction = soft_gradient(points, 520.0 + 180.0 * (polish_step % 5))
            direction -= direction.mean(axis=0, keepdims=True)
            direction_norm = np.sqrt(np.mean(direction * direction))
            if direction_norm > 1.0e-14:
                direction /= direction_norm
            else:
                direction.fill(0.0)

            candidate_points = points
            candidate_ratio = current_ratio

            # A pure force step is useful when the final soft objective still
            # agrees with the current hard contacts.
            trial = points + proposal_scale * direction
            normalize(trial)
            trial_ratio = exact_ratio(trial)
            if trial_ratio > candidate_ratio:
                candidate_ratio = trial_ratio
                candidate_points = trial

            # Perturb force-guided directions in the normalized configuration
            # space, avoiding wasteful translation and scale components.
            for _ in range(3):
                noise = rng.normal(size=(n, d))
                noise -= noise.mean(axis=0, keepdims=True)
                noise /= np.sqrt(np.mean(noise * noise))
                trial = points + proposal_scale * (direction + 0.38 * noise)
                normalize(trial)
                trial_ratio = exact_ratio(trial)
                if trial_ratio > candidate_ratio:
                    candidate_ratio = trial_ratio
                    candidate_points = trial

            # Local moves are especially effective when one point participates
            # in several nearly active shortest or longest pairs.
            for _ in range(2):
                index = rng.integers(n)
                trial = points.copy()
                local_noise = rng.normal(size=d)
                local_noise /= np.linalg.norm(local_noise)
                trial[index] += proposal_scale * (
                    direction[index] + 0.55 * local_noise
                )
                normalize(trial)
                trial_ratio = exact_ratio(trial)
                if trial_ratio > candidate_ratio:
                    candidate_ratio = trial_ratio
                    candidate_points = trial

            if candidate_ratio > current_ratio + 1.0e-14:
                points = candidate_points
                current_ratio = candidate_ratio
                proposal_scale = min(0.022, proposal_scale * 1.045)
                stalled_batches = 0
            else:
                stalled_batches += 1
                if stalled_batches % 14 == 0:
                    proposal_scale *= 0.60
                    if proposal_scale < 1.5e-6:
                        break

        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()

    return best_points


# EVOLVE-BLOCK-END