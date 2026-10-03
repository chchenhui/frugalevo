# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Construct 14 three-dimensional points with a large minimum-distance to
    diameter ratio.  Translation and uniform scale are removed internally,
    since they do not affect the objective.
    """
    n = 14
    rng = np.random.default_rng(20250308)

    ii, jj = np.triu_indices(n, 1)
    pair_count = len(ii)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = x - np.mean(x, axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(scale, 1.0e-12)

    def exact_ratio_sq(x: np.ndarray) -> float:
        delta = x[ii] - x[jj]
        dsq = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(dsq) / np.max(dsq))

    def fibonacci_seed(offset: float) -> np.ndarray:
        k = np.arange(n, dtype=float)
        z = 1.0 - 2.0 * (k + 0.5) / n
        radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        golden = np.pi * (3.0 - np.sqrt(5.0))
        theta = golden * k + offset
        return normalize(np.column_stack((radius * np.cos(theta),
                                          radius * np.sin(theta), z)))

    def ring_seed(phase: float) -> np.ndarray:
        # Two staggered hexagonal rings and two displaced cap points.
        a = np.arange(6) * (2.0 * np.pi / 6.0)
        h = 0.43
        r = np.sqrt(1.0 - h * h)
        upper = np.column_stack((r * np.cos(a), r * np.sin(a),
                                 np.full(6, h)))
        lower = np.column_stack((r * np.cos(a + phase),
                                 r * np.sin(a + phase),
                                 np.full(6, -h)))
        caps = np.array([[0.17, -0.08, 1.0],
                         [-0.17, 0.08, -1.0]])
        caps /= np.linalg.norm(caps, axis=1, keepdims=True)
        return normalize(np.vstack((upper, lower, caps)))

    seeds = [
        fibonacci_seed(0.0),
        fibonacci_seed(0.37),
        ring_seed(np.pi / 6.0),
        ring_seed(np.pi / 5.0),
    ]

    # Add reproducible spherical starts, useful because the contact graph of
    # the best finite packing need not share the symmetry of a ring seed.
    for _ in range(3):
        x = rng.normal(size=(n, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        seeds.append(normalize(x))

    best = seeds[0].copy()
    best_value = exact_ratio_sq(best)

    # Smooth min/max continuation.  The logarithmic mean-exponential form is
    # numerically stable and has gradients that are weighted contact forces.
    iterations = 3600
    for seed_number, start in enumerate(seeds):
        x = start.copy()

        # Exponential-memory estimates of the lower and upper contact graph.
        # They distinguish stable limiting pairs from pairs which only enter a
        # soft band transiently during a large update.
        near_persistence = np.zeros(pair_count)
        far_persistence = np.zeros(pair_count)

        for it in range(iterations):
            t = it / (iterations - 1)
            beta = 7.0 * (80.0 / 7.0) ** t

            diff = x[ii] - x[jj]
            dsq = np.einsum("ij,ij->i", diff, diff)

            # Soft minimum and soft maximum of squared pair distances.
            # Shift both log-sum-exp calculations as well as their weights.
            # The unshifted scalar expressions can overflow at the high-beta
            # end of continuation, particularly for initially uneven starts.
            dmin_shift = np.min(dsq)
            amin = -beta * (dsq - dmin_shift)
            wmin = np.exp(amin)
            wmin /= np.sum(wmin)
            soft_min = dmin_shift - np.log(np.mean(np.exp(amin))) / beta

            dmax_shift = np.max(dsq)
            amax = beta * (dsq - dmax_shift)
            wmax = np.exp(amax)
            wmax /= np.sum(wmax)
            soft_max = dmax_shift + np.log(np.mean(np.exp(amax))) / beta

            # Persistent, violation-weighted contact forces.  The temperature
            # narrows with continuation, so the final force is concentrated on
            # the actual diameter and minimum-distance contact graphs instead
            # of being diluted by temporary near-band members.
            contact_tau = 0.18 - 0.145 * t
            low_gap = np.maximum(dsq / max(dmin_shift, 1.0e-12) - 1.0, 0.0)
            high_gap = np.maximum(
                (dmax_shift - dsq) / max(dmax_shift, 1.0e-12), 0.0
            )
            low_signal = np.exp(-low_gap / contact_tau)
            high_signal = np.exp(-high_gap / contact_tau)

            memory = 0.90 + 0.07 * t
            near_persistence = memory * near_persistence + (1.0 - memory) * low_signal
            far_persistence = memory * far_persistence + (1.0 - memory) * high_signal

            # A current violation always contributes, while repeated contacts
            # acquire an additional coefficient through their persistence.
            wmin = low_signal * (0.20 + near_persistence)
            wmin /= np.sum(wmin)
            wmax = high_signal * (0.20 + far_persistence)
            wmax /= np.sum(wmax)

            # Gradient of the corresponding persistent approximation to
            # log(soft_min / soft_max).
            pair_weight = wmin / max(soft_min, 1.0e-10)
            pair_weight -= wmax / max(soft_max, 1.0e-10)

            grad = np.zeros_like(x)
            force = 2.0 * pair_weight[:, None] * diff
            np.add.at(grad, ii, force)
            np.add.at(grad, jj, -force)

            # Large early moves reshape the global contact graph; later moves
            # refine the limiting distances without destabilizing the diameter.
            step = 0.020 * (1.0 - t) + 0.0025
            x = normalize(x + step * grad)

            # A small deterministic annealing kick helps distinct starts leave
            # symmetric but inferior contact arrangements.
            if it < 500 and (it % 25 == 0):
                kick = rng.normal(size=x.shape)
                kick -= np.mean(kick, axis=0, keepdims=True)
                x = normalize(x + (0.010 * (1.0 - it / 500.0)) * kick)

            if it % 20 == 0 or it == iterations - 1:
                value = exact_ratio_sq(x)
                if value > best_value:
                    best_value = value
                    best = x.copy()

    return np.asarray(best, dtype=float)


# EVOLVE-BLOCK-END