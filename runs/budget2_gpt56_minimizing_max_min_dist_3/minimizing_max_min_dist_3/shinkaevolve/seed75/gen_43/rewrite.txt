# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """Return a deterministic 14-point three-dimensional distance code."""
    n = 14
    ii, jj = np.triu_indices(n, 1)
    rng = np.random.default_rng(91724031)

    def normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = x - np.mean(x, axis=0, keepdims=True)
        scale = np.sqrt(np.mean(np.sum(x * x, axis=1)))
        return x / max(float(scale), 1.0e-14)

    def ratio_sq(x: np.ndarray) -> float:
        delta = x[ii] - x[jj]
        q = np.einsum("ij,ij->i", delta, delta)
        return float(np.min(q) / max(float(np.max(q)), 1.0e-15))

    def tangent(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Remove irrelevant translation and radial scale components."""
        v = v - np.mean(v, axis=0, keepdims=True)
        denom = max(float(np.sum(x * x)), 1.0e-15)
        v = v - (np.sum(v * x) / denom) * x
        return v

    def soft_direction(x: np.ndarray, beta: float,
                       active_mix: float) -> np.ndarray:
        """Gradient of a soft log(min distance squared / max distance squared)."""
        delta = x[ii] - x[jj]
        q = np.maximum(np.einsum("ij,ij->i", delta, delta), 1.0e-15)
        logq = np.log(q)

        lo = -beta * logq
        lo -= np.max(lo)
        wlo = np.exp(lo)
        wlo /= np.sum(wlo)

        hi = beta * logq
        hi -= np.max(hi)
        whi = np.exp(hi)
        whi /= np.sum(whi)

        coeff = (wlo - whi) / q

        # Late continuation mixes in equal active-contact forces.  This is
        # less prone than a pure softmax to committing to one arbitrary pair.
        if active_mix > 0.0:
            qmin = float(np.min(q))
            qmax = float(np.max(q))
            close = q <= qmin * 1.030
            far = q >= qmax * 0.970
            contact = np.zeros_like(q)
            contact[close] += 1.0 / max(int(np.count_nonzero(close)), 1)
            contact[far] -= 1.0 / max(int(np.count_nonzero(far)), 1)
            coeff = ((1.0 - active_mix) * coeff +
                     active_mix * contact / q)

        grad = np.zeros_like(x)
        pair_force = 2.0 * coeff[:, None] * delta
        np.add.at(grad, ii, pair_force)
        np.add.at(grad, jj, -pair_force)
        return tangent(x, grad)

    def continuation(x: np.ndarray, schedule,
                     checkpoints: bool = False):
        """Momentum continuation, optionally retaining late-stage states."""
        x = normalize(x)
        velocity = np.zeros_like(x)
        best_x = x.copy()
        best_value = ratio_sq(x)
        saved = []

        for stage, (beta, count, step, mix) in enumerate(schedule):
            momentum = 0.70 if beta < 35.0 else 0.48
            for it in range(count):
                grad = soft_direction(x, beta, mix)
                gnorm = np.sqrt(np.mean(np.sum(grad * grad, axis=1)))
                if gnorm > 1.0e-15:
                    direction = grad / gnorm
                    velocity = momentum * velocity + (1.0 - momentum) * direction
                    velocity = tangent(x, velocity)
                    vnorm = np.sqrt(np.mean(np.sum(velocity * velocity, axis=1)))
                    if vnorm > 1.0e-15:
                        progress = it / max(count - 1, 1)
                        local_step = step * (1.0 - 0.42 * progress)
                        x = normalize(x + local_step * velocity / vnorm)

                if it % 16 == 0 or it == count - 1:
                    value = ratio_sq(x)
                    if value > best_value:
                        best_value = value
                        best_x = x.copy()

            # Checkpoints at about 65%, 80%, and 100% of continuation.
            if checkpoints and stage >= 2:
                saved.append((best_value, best_x.copy()))

        return best_x, best_value, saved

    def active_polish(x: np.ndarray) -> tuple[np.ndarray, float]:
        """Accept-only exact-ratio contact-set refinement."""
        x = normalize(x)
        best_x = x.copy()
        best_value = ratio_sq(x)

        rounds = (
            (0.035, 0.024, 130),
            (0.014, 0.012, 180),
            (0.0045, 0.0052, 230),
            (0.0015, 0.0020, 190),
        )

        for tolerance, base_step, count in rounds:
            for _ in range(count):
                delta = x[ii] - x[jj]
                q = np.maximum(np.einsum("ij,ij->i", delta, delta), 1.0e-15)
                d = np.sqrt(q)
                dmin = float(np.min(d))
                dmax = float(np.max(d))

                near = d <= dmin * (1.0 + tolerance)
                far = d >= dmax * (1.0 - tolerance)
                weights = np.zeros_like(d)

                # Exponentially taper contacts near the boundary of each set.
                wl = np.exp(-(d[near] / dmin - 1.0) /
                            max(0.28 * tolerance, 1.0e-6))
                wh = np.exp(-(1.0 - d[far] / dmax) /
                            max(0.28 * tolerance, 1.0e-6))
                weights[near] = wl / np.sum(wl)
                weights[far] -= wh / np.sum(wh)

                force = np.zeros_like(x)
                pair_force = 2.0 * (weights / q)[:, None] * delta
                np.add.at(force, ii, pair_force)
                np.add.at(force, jj, -pair_force)
                force = tangent(x, force)
                fnorm = np.sqrt(np.mean(np.sum(force * force, axis=1)))
                if fnorm < 1.0e-15:
                    break

                accepted = False
                for mult in (1.0, 0.55, 0.28, 0.13, 0.06):
                    trial = normalize(x + base_step * mult * force / fnorm)
                    value = ratio_sq(trial)
                    if value > best_value + 1.0e-14:
                        x = trial
                        best_x = trial.copy()
                        best_value = value
                        accepted = True
                        break
                if not accepted:
                    break

        return best_x, best_value

    cube = np.array(
        [[a, b, c] for a in (-1.0, 1.0)
         for b in (-1.0, 1.0)
         for c in (-1.0, 1.0)],
        dtype=float,
    )
    axes = np.vstack((np.eye(3), -np.eye(3)))

    def cube_axis(radius: float, noise: float) -> np.ndarray:
        x = np.vstack((cube, radius * axes))
        if noise:
            x = x + noise * rng.normal(size=x.shape)
        return normalize(x)

    k = np.arange(n, dtype=float)
    z = 1.0 - 2.0 * (k + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    golden = np.pi * (3.0 - np.sqrt(5.0))
    fibonacci = np.column_stack((r * np.cos(golden * k),
                                  r * np.sin(golden * k), z))

    seeds = [
        cube_axis(1.34, 0.025),
        cube_axis(1.55, 0.050),
        cube_axis(1.77, 0.075),
        cube_axis(2.00, 0.100),
        normalize(fibonacci),
    ]
    for _ in range(3):
        seeds.append(normalize(rng.normal(size=(n, 3))))

    primary_schedule = (
        (5.0, 95, 0.066, 0.00),
        (13.0, 145, 0.047, 0.00),
        (32.0, 195, 0.031, 0.08),
        (78.0, 240, 0.019, 0.25),
        (190.0, 250, 0.010, 0.46),
    )

    candidates = []
    for seed in seeds:
        point, value, saved = continuation(seed, primary_schedule, True)
        candidates.append((value, point))
        candidates.extend(saved)

    # Preserve several distinct late-stage basins, then branch each using
    # small free-coordinate perturbations rather than radial-only changes.
    candidates.sort(key=lambda item: item[0], reverse=True)
    branch_sources = candidates[:6]

    branch_schedule = (
        (55.0, 90, 0.018, 0.18),
        (125.0, 145, 0.010, 0.40),
        (280.0, 180, 0.0055, 0.62),
    )

    refined = list(candidates[:3])
    for branch_id, (_, source) in enumerate(branch_sources):
        noise = 0.009 + 0.004 * (branch_id % 3)
        start = normalize(source + noise * rng.normal(size=(n, 3)))
        point, value, _ = continuation(start, branch_schedule, False)
        refined.append((value, point))

    refined.sort(key=lambda item: item[0], reverse=True)
    best_value, best = refined[0]

    # Exact nonsmooth finishing is applied to several independent high-quality
    # branches, which is more reliable than polishing only one contact graph.
    for _, point in refined[:4]:
        polished, value = active_polish(point)
        if value > best_value:
            best_value = value
            best = polished

    # Small adaptive accept-only point moves resolve remaining contact swaps.
    x = best.copy()
    value = best_value
    sigma = 0.014
    successes = 0
    for iteration in range(1500):
        index = int(rng.integers(n))
        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-15)
        trial = x.copy()
        trial[index] += sigma * direction
        trial = normalize(trial)
        trial_value = ratio_sq(trial)

        if trial_value > value + 1.0e-14:
            x = trial
            value = trial_value
            successes += 1

        if (iteration + 1) % 125 == 0:
            if successes >= 4:
                sigma = min(0.025, sigma * 1.13)
            elif successes == 0:
                sigma *= 0.56
            else:
                sigma *= 0.80
            successes = 0

    x, value = active_polish(x)
    if value > best_value:
        best = x

    return np.asarray(normalize(best), dtype=float)


# EVOLVE-BLOCK-END