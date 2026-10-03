# EVOLVE-BLOCK-START
import numpy as np


_SQRT3 = float(np.sqrt(3.0))
_HEIGHT = 0.5 * _SQRT3


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct eleven points in the unit equilateral triangle.

    Search is performed in reference-simplex coordinates (u,v), where
    u >= 0, v >= 0, u+v <= 1.  Determinants in those coordinates equal
    triangle areas normalized by the containing equilateral triangle area.
    """
    try:
        if hasattr(heilbronn_triangle11, "_cached"):
            return heilbronn_triangle11._cached.copy()

        rng = np.random.default_rng(11031987)
        n = 11
        free = 8
        triples = np.asarray(
            [(i, j, k)
             for i in range(n)
             for j in range(i + 1, n)
             for k in range(j + 1, n)],
            dtype=np.intp,
        )

        fixed = np.array(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
            dtype=float,
        )

        def project(x):
            """Project batched (..., 8, 2) arrays into the reference simplex."""
            z = np.asarray(x, dtype=float).copy()
            z = np.maximum(z, 0.0)
            total = z[..., 0] + z[..., 1]
            mask = total > 1.0
            if np.any(mask):
                z[mask] /= total[mask, None]
            return z

        def all_areas(states):
            """Absolute normalized determinants for a state batch."""
            m = len(states)
            p = np.empty((m, 11, 2), dtype=float)
            p[:, :3] = fixed
            p[:, 3:] = states
            a = p[:, triples[:, 0]]
            b = p[:, triples[:, 1]]
            c = p[:, triples[:, 2]]
            return np.abs(
                (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
            )

        def score_from_areas(areas, temperature):
            """
            Smooth lower-tail objective.  It rewards the minimum determinant
            while also increasing the group of nearly active constraints.
            """
            low = np.min(areas, axis=1)
            shifted = (areas - low[:, None]) / temperature
            soft = low - temperature * np.log(np.exp(-shifted).sum(axis=1))
            return 0.58 * low + 0.42 * soft

        def state_from_seed(seed):
            """Generate a dispersed simplex layout from a random candidate pool."""
            pool = rng.dirichlet((1.05, 1.05, 1.05), size=seed)[:, 1:]
            selected = [pool[rng.integers(seed)]]
            while len(selected) < free:
                old = np.asarray(selected)
                d = pool[:, None, :] - old[None, :, :]
                distance = np.sum(d * d, axis=2)
                selected.append(pool[np.argmax(np.min(distance, axis=1))])
            return np.asarray(selected)

        # A moderate population of complete configurations is much more able
        # to change the active-triple topology than point-by-point search.
        pop_size = 112
        population = np.empty((pop_size, free, 2), dtype=float)

        hand = np.array(
            [
                (0.19, 0.08), (0.49, 0.07), (0.78, 0.08),
                (0.10, 0.30), (0.39, 0.27), (0.68, 0.22),
                (0.20, 0.56), (0.48, 0.44),
            ],
            dtype=float,
        )
        population[0] = hand
        for i in range(1, pop_size):
            population[i] = state_from_seed(90 + (i % 5) * 35)

        population = project(population)
        areas = all_areas(population)
        archive = np.empty((0, free, 2), dtype=float)

        best_state = population[np.argmax(np.min(areas, axis=1))].copy()
        best_value = float(np.max(np.min(areas, axis=1)))

        # JADE-like current-to-pbest differential evolution.  Unlike classic
        # DE, p-best mutation preserves strong geometric motifs while archive
        # differences inject successful displaced layouts.
        generations = 880
        mean_f = 0.63
        mean_cr = 0.80

        for generation in range(generations):
            ratio = generation / float(generations - 1)
            temperature = 0.0048 * (1.0 - ratio) ** 1.7 + 0.00016
            scores = score_from_areas(areas, temperature)
            order = np.argsort(scores)
            elite_count = max(8, int(pop_size * (0.28 - 0.16 * ratio)))
            elite = order[-elite_count:]

            f_values = np.clip(
                mean_f + 0.13 * rng.standard_cauchy(pop_size),
                0.18, 0.96,
            )
            cr_values = np.clip(
                rng.normal(mean_cr, 0.12, pop_size),
                0.12, 0.98,
            )

            pbest = elite[rng.integers(elite_count, size=pop_size)]
            r1 = rng.integers(pop_size, size=pop_size)
            same = r1 == np.arange(pop_size)
            while np.any(same):
                r1[same] = rng.integers(pop_size, size=np.count_nonzero(same))
                same = r1 == np.arange(pop_size)

            source = population
            if len(archive):
                union = np.concatenate((population, archive), axis=0)
            else:
                union = population
            r2 = rng.integers(len(union), size=pop_size)

            mutant = (
                population
                + 0.55 * (population[pbest] - population)
                + f_values[:, None, None] * (population[r1] - union[r2])
            )

            # A small number of coordinated Gaussian trials avoids excessive
            # reliance on affine DE directions in a converged population.
            shake = rng.random(pop_size) < 0.12 * (1.0 - ratio)
            if np.any(shake):
                mutant[shake] += rng.normal(
                    0.0, 0.055 * (1.0 - ratio) + 0.004,
                    size=(np.count_nonzero(shake), free, 2),
                )

            cross = rng.random((pop_size, free, 2)) < cr_values[:, None, None]
            mandatory = rng.integers(free * 2, size=pop_size)
            cross.reshape(pop_size, -1)[np.arange(pop_size), mandatory] = True
            trial = np.where(cross, mutant, population)
            trial = project(trial)

            trial_areas = all_areas(trial)
            trial_scores = score_from_areas(trial_areas, temperature)
            accepted = trial_scores >= scores

            if np.any(accepted):
                displaced = population[accepted].copy()
                archive = np.concatenate((archive, displaced), axis=0)
                if len(archive) > pop_size:
                    archive = archive[
                        rng.choice(len(archive), size=pop_size, replace=False)
                    ]
                population[accepted] = trial[accepted]
                areas[accepted] = trial_areas[accepted]

                sf = f_values[accepted]
                mean_f = 0.88 * mean_f + 0.12 * (
                    np.sum(sf * sf) / max(np.sum(sf), 1.0e-12)
                )
                mean_cr = 0.88 * mean_cr + 0.12 * np.mean(cr_values[accepted])

            minima = np.min(areas, axis=1)
            leader = int(np.argmax(minima))
            if minima[leader] > best_value:
                best_value = float(minima[leader])
                best_state = population[leader].copy()

        # Exact local maximin polish after determining a promising orientation
        # pattern.  Several topologically distinct elite states are attempted.
        try:
            from scipy.optimize import minimize

            final_minima = np.min(areas, axis=1)
            finalists = np.argsort(final_minima)[-5:]

            def dets(uv):
                q = uv[triples]
                return (
                    (q[:, 1, 0] - q[:, 0, 0]) *
                    (q[:, 2, 1] - q[:, 0, 1])
                    - (q[:, 1, 1] - q[:, 0, 1]) *
                    (q[:, 2, 0] - q[:, 0, 0])
                )

            for idx in finalists:
                initial = population[idx]
                uv0 = np.vstack((fixed, initial))
                signed = dets(uv0)
                signs = np.sign(signed)
                signs[signs == 0.0] = 1.0
                initial_t = float(np.min(np.abs(signed)))

                def constraints(w):
                    uv = np.vstack((fixed, w[:-1].reshape(free, 2)))
                    tail = w[:-1].reshape(free, 2)
                    return np.concatenate((
                        signs * dets(uv) - w[-1],
                        tail.ravel(),
                        1.0 - tail.sum(axis=1),
                    ))

                result = minimize(
                    lambda w: -w[-1],
                    np.r_[initial.ravel(), initial_t],
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * (free * 2) + [(0.0, 1.0)],
                    constraints={"type": "ineq", "fun": constraints},
                    options={"maxiter": 1100, "ftol": 2.0e-12, "disp": False},
                )

                if result.success and np.all(np.isfinite(result.x)):
                    candidate = result.x[:-1].reshape(free, 2)
                    candidate = project(candidate)
                    candidate_value = float(np.min(all_areas(candidate[None])[0]))
                    if candidate_value > best_value + 1.0e-11:
                        best_value = candidate_value
                        best_state = candidate.copy()
        except Exception:
            pass

        output = np.empty((11, 2), dtype=float)
        output[:3] = (
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, _HEIGHT),
        )
        output[3:, 0] = best_state[:, 0] + 0.5 * best_state[:, 1]
        output[3:, 1] = _HEIGHT * best_state[:, 1]

        heilbronn_triangle11._cached = output.copy()
        return output

    except Exception:
        return np.array(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (0.5, _HEIGHT),
                (0.19, 0.08),
                (0.49, 0.07),
                (0.78, 0.08),
                (0.10, 0.30),
                (0.39, 0.27),
                (0.68, 0.22),
                (0.20, 0.56),
                (0.48, 0.44),
            ],
            dtype=float,
        )


# EVOLVE-BLOCK-END