# EVOLVE-BLOCK-START
    # Phase 1: population continuation with a smooth log(min/max) force.
    # Each member is independently evolved, allowing substantially different
    # contact graphs to survive until elite selection.
    archive: list[tuple[float, np.ndarray]] = []

    for start_id, start in enumerate(seeds):
        x = start.copy()
        velocity = np.zeros_like(x)
        local_best = x.copy()
        local_value = objective(x)

        iterations = 2050
        for it in range(iterations):
            t = it / (iterations - 1)
            beta = 6.0 * (105.0 / 6.0) ** (t * t)

            delta, q = distances_sq(x)
            qlo = q.min()
            qhi = q.max()

            low_exp = np.exp(-beta * (q - qlo))
            high_exp = np.exp(beta * (q - qhi))
            wlow = low_exp / low_exp.sum()
            whigh = high_exp / high_exp.sum()

            soft_lo = qlo - np.log(np.mean(low_exp)) / beta
            soft_hi = qhi + np.log(np.mean(high_exp)) / beta
            pair_weight = wlow / max(soft_lo, 1.0e-12)
            pair_weight -= whigh / max(soft_hi, 1.0e-12)

            grad = accumulate_gradient(x, pair_weight)

            # Momentum is useful early for escaping nearly symmetric starts;
            # its influence fades before the nonsmooth limiting regime.
            step = 0.021 * (1.0 - 0.72 * t) + 0.0018
            momentum = 0.72 - 0.24 * t
            velocity = momentum * velocity + step * grad
            x = normalize(x + velocity)

            # Deterministic small symmetry breaking only in the global phase.
            if it < 360 and it % 30 == 0:
                kick = rng.normal(size=x.shape)
                kick -= kick.mean(axis=0, keepdims=True)
                x = normalize(x + (0.010 * (1.0 - it / 360.0)) * kick)

            if it % 25 == 0 or it == iterations - 1:
                value = objective(x)
                if value > local_value:
                    local_value = value
                    local_best = x.copy()

        archive.append((local_value, local_best))

    archive.sort(key=lambda item: item[0], reverse=True)
    best_value, best = archive[0][0], archive[0][1].copy()

    # Phase 2: refine several elite basins using persistent lower and upper
    # contact bands.  Contacts that remain active are deliberately emphasized,
    # rather than averaging every temporary band member equally.
    elite = archive[:6]
    refinement_schedule = (
        (0.055, 0.055, 720),
        (0.024, 0.024, 820),
        (0.010, 0.012, 920),
    )

    for elite_id, (_, elite_point) in enumerate(elite):
        for replica in range(2):
            x = elite_point.copy()
            if replica:
                perturb = rng.normal(size=x.shape)
                perturb -= perturb.mean(axis=0, keepdims=True)
                x = normalize(x + (0.0035 + 0.0015 * elite_id) * perturb)

            persistence_low = np.zeros(len(ii))
            persistence_high = np.zeros(len(ii))
            velocity = np.zeros_like(x)
            local_value = objective(x)

            for low_band, high_band, count in refinement_schedule:
                for it in range(count):
                    frac = it / max(count - 1, 1)
                    delta, q = distances_sq(x)
                    qlo = q.min()
                    qhi = q.max()

                    low_relative = q / qlo - 1.0
                    high_relative = qhi / q - 1.0
                    active_low = low_relative <= low_band
                    active_high = high_relative <= high_band

                    persistence_low = 0.945 * persistence_low + active_low
                    persistence_high = 0.945 * persistence_high + active_high

                    # Exponential proximity supplies a smooth ordering inside
                    # a band; persistence prevents fleeting contacts from
                    # overpowering the stable constraint graph.
                    low_shape = np.exp(-low_relative / max(low_band, 1.0e-5))
                    high_shape = np.exp(-high_relative / max(high_band, 1.0e-5))

                    low_weight = low_shape * (0.25 + persistence_low)
                    high_weight = high_shape * (0.25 + persistence_high)

                    # Keep a tiny contribution from all pairs for continuity
                    # when a contact graph changes.
                    low_weight += 0.002 * np.exp(-4.0 * low_relative)
                    high_weight += 0.002 * np.exp(-4.0 * high_relative)

                    low_weight /= low_weight.sum()
                    high_weight /= high_weight.sum()
                    pair_weight = low_weight / qlo - high_weight / qhi

                    grad = accumulate_gradient(x, pair_weight)
                    step0 = 0.010 if low_band > 0.03 else 0.0050
                    step = step0 * (1.0 - 0.55 * frac)

                    velocity = 0.42 * velocity + step * grad
                    candidate = normalize(x + velocity)

                    # Short monotonic safeguard: if the persistent force moves
                    # sharply downhill, retain a damped version instead.
                    if it % 8 == 0:
                        candidate_value = objective(candidate)
                        current_value = objective(x)
                        if candidate_value + 2.0e-5 < current_value:
                            velocity *= 0.30
                            candidate = normalize(x + 0.35 * velocity)

                    x = candidate

                    if it % 10 == 0 or it == count - 1:
                        value = objective(x)
                        if value > local_value:
                            local_value = value
                        if value > best_value:
                            best_value = value
                            best = x.copy()

    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END