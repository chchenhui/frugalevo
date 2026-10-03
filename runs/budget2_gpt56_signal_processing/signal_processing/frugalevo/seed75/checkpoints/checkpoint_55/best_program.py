"""
Batch-capable adaptive filtering for volatile non-stationary signals.

The returned output is aligned to the requested window right edge:
output[j] estimates input[j + window_size - 1].
"""
import numpy as np

try:
    from scipy.signal import savgol_filter, butter, sosfiltfilt
    from scipy.interpolate import UnivariateSpline
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _finite_signal(x):
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    if x.size == 0:
        return x
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if not np.all(good):
        ii = np.arange(x.size)
        x[~good] = np.interp(ii[~good], ii[good], x[good])
    return x


def adaptive_filter(x, window_size=20):
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError("Input signal length must be >= window_size")
    return np.convolve(x, np.ones(window_size) / float(window_size), mode="valid")


def _odd_span(value, n):
    largest = n if n % 2 else n - 1
    if largest < 3:
        return largest
    return min(max(3, int(round(value)) | 1), largest)


def _fallback_smooth(x, span):
    if span < 3:
        return x.copy()
    p = span // 2
    return np.convolve(
        np.pad(x, p, mode="edge"),
        np.ones(span) / float(span),
        mode="valid",
    )


def _reversals(z):
    d = np.diff(z)
    if d.size < 2:
        return 0
    eps = 1e-8 * (np.std(z) + 1.0)
    s = np.sign(d)
    for i in range(len(s)):
        if abs(d[i]) <= eps:
            s[i] = s[i - 1] if i else 0.0
    return int(np.sum(s[1:] * s[:-1] < 0))


def _safe_corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def _spline_candidates(x):
    """Reconstruct selected chirplets with one globally coupled continuity solve."""
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    n = int(x.size)
    if n < 12:
        return []

    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-10:
        return [x.copy()]

    fractions = (0.44, 0.56, 0.68)
    freqs = (0.55, 1.05, 1.75, 2.65)
    rates = (-0.12, -0.04, 0.0, 0.04, 0.12, 0.20)
    scale2 = scale * scale + 1e-12
    candidates = []

    for fraction in fractions:
        length = min(n, max(12, int(round(fraction * n))))
        starts = tuple(dict.fromkeys((
            0, max(0, (n - length) // 2), max(0, n - length)
        )))
        blocks = []

        for start in starts:
            stop = min(n, start + length)
            m = int(stop - start)
            if m < 8:
                continue

            u = np.linspace(-1.0, 1.0, m)
            y = x[start:stop]
            edge = 0.22 + 0.78 * np.sin(np.linspace(0.0, np.pi, m)) ** 2
            sqrt_w = np.sqrt(edge)
            best_value = np.inf
            best_basis = None
            best_coef = None

            for rate in rates:
                for freq in freqs:
                    phase = 2.0 * np.pi * (freq * u + 0.5 * rate * u * u)
                    basis = np.column_stack((
                        np.ones(m), u,
                        np.sin(phase), np.cos(phase),
                        np.sin(2.0 * phase), np.cos(2.0 * phase),
                    ))
                    weighted = basis * sqrt_w[:, None]
                    lhs = weighted.T @ weighted
                    rhs = weighted.T @ (y * sqrt_w)
                    penalty = np.diag((0.0, 0.0, 0.018, 0.018, 0.030, 0.030))
                    try:
                        coef = np.linalg.solve(lhs + penalty, rhs)
                    except (np.linalg.LinAlgError, ValueError):
                        continue

                    fit = basis @ coef
                    residual = float(np.mean(edge * (y - fit) ** 2))
                    curvature = float(np.mean(np.diff(fit, 2) ** 2))
                    value = (
                        np.log(residual + 1e-12)
                        + 5.2 * np.log(m) / m
                        + 0.026 * curvature / scale2
                    )
                    if value < best_value:
                        best_value = value
                        best_basis = basis
                        best_coef = coef

            if best_basis is not None:
                blocks.append((int(start), int(stop), edge, best_basis, best_coef))

        if not blocks:
            continue

        # Jointly refit the independently selected atoms.  Data terms retain
        # the original edge weighting; contact rows suppress value and slope
        # discontinuities between neighboring overlapping blocks.
        q = 6 * len(blocks)
        lhs = np.zeros((q, q), dtype=float)
        rhs = np.zeros(q, dtype=float)
        for j, (start, stop, edge, basis, coef) in enumerate(blocks):
            sl = slice(6 * j, 6 * j + 6)
            weighted = basis * np.sqrt(edge)[:, None]
            lhs[sl, sl] += weighted.T @ weighted
            rhs[sl] += weighted.T @ (x[start:stop] * np.sqrt(edge))
            lhs[sl, sl] += np.diag((1e-8, 1e-8, 0.018, 0.018, 0.030, 0.030))

        for j in range(len(blocks) - 1):
            a, b = blocks[j], blocks[j + 1]
            left = max(a[0], b[0])
            right = min(a[1], b[1])
            if right <= left:
                continue
            point = (left + right - 1) // 2

            def row_at(block, index, derivative=False):
                start, stop, _, basis, _ = block
                m = stop - start
                k = min(m - 1, max(0, index - start))
                if not derivative:
                    return basis[k].copy()
                if 0 < k < m - 1:
                    return (basis[k + 1] - basis[k - 1]) * 0.5
                return basis[min(m - 1, k + 1)] - basis[max(0, k - 1)]

            rv = row_at(a, point)
            cv = row_at(b, point)
            rd = row_at(a, point, True)
            cd = row_at(b, point, True)

            # Normalize contact penalties by a robust local residual scale.
            # This keeps continuity strong on quiet overlaps without allowing
            # large-amplitude samples to dominate the coupled solve.
            overlap = x[left:right]
            local_scale = float(np.median(
                np.abs(np.diff(overlap))
            )) if overlap.size > 1 else 0.0
            global_scale = float(np.std(x)) + 1e-12
            residual_scale = max(local_scale, 0.10 * global_scale, 1e-8)
            contact_gain = min(4.0, max(0.5, global_scale / residual_scale))
            value_weight = contact_gain
            slope_weight = 0.12 * max(8, right - left) * contact_gain

            for row, weight in ((rv, value_weight), (rd, slope_weight)):
                joint = np.zeros(q, dtype=float)
                joint[6*j:6*j+6] = row
                lhs += weight * np.outer(joint, joint)
            for row, weight in ((cv, value_weight), (cd, slope_weight)):
                joint = np.zeros(q, dtype=float)
                joint[6*(j+1):6*(j+1)+6] = row
                lhs += weight * np.outer(joint, joint)

            lhs[6*j:6*j+6, 6*(j+1):6*(j+1)+6] -= (
                value_weight * np.outer(rv, cv)
            )
            lhs[6*(j+1):6*(j+1)+6, 6*j:6*j+6] -= (
                value_weight * np.outer(cv, rv)
            )
            lhs[6*j:6*j+6, 6*(j+1):6*(j+1)+6] -= (
                slope_weight * np.outer(rd, cd)
            )
            lhs[6*(j+1):6*(j+1)+6, 6*j:6*j+6] -= (
                slope_weight * np.outer(cd, rd)
            )

        try:
            joint_coef = np.linalg.solve(lhs + 1e-8 * np.eye(q), rhs)
        except (np.linalg.LinAlgError, ValueError):
            joint_coef = np.concatenate([b[4] for b in blocks])

        result = np.zeros(n, dtype=float)
        weights = np.zeros(n, dtype=float)
        for j, (start, stop, edge, basis, _) in enumerate(blocks):
            fit = basis @ joint_coef[6*j:6*j+6]
            result[start:stop] += fit * edge
            weights[start:stop] += edge
        if np.all(weights > 0):
            z = result / weights
            if np.all(np.isfinite(z)):
                candidates.append(z)

    return candidates


def _quality(x, z):
    scale = np.std(x) + 1e-8
    mae = np.mean(np.abs(x - z)) / scale
    return mae + 0.022 * _reversals(z) - 0.10 * max(0.0, _safe_corr(x, z))


def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = _finite_signal(x)
    n = len(x)
    if n < window_size:
        raise ValueError("Input signal length must be >= window_size")
    if n <= 3:
        return x[window_size - 1:].copy()

    candidates = [x.copy()]

    spans = []
    for mult in (0.75, 1.0, 1.5, 2.0, 3.0):
        sp = _odd_span(mult * window_size, n)
        if sp >= 3 and sp not in spans:
            spans.append(sp)

    for sp in spans:
        if _HAVE_SCIPY:
            order = 2 if sp < 7 else 3
            try:
                candidates.append(savgol_filter(x, sp, order, mode="interp"))
            except Exception:
                candidates.append(_fallback_smooth(x, sp))
        else:
            candidates.append(_fallback_smooth(x, sp))

    if _HAVE_SCIPY and n >= 16:
        for cutoff in (0.055, 0.085, 0.13, 0.20):
            try:
                sos = butter(3, cutoff, output="sos")
                candidates.append(sosfiltfilt(sos, x))
            except Exception:
                pass

    # Bounded global harmonic-frequency refinement.  Shared frequencies and
    # phases avoid the slope discontinuities introduced by local block fits.
    if n >= 24:
        t = np.linspace(-1.0, 1.0, n)
        edge = 0.30 + 0.70 * np.sin(np.linspace(0.0, np.pi, n)) ** 2
        affine = np.column_stack((np.ones(n), t))
        aw = affine * np.sqrt(edge)[:, None]
        try:
            detrend_coef = np.linalg.solve(
                aw.T @ aw + np.diag((1e-8, 1e-8)),
                aw.T @ (x * np.sqrt(edge)),
            )
            residual = x - affine @ detrend_coef
            spec = np.abs(np.fft.rfft(residual))
            spec[0] = 0.0
            if spec.size > 2:
                spec[-1] *= 0.5

            # Select dominant peaks while rejecting nearby FFT lobes.  A
            # three-bin exclusion is wide enough to avoid duplicate leakage
            # peaks but still preserves distinct low-frequency components.
            order = np.argsort(spec[1:])[::-1] + 1
            selected = []
            separation = 3
            for k in order:
                if len(selected) >= 6:
                    break
                if all(abs(int(k) - int(q)) >= separation for q in selected):
                    selected.append(int(k))

            refined = []
            sample_index = np.arange(n, dtype=float)
            weighted_residual = residual * np.sqrt(edge)
            for k in selected:
                center = float(k) / n
                stencil = center + np.linspace(-0.35, 0.35, 9) / n
                stencil = stencil[(stencil > 1.0 / n) & (stencil < 0.49)]
                if stencil.size == 0:
                    continue
                best_freq, best_power = float(stencil[0]), -np.inf
                for freq in stencil:
                    phase = 2.0 * np.pi * freq * sample_index
                    projection = np.column_stack((
                        np.sin(phase), np.cos(phase)
                    ))
                    weighted_projection = projection * np.sqrt(edge)[:, None]
                    gram = weighted_projection.T @ weighted_projection
                    response = weighted_projection.T @ weighted_residual
                    try:
                        power = float(response @ np.linalg.solve(
                            gram + 1e-8 * np.eye(2), response
                        ))
                    except (np.linalg.LinAlgError, ValueError):
                        continue
                    if power > best_power:
                        best_power = power
                        best_freq = float(freq)
                refined.append(best_freq)

            for count in (2, 4, 6):
                freqs = refined[:count]
                if not freqs:
                    continue
                cols = [np.ones(n), t]
                for freq in freqs:
                    phase = 2.0 * np.pi * freq * np.arange(n)
                    cols.extend((np.sin(phase), np.cos(phase)))
                basis = np.column_stack(cols)
                weighted = basis * np.sqrt(edge)[:, None]
                ridge = np.eye(basis.shape[1]) * 0.012
                ridge[:2, :2] = 1e-8 * np.eye(2)
                coef = np.linalg.solve(
                    weighted.T @ weighted + ridge,
                    weighted.T @ (x * np.sqrt(edge)),
                )
                harmonic = basis @ coef
                if np.all(np.isfinite(harmonic)):
                    candidates.append(harmonic)

                    # Replace the broadband affine trend with a robust trend
                    # estimated from median slopes across a wide set of pairs.
                    pair_slopes = []
                    for gap in (max(2, n // 12), max(2, n // 8),
                                max(2, n // 5)):
                        if gap < n:
                            pair_slopes.extend(
                                (x[gap:] - x[:-gap]) / float(gap)
                            )
                    if pair_slopes:
                        robust_slope = float(np.median(pair_slopes))
                        robust_intercept = float(
                            np.median(x - robust_slope * np.arange(n))
                        )
                        trend = robust_intercept + robust_slope * np.arange(n)
                        trend_candidate = harmonic.copy()
                        trend_candidate -= affine @ coef[:2]
                        trend_candidate += trend
                        if np.all(np.isfinite(trend_candidate)):
                            candidates.append(trend_candidate)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            pass

    candidates.extend(_spline_candidates(x))

    valid = []
    for z in candidates:
        z = np.asarray(z, dtype=float)
        if z.shape != x.shape or not np.all(np.isfinite(z)):
            continue
        corr = _safe_corr(z, x)
        if corr >= 0.20 or np.std(z) < 1e-12:
            valid.append((_quality(x, z), z))

    best = min(valid, key=lambda item: item[0])[1] if valid else x
    return np.asarray(best[window_size - 1:], dtype=float).copy()


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, length)
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean_signal += 0.1 * t * np.sin(0.2 * t)
    clean_signal += np.cumsum(rng.randn(length) * 0.05)
    noisy_signal = clean_signal + rng.normal(0, noise_level, length)
    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level
        )
    else:
        noisy_signal = _finite_signal(noisy_signal)
        clean_signal = None

    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if clean_signal is None:
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }

    delay = window_size - 1
    aligned_clean = clean_signal[delay:]
    aligned_noisy = noisy_signal[delay:]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    correlation = _safe_corr(filtered_signal, aligned_clean) if m > 1 else 0.0
    before = np.var(aligned_noisy - aligned_clean)
    after = np.var(filtered_signal - aligned_clean)
    reduction = (before - after) / before if before > 0 else 0.0

    return {
        "filtered_signal": filtered_signal,
        "clean_signal": aligned_clean,
        "noisy_signal": aligned_noisy,
        "correlation": correlation,
        "noise_reduction": reduction,
        "signal_length": m,
    }


if __name__ == "__main__":
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")