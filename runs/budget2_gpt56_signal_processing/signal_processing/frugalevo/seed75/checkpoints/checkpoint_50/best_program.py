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
    """Fit three-scale overlap-add chirplet projections with bounded 24-model search."""
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    n = int(x.size)
    if n < 12:
        return []

    scale = float(np.std(x))
    if not np.isfinite(scale) or scale < 1e-10:
        return [x.copy()]

    # Frequencies are cycles over the normalized block coordinate.  Using
    # several-cycle atoms matches oscillatory content while the quadratic
    # phase terms allow modest local frequency drift.
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
        result = np.zeros(n, dtype=float)
        weights = np.zeros(n, dtype=float)

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
            best_fit = None

            # Four frequencies times six rates gives exactly 24 bounded
            # deterministic local projections per block.
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
                        best_fit = fit

            if best_fit is not None:
                result[start:stop] += best_fit * edge
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