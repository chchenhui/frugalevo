"""
Batch-capable adaptive filtering for volatile non-stationary signals.

The public output is aligned with the right edge of the requested window:
output[j] estimates input sample j + window_size - 1.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter, butter, sosfiltfilt
    from scipy.interpolate import PchipInterpolator
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
        indices = np.arange(x.size)
        x[~good] = np.interp(indices[~good], indices[good], x[good])
    return x


def adaptive_filter(x, window_size=20):
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(x, kernel, mode="valid")


def _odd_span(value, n):
    span = int(round(value))
    span = max(5, span | 1)
    largest = n if n % 2 else n - 1
    return min(span, largest)


def _reversals(z):
    d = np.diff(z)
    if d.size < 2:
        return 0
    # Ignore numerical near-zero slopes rather than manufacturing a sign change.
    eps = 1e-10 * (np.std(z) + 1.0)
    s = np.sign(d)
    for i in range(1, len(s)):
        if abs(d[i]) <= eps:
            s[i] = s[i - 1]
    return int(np.sum(s[1:] * s[:-1] < 0))


def _fallback_smooth(x, span):
    kernel = np.ones(span, dtype=float) / span
    pad = span // 2
    return np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Select a zero-phase, multiscale smoother using a reversal/fit proxy."""
    x = _finite_signal(x)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n < 5:
        return x[window_size - 1:].copy()

    candidates = [x.copy()]
    spans = []
    for multiple in (1.0, 1.5, 2.0, 3.0):
        span = _odd_span(multiple * window_size, n)
        if span not in spans and span >= 5:
            spans.append(span)

    if _HAVE_SCIPY:
        for span in spans:
            order = 3 if span >= 7 else 2
            candidates.append(savgol_filter(x, span, order, mode="interp"))
        # These broad cutoffs offer a complementary smoother family to
        # local-polynomial fitting while remaining zero-phase.
        if n > 15:
            for cutoff in (0.06, 0.10, 0.16):
                try:
                    sos = butter(3, cutoff, btype="lowpass", output="sos")
                    candidates.append(sosfiltfilt(sos, x))
                except ValueError:
                    pass
    else:
        for span in spans:
            candidates.append(_fallback_smooth(x, span))

    xstd = np.std(x)
    best = candidates[0]
    best_score = np.inf
    for candidate in candidates:
        candidate = np.nan_to_num(candidate, nan=0.0, posinf=0.0, neginf=0.0)
        if xstd > 1e-12 and np.std(candidate) > 1e-12:
            corr = np.corrcoef(candidate, x)[0, 1]
        else:
            corr = 1.0 if xstd <= 1e-12 else 0.0
        # Prevent an almost-flat low-pass candidate from winning solely
        # by having no extrema.
        if not np.isfinite(corr) or corr < 0.35:
            continue
        mae = np.mean(np.abs(candidate - x))
        proxy = 0.4 * min(mae, 2.0) + 0.6 * min(_reversals(candidate) / 50.0, 2.0)
        if proxy < best_score:
            best_score = proxy
            best = candidate

    """Reconstruct a trend from extrema persistent across three smoothing scales."""
    raw = np.asarray(best[window_size - 1:], dtype=float).copy()
    if raw.size < 2 or not _HAVE_SCIPY:
        return raw

    nraw = raw.size
    radius = max(1, window_size // 3)
    residual = raw - savgol_filter(
        raw, _odd_span(min(window_size, nraw), nraw),
        2 if nraw < 7 else 3, mode="interp"
    )
    prominence = 1.5 * (1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8)

    scales = [
        _odd_span(max(5, window_size // 2), nraw),
        _odd_span(max(5, window_size), nraw),
        _odd_span(max(5, 2 * window_size), nraw),
    ]
    evidence = []
    for span in dict.fromkeys(scales):
        order = 2 if span < 7 else 3
        smooth = savgol_filter(raw, span, order, mode="interp")
        d = np.diff(smooth)
        s = np.sign(d)
        s[s == 0] = 1
        crossing = np.flatnonzero(s[:-1] != s[1:]) + 1
        evidence.append(crossing)

    candidates = []
    for p in evidence[1]:
        support = sum(np.any(np.abs(q - p) <= radius) for q in (evidence[0], evidence[2]))
        left = max(0, p - radius)
        right = min(nraw - 1, p + radius)
        level = abs(raw[p] - 0.5 * (np.median(raw[left:p + 1]) +
                                    np.median(raw[p:right + 1])))
        if support >= 1 and level >= prominence:
            candidates.append(p)

    knots = [0]
    for p in candidates:
        if p > knots[-1] + 1:
            knots.append(int(p))
    if knots[-1] != nraw - 1:
        knots.append(nraw - 1)

    if len(knots) < 3:
        return raw

    # Use local medians at knots to suppress residual noise while retaining
    # the observed level and preserve the resulting trend shape.
    values = np.asarray([
        np.median(raw[max(0, p - radius):min(nraw, p + radius + 1)])
        for p in knots
    ], dtype=float)
    values[0] = raw[0]
    values[-1] = raw[-1]
    try:
        reconstructed = PchipInterpolator(
            np.asarray(knots, dtype=float), values
        )(np.arange(nraw, dtype=float))
    except Exception:
        return raw
    return np.asarray(reconstructed, dtype=float)


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
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
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

    correlation = (
        np.corrcoef(filtered_signal, aligned_clean)[0, 1]
        if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0
        else 0
    )
    noise_before = np.var(aligned_noisy - aligned_clean)
    noise_after = np.var(filtered_signal - aligned_clean)
    noise_reduction = (
        (noise_before - noise_after) / noise_before if noise_before > 0 else 0
    )
    return {
        "filtered_signal": filtered_signal,
        "clean_signal": aligned_clean,
        "noisy_signal": aligned_noisy,
        "correlation": correlation,
        "noise_reduction": noise_reduction,
        "signal_length": m,
    }


if __name__ == "__main__":
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")