"""
Offline, length-preserving robust smoother for volatile non-stationary signals.
"""
import numpy as np

try:
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def adaptive_filter(x, window_size=20):
    """Trailing moving-average baseline with the required output length."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[window_size:] - c[:-window_size]) / float(window_size)


def _finite_signal(x):
    x = np.asarray(x, dtype=float).copy()
    if np.all(np.isfinite(x)):
        return x
    good = np.flatnonzero(np.isfinite(x))
    if len(good) == 0:
        return np.zeros_like(x)
    bad = np.flatnonzero(~np.isfinite(x))
    x[bad] = np.interp(bad, good, x[good])
    return x


def _fallback_smooth(z, span):
    """Centered finite fallback used only when scipy is unavailable."""
    if len(z) < 3:
        return z.copy()
    span = min(span, len(z) if len(z) % 2 else len(z) - 1)
    if span < 3:
        return z.copy()
    half = span // 2
    padded = np.pad(z, (half, half), mode="edge")
    return np.convolve(padded, np.ones(span) / span, mode="valid")


def _smooth_segment(z, target_span):
    """
    Reconstruct one change-point-bounded segment with median rejection, cubic
    SG fitting, and a zero-phase nine-tap binomial residual refinement.
    """
    z = np.asarray(z, dtype=float)
    m = len(z)
    if m < 5:
        return z.copy()

    span = min(target_span, m if m % 2 else m - 1)
    if span < 5:
        return _fallback_smooth(z, span)

    if _HAVE_SCIPY:
        # This is intentionally segment-local: change-point boundaries remain
        # hard, so neither the median stage nor the polynomial fit leaks a
        # plateau level through a genuine step.
        med = median_filter(z, size=3, mode="nearest")
        base = savgol_filter(
            med, window_length=span, polyorder=min(3, span - 2), mode="interp"
        )
    else:
        base = _fallback_smooth(z, span)

    # The failed direct-SG variant established that residual suppression is
    # essential.  A fourth centered binomial pass modestly rejects remaining
    # alternating derivative noise without phase delay or cross-cut blending.
    padded = np.pad(base, (4, 4), mode="edge")
    return (
        padded[:-8]
        + 8.0 * padded[1:-7]
        + 28.0 * padded[2:-6]
        + 56.0 * padded[3:-5]
        + 70.0 * padded[4:-4]
        + 56.0 * padded[5:-3]
        + 28.0 * padded[6:-2]
        + 8.0 * padded[7:-1]
        + padded[8:]
    ) * (1.0 / 256.0)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Batch, zero-phase robust local-polynomial reconstruction.

    Strong discontinuities are used as segment boundaries so smoothing does not
    leak a plateau level through a genuine step.
    """
    x = _finite_signal(x)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n == 0:
        return np.empty(0, dtype=float)

    # A 15-sample cubic fit retains local curvature while providing enough
    # derivative stability for the subsequent zero-phase binomial refinement.
    # It remains bounded by the requested sliding-window scale.
    target_span = min(15, 2 * (int(window_size) // 2) + 1)
    target_span = max(5, target_span)
    y = np.empty(n, dtype=float)

    # Robustly identify only exceptionally large jumps.  This deliberately
    # avoids treating normal sinusoidal slopes as changes of topology.
    if n >= 3:
        d = np.diff(x)
        center = np.median(d)
        scale = np.median(np.abs(d - center)) / 0.6744897501960817
        scale = max(float(scale), 1e-10)
        cuts = np.flatnonzero(np.abs(d - center) > 8.0 * scale) + 1
        # Adjacent impulse-derived cuts are not useful as separate plateaus.
        if len(cuts) > 1:
            cuts = cuts[np.r_[True, np.diff(cuts) > 2]]
    else:
        cuts = np.empty(0, dtype=int)

    bounds = np.concatenate(([0], cuts, [n]))
    for left, right in zip(bounds[:-1], bounds[1:]):
        y[left:right] = _smooth_segment(x[left:right], target_span)

    # A defensive finite guarantee also covers unusual numerical-library input.
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 10, length)
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean_signal += 0.1 * t * np.sin(0.2 * t)
    clean_signal += np.cumsum(np.random.randn(length) * 0.05)
    noisy_signal = clean_signal + np.random.normal(0, noise_level, length)
    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
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
    aligned_noisy = np.asarray(noisy_signal, dtype=float)[delay:]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0:
        correlation = float(np.corrcoef(filtered_signal, aligned_clean)[0, 1])
    else:
        correlation = 0.0
    noise_before = np.var(aligned_noisy - aligned_clean)
    noise_after = np.var(filtered_signal - aligned_clean)
    noise_reduction = (
        float((noise_before - noise_after) / noise_before)
        if noise_before > 0 else 0.0
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