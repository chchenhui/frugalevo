"""
Batch zero-phase adaptive signal filtering with robust local smoothing.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def adaptive_filter(x, window_size=20):
    """Baseline trailing-window mean, retained for the public interface."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size, dtype=float) / window_size,
                       mode="valid")


def _odd_at_most(value, n):
    value = int(min(value, n))
    if value % 2 == 0:
        value -= 1
    return value


def _fallback_smooth(x, width):
    """Dependency-free centered triangular smoother used only without SciPy."""
    n = len(x)
    width = _odd_at_most(width, n)
    if width < 3:
        return x.copy()
    half = width // 2
    weights = np.arange(1, half + 2, dtype=float)
    weights = np.r_[weights, weights[-2::-1]]
    weights /= weights.sum()
    padded = np.pad(x, (half, half), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Detect high-confidence difference change points, then zero-phase smooth each
    regime independently while preserving samples directly adjacent to boundaries.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    n = len(x)
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)

    dx = np.diff(x)
    if dx.size:
        median_dx = float(np.median(dx))
        sigma_d = float(np.median(np.abs(dx - median_dx)) / 0.67448975)
        signal_range = float(np.max(x) - np.min(x))
        threshold = max(6.0 * sigma_d, 0.12 * signal_range)
        boundaries = np.flatnonzero(np.abs(dx) > threshold) + 1
    else:
        boundaries = np.empty(0, dtype=int)

    y = x.copy()
    starts = np.r_[0, boundaries]
    stops = np.r_[boundaries, n]
    cutoff = float(np.clip(2.6 / max(float(window_size), 2.0), 0.10, 0.30))
    sos = butter(3, cutoff, btype="low", output="sos") if _HAS_SCIPY else None

    for start, stop in zip(starts, stops):
        segment = x[start:stop]
        length = len(segment)
        if length < 8:
            continue
        if sos is not None:
            padlen = min(3 * (2 * sos.shape[0] + 1), length - 1)
            try:
                y[start:stop] = sosfiltfilt(sos, segment, padlen=padlen)
            except ValueError:
                y[start:stop] = _fallback_smooth(segment, min(9, length))
        else:
            y[start:stop] = _fallback_smooth(segment, min(9, length))

    # Keep both sides of every accepted regime transition exactly unsmoothed.
    for boundary in boundaries:
        y[boundary - 1] = x[boundary - 1]
        y[boundary] = x[boundary]

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
    aligned_clean = np.asarray(clean_signal, dtype=float)[delay:]
    aligned_noisy = np.asarray(noisy_signal, dtype=float)[delay:]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    correlation = (
        float(np.corrcoef(filtered_signal, aligned_clean)[0, 1])
        if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0
        else 0.0
    )
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