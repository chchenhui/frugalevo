"""
Batch-capable adaptive signal processing for volatile non-stationary series.

The public output convention is retained: each output corresponds to the input
sample at the right edge of a window, so output length is len(x)-window_size+1.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def adaptive_filter(x, window_size=20):
    """Basic trailing-window moving-average baseline."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1 or len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def _finite_signal(x):
    """Make filtering well-defined for occasional missing or infinite samples."""
    z = np.asarray(x, dtype=float).copy()
    good = np.isfinite(z)
    if not np.any(good):
        return np.zeros_like(z)
    if not np.all(good):
        indices = np.arange(z.size)
        z[~good] = np.interp(indices[~good], indices[good], z[good])
    return z


def _median3(z):
    """Centered, edge-replicated median that does not introduce zero padding."""
    if z.size < 3:
        return z.copy()
    p = np.pad(z, (1, 1), mode="edge")
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


def _triangular_fallback(z, window_size):
    """Centered zero-phase fallback used if SciPy is not installed."""
    radius = max(2, min(12, int(round(window_size * 0.45))))
    ramp = np.arange(1, radius + 2, dtype=float)
    kernel = np.r_[ramp, ramp[-2::-1]]
    kernel /= kernel.sum()
    pad = kernel.size // 2
    return np.convolve(np.pad(z, (pad, pad), mode="reflect"), kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Apply centered median rejection followed by zero-phase Butterworth smoothing."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1 or len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    z = _finite_signal(x)
    if z.size <= 3:
        return z[window_size - 1:].copy()

    # The median removes isolated impulses before the low-pass stage can spread
    # them into neighboring samples and create artificial reversals.
    robust = _median3(z)
    cutoff = float(np.clip(1.4 / max(window_size, 1), 0.045, 0.12))

    if _HAVE_SCIPY and robust.size >= 10:
        try:
            sos = butter(3, cutoff, btype="lowpass", output="sos")
            # Use the largest safe filtfilt extension for better edge behavior;
            # unlike a fixed short pad, this reduces endpoint transients while
            # remaining valid for short input batches.
            padlen = min(3 * (2 * sos.shape[0] + 1), robust.size - 1)
            estimate = sosfiltfilt(sos, robust, padlen=padlen)
        except Exception:
            estimate = _triangular_fallback(robust, window_size)
    else:
        estimate = _triangular_fallback(robust, window_size)

    estimate = np.asarray(estimate, dtype=float)
    if not np.all(np.isfinite(estimate)):
        estimate = _triangular_fallback(z, window_size)
    return estimate[window_size - 1:].copy()


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply the selected filtering algorithm."""
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """Generate the original demonstration signal."""
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
    """
    Process either a supplied series or the demonstration series.

    `filtered_signal` always has exactly len(input)-window_size+1 samples.
    """
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
        noisy_signal = np.asarray(noisy_signal, dtype=float)
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
    aligned_clean = clean_signal[delay:delay + len(filtered_signal)]
    aligned_noisy = noisy_signal[delay:delay + len(filtered_signal)]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0:
        correlation = float(np.corrcoef(filtered_signal, aligned_clean)[0, 1])
    else:
        correlation = 0.0
    noise_before = float(np.var(aligned_noisy - aligned_clean))
    noise_after = float(np.var(filtered_signal - aligned_clean))
    noise_reduction = (
        (noise_before - noise_after) / noise_before if noise_before > 0 else 0.0
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