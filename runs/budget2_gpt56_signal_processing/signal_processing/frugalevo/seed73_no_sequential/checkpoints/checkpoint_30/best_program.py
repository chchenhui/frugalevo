"""
Offline segment-aware signal reconstruction with persistent-extrema pruning and
endpoint-aligned output.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    cs = np.concatenate(([0.0], np.cumsum(x)))
    return (cs[window_size:] - cs[:-window_size]) / float(window_size)


def _finite_signal(x):
    if np.all(np.isfinite(x)):
        return x
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def _median3(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


def _smooth_piece(piece):
    clean = _median3(piece)
    n = clean.size
    if savgol_filter is None or n < 5:
        return clean
    frame = min(21, n if n % 2 else n - 1)
    if frame < 5:
        return clean
    return np.asarray(
        savgol_filter(clean, window_length=frame, polyorder=min(3, frame - 2),
                      mode="interp"),
        dtype=float,
    )


def _turning_points(y):
    """Detect endpoint-inclusive turning points after suppressing tiny derivative reversals."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return np.arange(n, dtype=int)

    delta = np.diff(y)
    scale = 1.4826 * np.median(np.abs(delta - np.median(delta))) + 1e-12
    deadband = max(0.10 * scale, 1e-12)

    signs = np.sign(delta).astype(float, copy=True)
    signs[np.abs(delta) <= deadband] = 0.0
    nonzero = np.flatnonzero(signs)
    if nonzero.size == 0:
        return np.array([0, n - 1], dtype=int)

    signs[:nonzero[0]] = signs[nonzero[0]]
    for i in range(nonzero[0] + 1, signs.size):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    extrema = np.flatnonzero(signs[:-1] != signs[1:]) + 1
    return np.unique(np.concatenate(([0], extrema, [n - 1]))).astype(int)


def _persistent_shape(y, sigma, spread):
    """Prune weak extrema by robust prominence and reconstruct retained trends with PCHIP."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()

    knots = list(_turning_points(y))
    threshold = max(0.75 * float(sigma), 0.006 * float(spread), 1e-12)

    # Repeatedly remove the weakest local feature so neighboring prominence
    # is recomputed after every deletion.
    while len(knots) > 2:
        prominences = [
            min(abs(y[knots[j]] - y[knots[j - 1]]),
                abs(y[knots[j]] - y[knots[j + 1]]))
            for j in range(1, len(knots) - 1)
        ]
        weakest = int(np.argmin(prominences))
        if prominences[weakest] >= threshold:
            break
        del knots[weakest + 1]

    xp = np.asarray(knots, dtype=float)
    fp = y[np.asarray(knots, dtype=int)]
    grid = np.arange(n, dtype=float)
    if xp.size < 3 or PchipInterpolator is None:
        return np.interp(grid, xp, fp)

    try:
        result = np.asarray(
            PchipInterpolator(xp, fp, extrapolate=False)(grid), dtype=float
        )
        if np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    return np.interp(grid, xp, fp)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Apply robust segment smoothing, persistent-extrema pruning, and shape-preserving reconstruction."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    x = _finite_signal(x)
    n = x.size
    if n < 5:
        return x[window_size - 1:].copy()

    d = np.diff(x)
    dmed = np.median(d)
    sigma_d = 1.4826 * np.median(np.abs(d - dmed)) + 1e-12
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    threshold = max(5.0 * sigma_d, 0.30 * spread)
    candidates = np.flatnonzero(np.abs(d) > threshold) + 1

    boundaries = [0]
    if candidates.size:
        group = [int(candidates[0])]
        for candidate in candidates[1:]:
            candidate = int(candidate)
            if candidate - group[-1] <= 4:
                group.append(candidate)
            else:
                boundaries.append(max(group, key=lambda q: abs(d[q - 1])))
                group = [candidate]
        boundaries.append(max(group, key=lambda q: abs(d[q - 1])))
    boundaries.append(n)

    reconstructed = np.empty(n, dtype=float)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        piece = x[left:right]
        smooth = _smooth_piece(piece)
        residual = piece - smooth
        residual_center = np.median(residual)
        sigma = 1.4826 * np.median(np.abs(residual - residual_center)) + 1e-12
        reconstructed[left:right] = _persistent_shape(smooth, sigma, spread)

    return reconstructed[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Dispatch to the persistent-extrema reconstruction or endpoint-aligned mean filter."""
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


def run_signal_processing(
    noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20
):
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
    aligned_clean = clean_signal[delay:]
    aligned_noisy = noisy_signal[delay:]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    correlation = (
        float(np.corrcoef(filtered_signal, aligned_clean)[0, 1]) if m > 1 else 0.0
    )
    before = np.var(aligned_noisy - aligned_clean)
    after = np.var(filtered_signal - aligned_clean)
    reduction = float((before - after) / before) if before > 0 else 0.0

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