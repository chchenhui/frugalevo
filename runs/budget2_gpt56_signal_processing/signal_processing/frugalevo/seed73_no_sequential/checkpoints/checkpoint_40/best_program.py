"""
Offline segment-aware signal reconstruction with persistent-extrema pruning,
followed by a positive endpoint-aligned affine amplitude calibration.
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
        savgol_filter(
            clean,
            window_length=frame,
            polyorder=min(3, frame - 2),
            mode="interp",
        ),
        dtype=float,
    )


def _turning_points(y):
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
    """Build a noise-tolerant Douglas-Peucker polyline and interpolate it with PCHIP."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()

    epsilon = max(0.95 * float(sigma), 0.008 * float(spread), 1e-12)
    max_interior = min(96, n // 2)
    selected = [0, n - 1]

    # Each entry is (largest deviation, left endpoint, right endpoint,
    # deviation index).  Splitting the largest residual first gives the
    # strongest geometric representation when the knot budget is reached.
    pending = []

    def best_split(left, right):
        """Return the maximum vertical residual from the endpoint chord."""
        if right - left < 2:
            return 0.0, -1
        indices = np.arange(left + 1, right, dtype=int)
        fraction = (indices - left) / float(right - left)
        chord = y[left] + fraction * (y[right] - y[left])
        residual = np.abs(y[indices] - chord)
        pos = int(np.argmax(residual))
        return float(residual[pos]), int(indices[pos])

    deviation, index = best_split(0, n - 1)
    if index >= 0 and deviation > epsilon:
        pending.append((-deviation, 0, n - 1, index))

    while pending and len(selected) - 2 < max_interior:
        pending.sort(key=lambda item: item[0])
        _, left, right, index = pending.pop(0)
        if index in selected:
            continue
        selected.append(index)

        deviation, split = best_split(left, index)
        if split >= 0 and deviation > epsilon:
            pending.append((-deviation, left, index, split))

        deviation, split = best_split(index, right)
        if split >= 0 and deviation > epsilon:
            pending.append((-deviation, index, right, split))

    knots = np.asarray(sorted(set(selected)), dtype=int)
    xp = knots.astype(float)
    fp = y[knots]
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


def _positive_affine_calibration(reference, estimate):
    """
    Positive global affine projection onto the observed endpoint-aligned signal.
    Positive gain leaves all derivative sign changes invariant.
    """
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)

    emean = float(np.mean(estimate))
    rmean = float(np.mean(reference))
    centered_est = estimate - emean
    variance = float(np.mean(centered_est * centered_est))

    if not np.isfinite(variance) or variance <= 1e-14:
        return estimate + (rmean - emean)

    covariance = float(np.mean(centered_est * (reference - rmean)))
    gain = covariance / variance
    if not np.isfinite(gain):
        gain = 1.0

    # Moderate positive limits prevent a noisy global regression from
    # amplifying individual reconstruction errors or reversing trend signs.
    gain = float(np.clip(gain, 0.85, 1.20))
    bias = rmean - gain * emean
    result = gain * estimate + bias
    return np.nan_to_num(result, nan=rmean, posinf=rmean, neginf=rmean)


def enhanced_filter_with_trend_preservation(x, window_size=20):
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

    # Project the reconstructed shape back toward the observed signal with a
    # bounded positive affine fit.  Positive gain preserves all turning-point
    # topology while correcting smoothing-induced attenuation and bias.
    centered = reconstructed - float(np.mean(reconstructed))
    variance = float(np.mean(centered * centered))
    if np.isfinite(variance) and variance > 1e-14:
        covariance = float(np.mean(centered * (x - float(np.mean(x)))))
        gain = covariance / (variance + 1e-12)
        gain = float(np.clip(gain if np.isfinite(gain) else 1.0, 0.85, 1.20))
        bias = float(np.mean(x)) - gain * float(np.mean(reconstructed))
        reconstructed = gain * reconstructed + bias
    else:
        reconstructed = reconstructed + (
            float(np.mean(x)) - float(np.mean(reconstructed))
        )

    reconstructed = np.nan_to_num(
        reconstructed, nan=float(np.mean(x)),
        posinf=float(np.mean(x)), neginf=float(np.mean(x))
    )
    return reconstructed[window_size - 1:]


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