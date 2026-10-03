import numpy as np


def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def _reflect_pad(x, radius):
    if radius <= 0:
        return x
    return np.pad(x, (radius, radius), mode="reflect")


def _gaussian_smooth(x, sigma):
    radius = int(np.ceil(3.0 * sigma))
    z = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (z / sigma) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(_reflect_pad(x, radius), kernel, mode="valid")


def _quadratic_center(x, width):
    width = int(width)
    if width < 3:
        return x.copy()
    if width % 2 == 0:
        width -= 1
    half = width // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.column_stack((np.ones(width), t, t * t))
    # Center estimate is the first row of the polynomial projection.
    projection = np.linalg.pinv(A)[0]
    padded = np.pad(x, (half, half), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return windows @ projection


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Apply an innovation-gated level/trend Kalman filter with RTS smoothing."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 3:
        return x.copy()

    # Robust scale supplies both the measurement noise estimate and a
    # physically reasonable bound against impulse-induced slope reversals.
    if not np.all(np.isfinite(x)):
        return np.asarray(_gaussian_smooth(np.nan_to_num(x), 1.0)[window_size - 1:])
    d = np.diff(x)
    dmed = np.median(d) if len(d) else 0.0
    dmad = 1.4826 * np.median(np.abs(d - dmed))
    if not np.isfinite(dmad) or dmad <= 1e-10:
        return np.asarray(x[window_size - 1:], dtype=float)

    r = max(dmad * dmad, 1e-8)
    qa = 0.02 * r
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = qa * np.array([[0.25, 0.5], [0.5, 1.0]])
    n = len(x)

    states = np.empty((n, 2), dtype=float)
    covariances = np.empty((n, 2, 2), dtype=float)
    states[0] = (x[0], dmed)
    covariances[0] = np.diag([r, max(dmad * dmad, 1e-8)])

    gate = 3.0 * dmad
    slope_limit = 4.0 * dmad
    for k in range(1, n):
        predicted = F @ states[k - 1]
        predicted_cov = F @ covariances[k - 1] @ F.T + Q
        innovation = x[k] - H @ predicted
        measurement_var = r * (9.0 if abs(innovation) > gate else 1.0)
        gain = predicted_cov @ H / (H @ predicted_cov @ H + measurement_var)
        updated = predicted + gain * innovation
        updated[1] = predicted[1] + np.clip(
            updated[1] - predicted[1], -slope_limit, slope_limit
        )
        updated_cov = (np.eye(2) - np.outer(gain, H)) @ predicted_cov
        states[k] = updated
        covariances[k] = 0.5 * (updated_cov + updated_cov.T)

    # Rauch-Tung-Striebel smoothing removes the causal trend lag while
    # retaining the independently estimated slope covariance.
    smoothed = states.copy()
    for k in range(n - 2, -1, -1):
        predicted_cov = F @ covariances[k] @ F.T + Q
        smoother_gain = covariances[k] @ F.T @ np.linalg.pinv(predicted_cov)
        smoothed[k] += smoother_gain @ (smoothed[k + 1] - F @ states[k])

    return np.asarray(smoothed[window_size - 1:, 0], dtype=float)


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


def run_signal_processing(
    noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20
):
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None
    else:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if clean_signal is not None:
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]
        m = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:m]
        aligned_clean = aligned_clean[:m]
        aligned_noisy = aligned_noisy[:m]
        correlation = (
            np.corrcoef(filtered_signal, aligned_clean)[0, 1] if m > 1 else 0.0
        )
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

    return {
        "filtered_signal": filtered_signal,
        "clean_signal": None,
        "noisy_signal": None,
        "correlation": 0,
        "noise_reduction": 0,
        "signal_length": len(filtered_signal),
    }


if __name__ == "__main__":
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")