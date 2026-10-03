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
    """Apply robust impulse repair and select a zero-phase Butterworth smoother."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 3:
        return x.copy()

    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)

    # Repair isolated impulses while preserving sustained turning points.
    repaired = x.copy()
    if len(x) >= 3:
        scale = 1.4826 * np.median(
            np.abs(np.diff(x) - np.median(np.diff(x)))
        )
        scale = max(float(scale), 1e-8)
        neighbors = 0.5 * (x[:-2] + x[2:])
        isolated = np.abs(x[1:-1] - neighbors) > 3.0 * scale
        repaired[1:-1][isolated] = neighbors[isolated]

    try:
        from scipy.signal import butter, sosfiltfilt

        candidates = []
        for order, cutoff in ((2, 0.08), (3, 0.12), (4, 0.18)):
            sos = butter(order, min(cutoff, 0.45), btype="low", output="sos")
            try:
                candidates.append(sosfiltfilt(sos, repaired, padtype="odd"))
            except ValueError:
                candidates.append(sosfiltfilt(sos, repaired, method="gust"))
    except Exception:
        # Preserve a valid low-latency fallback when SciPy is unavailable.
        return np.asarray(
            _gaussian_smooth(repaired, max(0.8, min(1.5, window_size / 18.0)))[
                window_size - 1:
            ],
            dtype=float,
        )

    base_d = np.diff(repaired)
    base_dd = np.diff(repaired, n=2)
    base_rev = np.count_nonzero(base_d[1:] * base_d[:-1] < 0)
    base_dd_energy = np.mean(base_dd * base_dd) + 1e-12
    base_residual = np.mean((repaired - np.mean(repaired)) ** 2) + 1e-12

    def reversal_count(y):
        d = np.diff(y)
        return np.count_nonzero(d[1:] * d[:-1] < 0)

    scores = []
    for y in candidates:
        rev = reversal_count(y) / (base_rev + 1e-12)
        dd = np.mean(np.diff(y, n=2) ** 2) / base_dd_energy
        residual = np.mean((repaired - y) ** 2) / base_residual
        scores.append(0.55 * rev + 0.25 * dd + 0.20 * residual)

    selected = np.asarray(candidates[int(np.argmin(scores))], dtype=float)
    return selected[window_size - 1:].copy()


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