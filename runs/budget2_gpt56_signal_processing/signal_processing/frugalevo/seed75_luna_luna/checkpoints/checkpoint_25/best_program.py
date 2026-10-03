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
    sigma = max(float(sigma), 1e-6)
    radius = int(np.ceil(3.0 * sigma))
    z = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (z / sigma) ** 2)
    k /= k.sum()
    return np.convolve(_reflect_pad(x, radius), k, mode="valid")


def _tv_denoise(y, weight, iterations=120):
    """Bounded 1-D Chambolle primal-dual TV denoiser."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 or weight <= 0:
        return y.copy()

    p = np.zeros(n - 1, dtype=float)
    tau = 0.24
    for _ in range(iterations):
        div = np.empty(n, dtype=float)
        div[0] = -p[0]
        div[1:-1] = p[:-1] - p[1:]
        div[-1] = p[-1]
        u = y - weight * div
        g = np.diff(u)
        p = (p + (tau / weight) * g)
        p /= np.maximum(1.0, np.abs(p))
    div = np.empty(n, dtype=float)
    div[0] = -p[0]
    div[1:-1] = p[:-1] - p[1:]
    div[-1] = p[-1]
    return y - weight * div


def _reversal_count(y):
    d = np.diff(y)
    if len(d) < 2:
        return 0
    return int(np.count_nonzero(d[:-1] * d[1:] < 0))


def enhanced_filter_with_trend_preservation(x, window_size=20):
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

    d = np.diff(x)
    center = np.median(d) if len(d) else 0.0
    scale = 1.4826 * np.median(np.abs(d - center)) if len(d) else 0.0
    scale = max(float(scale), 1e-7)

    # Repair only extreme isolated innovations.
    repaired = x.copy()
    if len(x) >= 3:
        pred = 0.5 * (x[:-2] + x[2:])
        bad = np.abs(x[1:-1] - pred) > 4.0 * scale
        repaired[1:-1][bad] = pred[bad]

    candidates = []
    for multiplier in (0.45, 0.75, 1.15):
        weight = multiplier * np.sqrt(max(window_size, 3)) * scale
        z = _tv_denoise(repaired, weight)
        # A small zero-phase residual smoother removes staircase micro-jitter.
        z = 0.75 * z + 0.25 * _gaussian_smooth(z, 0.7)
        candidates.append(z)

    base_dd = np.diff(repaired, n=2)
    dd0 = np.mean(base_dd * base_dd) + 1e-12
    var0 = np.var(repaired) + 1e-12
    rev0 = _reversal_count(repaired) + 1e-12

    scores = []
    for z in candidates:
        rev = _reversal_count(z) / rev0
        dd = np.mean(np.diff(z, n=2) ** 2) / dd0
        residual = np.mean((repaired - z) ** 2) / var0
        scores.append(0.55 * rev + 0.25 * dd + 0.20 * residual)

    selected = candidates[int(np.argmin(scores))]

    # Optional mature zero-phase refinement where available.
    try:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, 0.075, btype="low", output="sos")
        refined = sosfiltfilt(sos, selected, padtype="odd")
        if (
            _reversal_count(refined) <= _reversal_count(selected)
            and np.mean((repaired - refined) ** 2) <= 1.25 * np.mean((repaired - selected) ** 2)
        ):
            selected = refined
    except Exception:
        pass

    return np.asarray(selected[window_size - 1:], dtype=float)


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
    return noisy_signal if False else (
        clean_signal + np.random.normal(0, noise_level, length),
        clean_signal,
    )


def run_signal_processing(
    noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20
):
    if noisy_signal is not None:
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }

    noisy_signal, clean_signal = generate_test_signal(
        signal_length, noise_level
    )
    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
    delay = window_size - 1
    aligned_clean = clean_signal[delay:delay + len(filtered_signal)]
    aligned_noisy = noisy_signal[delay:delay + len(filtered_signal)]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]
    correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if m > 1 else 0.0
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