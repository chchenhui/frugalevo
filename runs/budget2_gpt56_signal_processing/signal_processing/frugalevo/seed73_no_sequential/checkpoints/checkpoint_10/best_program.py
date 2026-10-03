"""
Offline robust segment-aware signal reconstruction with endpoint-aligned output.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
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
    """Three point median with edge replication rather than zero padding."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


def _smooth_piece(piece):
    """Apply edge-safe median filtering followed by an edge-corrected local polynomial."""
    n = len(piece)
    clean = _median3(piece)
    if savgol_filter is None or n < 5:
        return clean
    frame = min(21, n if n % 2 else n - 1)
    if frame < 5:
        return clean
    order = min(3, frame - 2)
    return savgol_filter(clean, frame, order, mode="interp")


def _hysteretic_reconstruction(values, deadband):
    """Apply directional hysteresis with two-sample reversal confirmation."""
    y = np.asarray(values, dtype=float).copy()
    n = len(y)
    if n < 2 or deadband <= 0:
        return y

    accepted = float(y[0])
    direction = 0
    reversal_sign = 0
    reversal_count = 0

    for i in range(1, n):
        proposal = float(y[i])
        delta = proposal - accepted

        if direction == 0:
            if abs(delta) >= deadband:
                direction = 1 if delta > 0 else -1
                accepted = proposal
            else:
                accepted = 0.5 * accepted + 0.5 * proposal
            reversal_sign = 0
            reversal_count = 0
        elif direction > 0:
            if delta >= 0:
                accepted = proposal
                reversal_sign = 0
                reversal_count = 0
            elif -delta >= deadband:
                if reversal_sign == -1:
                    reversal_count += 1
                else:
                    reversal_sign = -1
                    reversal_count = 1
                if reversal_count >= 2:
                    direction = -1
                    accepted = proposal
                    reversal_sign = 0
                    reversal_count = 0
            else:
                reversal_sign = 0
                reversal_count = 0
        else:
            if delta <= 0:
                accepted = proposal
                reversal_sign = 0
                reversal_count = 0
            elif delta >= deadband:
                if reversal_sign == 1:
                    reversal_count += 1
                else:
                    reversal_sign = 1
                    reversal_count = 1
                if reversal_count >= 2:
                    direction = 1
                    accepted = proposal
                    reversal_sign = 0
                    reversal_count = 0
            else:
                reversal_sign = 0
                reversal_count = 0

        y[i] = accepted

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Segment-aware robust smoothing with edge-safe median and hysteresis."""
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
    n = len(x)
    if n < 5:
        return x[window_size - 1:].copy()

    d = np.diff(x)
    dmed = np.median(d)
    sigma_d = 1.4826 * np.median(np.abs(d - dmed)) + 1e-12
    spread = np.percentile(x, 95) - np.percentile(x, 5)
    threshold = max(5.0 * sigma_d, 0.30 * spread)
    candidates = np.flatnonzero(np.abs(d) > threshold) + 1

    boundaries = [0]
    if len(candidates):
        group = [int(candidates[0])]
        for b in candidates[1:]:
            b = int(b)
            if b - group[-1] <= 4:
                group.append(b)
            else:
                boundaries.append(max(group, key=lambda q: abs(d[q - 1])))
                group = [b]
        boundaries.append(max(group, key=lambda q: abs(d[q - 1])))
    boundaries.append(n)

    reconstructed = np.empty(n, dtype=float)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        reconstructed[left:right] = _smooth_piece(x[left:right])

    """Estimate residual scale after reconstruction and suppress weak reversals."""
    residual = x - reconstructed
    residual_center = np.median(residual)
    sigma = 1.4826 * np.median(
        np.abs(residual - residual_center)
    ) + 1e-12
    deadband = 0.35 * sigma

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        # Reset hysteresis independently at every detected discontinuity so
        # reversals cannot be carried across genuine regime changes.
        reconstructed[left:right] = _hysteretic_reconstruction(
            reconstructed[left:right], deadband
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