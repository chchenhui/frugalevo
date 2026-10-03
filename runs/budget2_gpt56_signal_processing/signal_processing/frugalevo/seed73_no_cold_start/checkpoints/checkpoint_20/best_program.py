"""
Batch robust adaptive denoising for volatile non-stationary time series.

Returned samples are aligned to input indices window_size - 1 through end.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
except Exception:
    butter = None
    sosfiltfilt = None
    savgol_filter = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


def _finite_signal(x):
    z = np.asarray(x, dtype=float).reshape(-1).copy()
    if z.size == 0:
        return z
    good = np.isfinite(z)
    if not np.all(good):
        if np.any(good):
            ind = np.arange(z.size)
            z[~good] = np.interp(ind[~good], ind[good], z[good])
        else:
            z.fill(0.0)
    return z


def _odd_at_most(value, n):
    w = min(int(value), int(n))
    return w if w % 2 else w - 1


def _savgol_safe(x, preferred=11):
    x = _finite_signal(x)
    n = len(x)
    if savgol_filter is None or n < 5:
        return x.copy()
    w = _odd_at_most(preferred, n)
    if w < 5:
        return x.copy()
    try:
        return savgol_filter(x, w, min(3, w - 2), mode="interp")
    except Exception:
        return x.copy()


def _median3(x):
    x = _finite_signal(x)
    if len(x) < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.vstack((p[:-2], p[1:-1], p[2:])), axis=0)


def _mad_sigma(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.67448975)


def _spectral_cutoff(x):
    n = len(x)
    if n < 8:
        return 0.18
    power = np.abs(np.fft.rfft(x)) ** 2
    bins = len(power)
    tail = power[max(1, int(0.70 * bins)):]
    floor = float(np.median(tail)) if len(tail) else 0.0
    useful = np.maximum(power - floor, 0.0)
    useful[0] = 0.0
    total = float(np.sum(useful))
    if not np.isfinite(total) or total <= 1e-14:
        return 0.18
    k = int(np.searchsorted(np.cumsum(useful), 0.94 * total))
    k = min(max(k, 1), bins - 1)
    return float(np.clip(2.0 * k / n, 0.07, 0.28))


def _hysteretic_extrema(y, excursion):
    """Alternating extrema confirmed only by a scale-supported reversal."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.arange(n, dtype=int)

    excursion = max(float(excursion), 1e-12)
    anchors = [0]

    direction = 1 if y[1] >= y[0] else -1
    candidate = 1

    for i in range(2, n):
        if direction > 0:
            if y[i] >= y[candidate]:
                candidate = i
            elif y[candidate] - y[i] >= excursion:
                if candidate > anchors[-1]:
                    anchors.append(candidate)
                direction = -1
                candidate = i
        else:
            if y[i] <= y[candidate]:
                candidate = i
            elif y[i] - y[candidate] >= excursion:
                if candidate > anchors[-1]:
                    anchors.append(candidate)
                direction = 1
                candidate = i

    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    return np.asarray(anchors, dtype=int)


def _adaptive_full_record_filter(x):
    """Denoise using robust median preconditioning and one low-rank Hankel SVD."""
    x = _finite_signal(x)
    n = len(x)
    if n < 2:
        return x.copy()

    robust = _median3(x)
    if n < 8:
        preliminary = _savgol_safe(robust, 11)
    else:
        width = min(40, max(8, n // 3), n)
        columns = n - width + 1
        try:
            # H[row, column] = robust[row + column]; this orientation makes
            # anti-diagonal averaging recover the original sample indices.
            trajectory = np.lib.stride_tricks.sliding_window_view(
                robust, width
            ).T.astype(float, copy=True)

            u, singular, vt = np.linalg.svd(
                trajectory, full_matrices=False
            )
            energy = singular * singular
            tail = energy[max(1, (2 * len(energy)) // 3):]
            floor = float(np.median(tail)) if len(tail) else 0.0
            useful = np.maximum(energy - floor, 0.0)
            total = float(np.sum(useful))
            if not np.isfinite(total) or total <= 1e-12:
                raise ValueError("degenerate trajectory energy")

            # Retain the smallest coherent trajectory subspace that explains
            # nearly all de-biased energy.  The slightly conservative target
            # suppresses weak noise-dominated components without discarding
            # the dominant trend and oscillatory modes.
            rank = int(np.searchsorted(
                np.cumsum(useful), 0.965 * total
            ) + 1)
            rank = min(max(rank, 2), 8, len(singular))
            low_rank = (
                u[:, :rank] * singular[:rank][None, :]
            ) @ vt[:rank, :]

            preliminary = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for row in range(width):
                end = min(columns, n - row)
                if end:
                    preliminary[row:row + end] += low_rank[row, :end]
                    counts[row:row + end] += 1.0
            preliminary /= np.maximum(counts, 1.0)
            preliminary = np.where(
                np.isfinite(preliminary), preliminary, robust
            )
        except Exception:
            preliminary = _savgol_safe(robust, 11)

    if n < 5:
        return preliminary

    residual_sigma = _mad_sigma(x - preliminary)
    signal_scale = _mad_sigma(preliminary)
    # Use a noise-adaptive hysteresis threshold.  The modest increase over one
    # MAD makes isolated residual spikes less likely to create reversals while
    # retaining genuine movements larger than the local noise scale.
    excursion = max(1.15 * residual_sigma, 0.015 * signal_scale, 1e-12)

    anchors = _hysteretic_extrema(preliminary, excursion)
    if len(anchors) < 4:
        return preliminary

    ay = preliminary[anchors]
    grid = np.arange(n)
    try:
        if PchipInterpolator is None:
            raise ValueError("no pchip")
        skeleton = PchipInterpolator(anchors, ay, extrapolate=True)(grid)
    except Exception:
        skeleton = np.interp(grid, anchors, ay)

    skeleton = np.asarray(skeleton, dtype=float)
    skeleton = np.where(np.isfinite(skeleton), skeleton, preliminary)
    # Trust the extrema skeleton more when the preliminary estimate contains
    # substantial residual noise; otherwise retain more of the responsive
    # low-rank estimate to limit lag on relatively clean segments.
    noise_ratio = residual_sigma / max(signal_scale, 1e-12)
    skeleton_weight = float(np.clip(0.84 + 0.10 * noise_ratio, 0.84, 0.94))
    y = skeleton_weight * skeleton + (1.0 - skeleton_weight) * preliminary
    return np.where(np.isfinite(y), y, preliminary)


def adaptive_filter(x, window_size=20):
    """Apply Hankel low-rank denoising followed by hysteretic trend projection."""
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return _adaptive_full_record_filter(x)[window_size - 1:]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, length)
    clean_signal = (
        2.0 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2.0 * t)
        + 0.5 * np.sin(2 * np.pi * 5.0 * t)
        + 0.8 * np.exp(-t / 5.0) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean_signal += 0.1 * t * np.sin(0.2 * t)
    clean_signal += np.cumsum(rng.randn(length) * 0.05)
    noisy_signal = clean_signal + rng.normal(0, noise_level, length)
    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level
        )
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

    aligned_clean = clean_signal[window_size - 1:]
    aligned_noisy = noisy_signal[window_size - 1:]
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