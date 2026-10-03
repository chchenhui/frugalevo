"""
Batch robust adaptive denoising for volatile, non-stationary time series.

Returned samples are aligned to input indices window_size - 1 through end.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
except Exception:
    butter = None
    sosfiltfilt = None
    savgol_filter = None


def _finite_signal(x):
    z = np.asarray(x, dtype=float).reshape(-1).copy()
    if z.size == 0:
        return z
    good = np.isfinite(z)
    if not np.all(good):
        if np.any(good):
            idx = np.arange(z.size)
            z[~good] = np.interp(idx[~good], idx[good], z[good])
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
    """Symmetric order-statistic preconditioner with endpoint replication."""
    x = _finite_signal(x)
    if len(x) < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.vstack((p[:-2], p[1:-1], p[2:])), axis=0)


def _noise_sigma(x):
    if len(x) < 2:
        return 0.0
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) /
                 (0.67448975 * np.sqrt(2.0)))


def _spectral_cutoff(x):
    """Noise-floor-corrected retained-power cutoff in scipy normalized units."""
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


def _adaptive_full_record_filter(x):
    """Build a persistence-pruned extrema skeleton over a robust preliminary estimate."""
    x = _finite_signal(x)
    n = len(x)
    if n < 2:
        return x.copy()

    robust = _median3(x)
    if butter is None or sosfiltfilt is None or n < 8:
        preliminary = _savgol_safe(robust, 11)
    else:
        try:
            cutoff = _spectral_cutoff(robust)
            sos = butter(3, cutoff, btype="lowpass", output="sos")
            padlen = min(3 * (2 * len(sos) + 1), n - 1)
            if padlen < 1:
                raise ValueError("short signal")
            preliminary = np.asarray(
                sosfiltfilt(sos, robust, padlen=padlen), dtype=float
            )
            preliminary = 0.93 * preliminary + 0.07 * robust
        except Exception:
            preliminary = _savgol_safe(robust, 11)

    preliminary = np.where(np.isfinite(preliminary), preliminary, robust)
    if n < 5:
        return preliminary

    # Replace zero derivative signs by the preceding nonzero sign so plateaus
    # do not create artificial extrema.
    d = np.diff(preliminary)
    signs = np.sign(d).astype(np.int8)
    nz = np.flatnonzero(signs)
    if len(nz):
        first = int(nz[0])
        signs[:first] = signs[first]
        for i in range(first + 1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
    if not len(nz):
        return preliminary

    extrema = [0]
    for i in range(1, n - 1):
        if signs[i - 1] != 0 and signs[i] != 0 and signs[i - 1] != signs[i]:
            extrema.append(i)
    extrema.append(n - 1)

    sigma = _noise_sigma(preliminary)
    threshold = 1.35 * max(sigma, 1e-12)

    # Remove weak extrema in-place, rechecking both neighbors after each
    # deletion. This deletes unsupported reversal pairs rather than merely
    # shrinking their amplitude.
    changed = True
    while changed and len(extrema) > 3:
        changed = False
        for j in range(1, len(extrema) - 1):
            left, cur, right = extrema[j - 1:j + 2]
            prominence = min(
                abs(preliminary[cur] - preliminary[left]),
                abs(preliminary[cur] - preliminary[right]),
            )
            if prominence < threshold:
                del extrema[j]
                changed = True
                break

    if len(extrema) < 4:
        return preliminary

    anchor_x = np.asarray(extrema, dtype=int)
    anchor_y = preliminary[anchor_x]
    try:
        from scipy.interpolate import PchipInterpolator
        skeleton = PchipInterpolator(anchor_x, anchor_y, extrapolate=True)(
            np.arange(n)
        )
    except Exception:
        skeleton = np.interp(np.arange(n), anchor_x, anchor_y)

    skeleton = np.asarray(skeleton, dtype=float)
    y = 0.82 * skeleton + 0.18 * preliminary
    return np.where(np.isfinite(y), y, preliminary)


def adaptive_filter(x, window_size=20):
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return _adaptive_full_record_filter(x)[window_size - 1:]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return _adaptive_full_record_filter(x)[window_size - 1:]


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