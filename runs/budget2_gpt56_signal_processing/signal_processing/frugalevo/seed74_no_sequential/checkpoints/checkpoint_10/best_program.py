"""
Batch adaptive signal processing with zero-phase spectral smoothing.

The public output is aligned to the newest sample in every nominal sliding
window: output[i] estimates input[i + window_size - 1].
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """Simple endpoint-aligned moving-average baseline."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size) / float(window_size), mode="valid")


def _robust_scale(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


def _spectral_cutoff(x):
    """Select a conservative bandwidth from excess spectral energy."""
    n = len(x)
    if n < 12:
        return 0.18
    z = x - np.median(x)
    power = np.abs(np.fft.rfft(z)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0)
    if power.size < 4:
        return 0.18

    # The upper band provides a robust broadband-noise floor estimate.
    hi_start = max(1, int(0.35 * power.size))
    floor = np.median(power[hi_start:])
    excess = np.maximum(power.copy() - floor, 0.0)
    excess[0] = 0.0
    total = excess.sum()
    if not np.isfinite(total) or total <= 1e-14:
        return 0.16

    f95 = freq[np.searchsorted(np.cumsum(excess), 0.95 * total)]
    # scipy's Wn is normalized to Nyquist.  The margin retains curved,
    # oscillatory trends while rejecting remaining high-frequency noise.
    return float(np.clip(2.5 * f95, 0.10, 0.30))


def _fallback_smoother(x, window_size):
    """Dependency-free fallback used only if scipy is unavailable."""
    n = len(x)
    k = min(max(5, 2 * (window_size // 2) + 1), n if n % 2 else n - 1)
    if k < 3:
        return x.copy()
    kernel = np.ones(k, dtype=float) / k
    padded = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:n]


def _deadband(y, residual_scale, reference=None):
    """Prune weak multi-sample reversals using bounded prominence passes."""
    y = np.asarray(y, dtype=float).copy()
    if len(y) < 4:
        return y

    if reference is None:
        reference = y
    reference = np.asarray(reference, dtype=float)
    delta = np.diff(y)
    ref_delta = np.diff(reference)

    mad_delta = _robust_scale(delta)
    mad_residual = _robust_scale(reference - y)
    threshold = max(
        0.35 * mad_delta,
        0.12 * mad_residual,
        1e-12,
    )
    if not np.isfinite(threshold):
        return y

    # A small fixed number of passes bounds latency and prevents over-smoothing.
    for _ in range(3):
        d = np.diff(y)
        signs = np.sign(d).astype(float)

        # Forward-fill zero slopes so flat filtered sections do not hide turns.
        for i in range(1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        for i in range(len(signs) - 2, -1, -1):
            if signs[i] == 0:
                signs[i] = signs[i + 1]

        extrema = np.flatnonzero(signs[1:] != signs[:-1]) + 1
        if extrema.size < 2:
            break

        changed = False
        for j in range(1, len(extrema) - 1):
            left = int(extrema[j - 1])
            cur = int(extrema[j])
            right = int(extrema[j + 1])
            left_prom = abs(y[cur] - y[left])
            right_prom = abs(y[cur] - y[right])

            # Preserve a persistent extremum if both surrounding excursions
            # are clearly above the noise scale.
            if left_prom >= 2.0 * threshold and right_prom >= 2.0 * threshold:
                continue
            if min(left_prom, right_prom) >= threshold:
                continue

            if right > left:
                y[left:right + 1] = np.linspace(
                    y[left], y[right], right - left + 1
                )
                changed = True

        if not changed:
            break

    # Remove only residual sub-noise movement after structural pruning.
    dead_threshold = 0.05 * threshold
    if dead_threshold > 0:
        for i in range(1, len(y)):
            if abs(y[i] - y[i - 1]) < dead_threshold:
                y[i] = y[i - 1]
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Denoise with cycle-spun multiscale wavelet shrinkage and persistence cleanup."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)

    n = len(x)
    if n < 7:
        return _fallback_smoother(x, window_size)[window_size - 1:]

    sigma = _robust_scale(np.diff(x)) / np.sqrt(2.0)
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = max(_robust_scale(x), 1e-12)

    smooth = None
    try:
        import pywt
        wavelet = pywt.Wavelet("db4")
        level = min(5, pywt.dwt_max_level(n, wavelet.dec_len))
        threshold = sigma * np.sqrt(2.0 * np.log(max(n, 2)))
        reconstructions = []

        if level > 0:
            for shift in (0, 2, 4, 6):
                shifted = np.roll(x, shift)
                coeffs = pywt.wavedec(
                    shifted, wavelet, mode="symmetric", level=level
                )
                filtered = [np.asarray(coeffs[0], dtype=float).copy()]
                detail_count = len(coeffs) - 1

                for band, detail in enumerate(coeffs[1:]):
                    detail = np.asarray(detail, dtype=float).copy()
                    # coeffs[1] is the coarsest detail; the final bands
                    # are the finest details and receive the strongest shrinkage.
                    from_finest = detail_count - 1 - band
                    if from_finest == 0:
                        factor = 1.0
                    elif from_finest == 1:
                        factor = 1.0
                    elif from_finest == 2:
                        factor = 0.60
                    elif from_finest == 3:
                        factor = 0.30
                    else:
                        factor = 0.0
                    filtered.append(
                        pywt.threshold(
                            detail, factor * threshold, mode="soft"
                        )
                    )

                restored = pywt.waverec(filtered, wavelet, mode="symmetric")
                restored = np.asarray(restored[:n], dtype=float)
                reconstructions.append(np.roll(restored, -shift))
            smooth = np.mean(np.stack(reconstructions), axis=0)
    except Exception:
        smooth = None

    if smooth is None or smooth.shape != x.shape or not np.all(np.isfinite(smooth)):
        smooth = _fallback_smoother(x, window_size)

    # Protect isolated, high-confidence steps from wavelet ringing.
    spread = float(np.percentile(x, 75) - np.percentile(x, 25))
    jump_threshold = max(5.0 * sigma, 0.18 * spread, 1e-12)
    jumps = np.flatnonzero(np.abs(np.diff(x)) >= jump_threshold) + 1
    for jump in jumps:
        left = x[max(0, jump - 4):jump]
        right = x[jump:min(n, jump + 4)]
        if left.size and right.size:
            if abs(np.median(right) - np.median(left)) >= jump_threshold:
                lo = max(0, jump - 2)
                hi = min(n, jump + 2)
                smooth[lo:hi] = np.r_[
                    np.full(min(2, hi - lo), np.median(left)),
                    np.full(max(0, hi - lo - 2), np.median(right)),
                ][:hi - lo]

    # One bounded directional-persistence pass suppresses isolated reversals
    # without repeatedly altering genuine curvature.
    persistence = max(0.10 * sigma, 1e-12)
    delta = np.diff(smooth)
    signs = np.sign(delta)
    for i in range(1, len(signs) - 1):
        if signs[i] != 0 and signs[i - 1] != 0 and signs[i + 1] != 0:
            if signs[i - 1] != signs[i + 1]:
                if abs(delta[i]) <= persistence:
                    smooth[i + 1] = 0.5 * (smooth[i] + smooth[i + 2])
    return np.asarray(smooth, dtype=float)[window_size - 1:]


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
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
        noisy_signal = np.asarray(noisy_signal, dtype=float)
        clean_signal = None

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