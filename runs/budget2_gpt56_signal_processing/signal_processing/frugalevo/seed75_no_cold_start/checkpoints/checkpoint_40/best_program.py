"""
Offline, zero-phase, regime-aware denoising with persistence-aware reversal
debouncing.  The complete record is intentionally available before the aligned
tail is returned.
"""
import numpy as np

try:
    from scipy.interpolate import UnivariateSpline
    _HAS_SPLINE = True
except Exception:
    UnivariateSpline = None
    _HAS_SPLINE = False


def _finite_signal(x):
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    if x.size == 0 or np.all(np.isfinite(x)):
        return x
    good = np.flatnonzero(np.isfinite(x))
    if good.size == 0:
        return np.zeros_like(x)
    bad = np.flatnonzero(~np.isfinite(x))
    x[bad] = np.interp(bad, good, x[good])
    return x


def _fallback_smooth(x, radius):
    n = len(x)
    if n < 3:
        return x.copy()
    radius = max(1, min(int(radius), (n - 1) // 2))
    up = np.arange(1, radius + 2, dtype=float)
    kernel = np.r_[up, up[-2::-1]]
    kernel /= kernel.sum()
    p = len(kernel) // 2
    return np.convolve(np.pad(x, p, mode="reflect"), kernel, mode="valid")


def _turns(z):
    """Return strict derivative-sign reversal sample indices."""
    d = np.diff(z)
    if d.size < 2:
        return np.empty(0, dtype=int)
    s = np.sign(d)
    # Carry neighboring signs through exact flat portions.
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    return np.flatnonzero(s[:-1] * s[1:] < 0) + 1


def _prune_prominent_extrema(segment, trend):
    """
    Remove weak turns, with a persistence test for narrow noise zigzags.
    Broad extrema are protected even when their local amplitude is moderate.
    """
    y = np.asarray(segment, dtype=float)
    z = np.asarray(trend, dtype=float).copy()
    m = len(z)
    if m < 7:
        return z

    residual = y - z
    residual -= np.median(residual)
    sigma = 1.4826 * float(np.median(np.abs(residual)))
    span = float(np.percentile(y, 95) - np.percentile(y, 5))
    base = max(1.15 * sigma, 0.06 * span, 1e-12)

    for _ in range(3):
        turns = _turns(z)
        if turns.size == 0:
            break
        extrema = np.r_[0, turns, m - 1]
        interior = extrema[1:-1]
        prom = np.minimum(
            np.abs(z[interior] - z[extrema[:-2]]),
            np.abs(z[extrema[2:]] - z[interior])
        )
        widths = np.minimum(interior - extrema[:-2], extrema[2:] - interior)

        weak = (prom < base) | ((widths <= 2) & (prom < 2.0 * base))
        if not np.any(weak):
            break

        # Small score means low amplitude and/or an implausibly brief turn.
        score = prom / base + 0.35 * widths.astype(float)
        candidates = np.flatnonzero(weak)
        j = int(candidates[np.argmin(score[candidates])]) + 1
        left, right = int(extrema[j - 1]), int(extrema[j + 1])
        if right <= left:
            break
        z[left:right + 1] = np.linspace(z[left], z[right], right - left + 1)

    z[0] = y[0]
    z[-1] = y[-1]
    return z


def _fit_segment(segment, sigma):
    """Choose a GCV spline, then perform two bounded Huber-weighted refits."""
    m = len(segment)
    if m < 8:
        return _fallback_smooth(segment, max(1, (m - 1) // 3)) if m >= 3 else segment

    if not _HAS_SPLINE:
        return _fallback_smooth(segment, max(1, min(4, (m - 1) // 2)))

    y = np.asarray(segment, dtype=float)
    t = np.arange(m, dtype=float)
    best = y.copy()
    best_gcv = np.inf
    best_factor = 1.0

    # Select smoothing strength once; the robust stage only changes
    # observation leverage and therefore has a strictly bounded cost.
    for factor in (0.15, 0.3, 0.6, 1.0, 1.7, 2.8, 4.5, 7.0):
        try:
            spl = UnivariateSpline(
                t, y, s=float(m * sigma * sigma * factor),
                k=min(3, m - 1)
            )
            candidate = np.asarray(spl(t), dtype=float)
            if not np.all(np.isfinite(candidate)):
                continue
            df = float(len(spl.get_knots()) + len(spl.get_coeffs()))
            mse = float(np.mean((y - candidate) ** 2))
            gcv = (mse + 1e-12) / max(
                (1.0 - min(df / m, 0.95)) ** 2, 0.05
            )
            if gcv < best_gcv:
                best_gcv = gcv
                best = candidate
                best_factor = factor
        except Exception:
            pass

    # Huber IRLS: isolated impulses receive reduced leverage, while the
    # original GCV-selected smoothness is retained for both deterministic
    # refits.  Two passes are enough to avoid excessive computation.
    for _ in range(2):
        residual = y - best
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        robust_scale = max(1.4826 * mad, 0.25 * float(sigma), 1e-12)
        weights = np.minimum(
            1.0,
            2.5 * robust_scale /
            (np.abs(residual - center) + 1e-12)
        )
        # Do not let noisy segment boundaries dominate the fit, but preserve
        # enough endpoint influence to prevent unconstrained spline drift.
        weights[0] = max(float(weights[0]), 0.65)
        weights[-1] = max(float(weights[-1]), 0.65)
        try:
            spl = UnivariateSpline(
                t, y, w=weights,
                s=float(m * sigma * sigma * best_factor),
                k=min(3, m - 1)
            )
            candidate = np.asarray(spl(t), dtype=float)
            if not np.all(np.isfinite(candidate)):
                break
            best = candidate
        except Exception:
            break

    best[0] = y[0]
    best[-1] = y[-1]
    return _prune_prominent_extrema(y, best)


def _full_denoise(x, window_size):
    """Apply two regime-local bilateral passes before the existing spline fit."""
    x = _finite_signal(x)
    n = len(x)
    if n < 4:
        return x.copy()

    d = np.diff(x)
    dmed = float(np.median(d))
    dmad = float(np.median(np.abs(d - dmed))) + 1e-12
    sigma = 1.4826 * dmad / np.sqrt(2.0)
    spread = float(np.std(x)) + 1e-12

    # Only exceptional jumps create a hard regime boundary.
    threshold = max(6.0 * dmad, 2.5 * spread)
    cuts = np.flatnonzero(np.abs(d - dmed) > threshold) + 1
    bounds = np.r_[0, cuts, n]
    out = np.empty(n, dtype=float)

    radius = min(max(3, int(window_size) // 2), 8)
    spatial_scale = max(2.0, float(window_size) / 4.0)
    for left, right in zip(bounds[:-1], bounds[1:]):
        seg = np.asarray(x[left:right], dtype=float).copy()
        if len(seg) >= 3:
            value_scale = max(
                1.5 * sigma,
                0.08 * float(np.percentile(seg, 95) -
                             np.percentile(seg, 5)),
                1e-12
            )
            offsets = np.arange(-radius, radius + 1, dtype=int)
            spatial = np.exp(
                -0.5 * (offsets.astype(float) / spatial_scale) ** 2
            )
            bilateral = seg
            for _ in range(2):
                current = bilateral.copy()
                smoothed = np.empty_like(current)
                for i in range(len(current)):
                    jj = i + offsets
                    jj = np.where(jj < 0, -jj, jj)
                    jj = np.where(jj >= len(current),
                                  2 * len(current) - 2 - jj, jj)
                    jj = np.clip(jj, 0, len(current) - 1)
                    delta = current[jj] - current[i]
                    weights = spatial * np.exp(
                        -0.5 * (delta / value_scale) ** 2
                    )
                    total = float(np.sum(weights))
                    smoothed[i] = (
                        float(np.sum(weights * current[jj])) / total
                        if total > 1e-12 else current[i]
                    )
                bilateral = smoothed
            seg = bilateral
        out[left:right] = _fit_segment(seg, sigma)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def adaptive_filter(x, window_size=20):
    return enhanced_filter_with_trend_preservation(x, window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = np.asarray(x, dtype=float).reshape(-1)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.asarray(_full_denoise(x, window_size)[window_size - 1:], dtype=float)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


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
    return clean_signal + rng.normal(0, noise_level, length), clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level
        )
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