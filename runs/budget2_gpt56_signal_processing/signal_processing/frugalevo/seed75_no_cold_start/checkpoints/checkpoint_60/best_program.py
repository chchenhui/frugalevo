"""
Offline zero-phase regime-aware denoising.

The returned signal is aligned to the final sample of each nominal sliding
window, preserving the required output length of len(x) - window_size + 1.
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
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()
    radius = max(1, min(int(radius), (n - 1) // 2))
    up = np.arange(1, radius + 2, dtype=float)
    k = np.r_[up, up[-2::-1]]
    k /= k.sum()
    p = len(k) // 2
    return np.convolve(np.pad(x, p, mode="reflect"), k, mode="valid")


def _turns(z):
    d = np.diff(np.asarray(z, dtype=float))
    if len(d) < 2:
        return np.empty(0, dtype=int)
    s = np.sign(d)
    for i in range(1, len(s)):
        if s[i] == 0:
            s[i] = s[i - 1]
    for i in range(len(s) - 2, -1, -1):
        if s[i] == 0:
            s[i] = s[i + 1]
    return np.flatnonzero(s[:-1] * s[1:] < 0) + 1


def _prune_prominent_extrema(y, z):
    """Remove small, narrow spline reversals without moving endpoints."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float).copy()
    m = len(z)
    if m < 7:
        return z

    r = y - z
    r -= np.median(r)
    sigma = 1.4826 * float(np.median(np.abs(r)))
    span = float(np.percentile(y, 95) - np.percentile(y, 5))
    base = max(1.12 * sigma, 0.055 * span, 1e-12)

    for _ in range(3):
        tr = _turns(z)
        if len(tr) == 0:
            break
        ext = np.r_[0, tr, m - 1]
        mid = ext[1:-1]
        prom = np.minimum(np.abs(z[mid] - z[ext[:-2]]),
                          np.abs(z[ext[2:]] - z[mid]))
        width = np.minimum(mid - ext[:-2], ext[2:] - mid)
        weak = (prom < base) | ((width <= 2) & (prom < 2.0 * base))
        if not np.any(weak):
            break
        score = prom / base + 0.35 * width.astype(float)
        q = np.flatnonzero(weak)
        j = int(q[np.argmin(score[q])]) + 1
        left, right = int(ext[j - 1]), int(ext[j + 1])
        z[left:right + 1] = np.linspace(z[left], z[right], right - left + 1)

    z[0] = y[0]
    z[-1] = y[-1]
    return z


def _coherent_slope_multiplier(y):
    """
    Elevate spline leverage on sustained same-direction slope runs.  Isolated
    spikes have no adjacent agreement and retain ordinary robust weighting.
    """
    m = len(y)
    gain = np.ones(m, dtype=float)
    if m < 5:
        return gain

    d = np.diff(y)
    center = float(np.median(d))
    ds = 1.4826 * float(np.median(np.abs(d - center)))
    ds = max(ds, 1e-12)
    active = np.abs(d - center) > 2.0 * ds
    signs = np.sign(d - center)

    i = 0
    while i < len(d):
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < len(d) and active[j] and signs[j] == signs[i]:
            j += 1
        if j - i >= 2:
            idx = np.arange(i, j + 1)
            # Smoothly emphasize the run interior, bounded for stability.
            if len(idx) > 2:
                position = np.linspace(0.0, 1.0, len(idx))
                shape = 1.0 - np.abs(2.0 * position - 1.0)
            else:
                shape = np.ones(len(idx))
            gain[idx] = np.maximum(gain[idx], 1.0 + 0.8 * shape)
        i = j
    return gain


def _fit_segment(segment, sigma):
    """Fit two robust cubic splines using additive leverage for coherent slopes."""
    y = np.asarray(segment, dtype=float)
    m = len(y)
    if m < 8:
        return _fallback_smooth(y, max(1, (m - 1) // 3)) if m >= 3 else y.copy()
    if not _HAS_SPLINE:
        return _fallback_smooth(y, max(1, min(4, (m - 1) // 2)))

    t = np.arange(m, dtype=float)
    best = y.copy()
    best_gcv = np.inf
    factor_best = 1.0

    for factor in (0.15, 0.3, 0.6, 1.0, 1.7, 2.8, 4.5, 7.0):
        try:
            spl = UnivariateSpline(
                t, y, s=float(m * sigma * sigma * factor), k=min(3, m - 1)
            )
            cand = np.asarray(spl(t), dtype=float)
            if not np.all(np.isfinite(cand)):
                continue
            df = float(len(spl.get_knots()) + len(spl.get_coeffs()))
            mse = float(np.mean((y - cand) ** 2))
            gcv = (mse + 1e-12) / max(
                (1.0 - min(df / m, 0.95)) ** 2, 0.05
            )
            if gcv < best_gcv:
                best_gcv = gcv
                best = cand
                factor_best = factor
        except Exception:
            continue

    coherent = _coherent_slope_multiplier(y)
    for _ in range(2):
        residual = y - best
        med = float(np.median(residual))
        scale = max(
            1.4826 * float(np.median(np.abs(residual - med))),
            0.25 * float(sigma),
            1e-12,
        )
        huber = np.minimum(1.0, 2.5 * scale / (np.abs(residual - med) + 1e-12))

        # Add bounded leverage for persistent same-sign slope runs instead
        # of multiplying it into Huber weights.  This prevents a genuine
        # sharp feature from being discarded solely because its residual is
        # temporarily large, while isolated excursions remain suppressed.
        # Apply stronger bounded feature leverage to coherent ramps while
        # retaining Huber suppression for isolated residual excursions.
        weights = huber + 0.60 * (coherent - 1.0)
        weights /= max(float(np.median(weights)), 1e-12)
        weights = np.clip(weights, 0.10, 1.80)
        weights[0] = max(weights[0], 0.65)
        weights[-1] = max(weights[-1], 0.65)
        try:
            spl = UnivariateSpline(
                t,
                y,
                w=weights,
                s=float(m * sigma * sigma * factor_best),
                k=min(3, m - 1),
            )
            cand = np.asarray(spl(t), dtype=float)
            if np.all(np.isfinite(cand)):
                best = cand
        except Exception:
            break

    best[0] = y[0]
    best[-1] = y[-1]
    return _prune_prominent_extrema(y, best)


def _bilateral_segment(seg, radius, spatial_scale, sigma):
    seg = np.asarray(seg, dtype=float)
    n = len(seg)
    if n < 3:
        return seg.copy()

    value_scale = max(
        1.5 * sigma,
        0.06 * float(np.percentile(seg, 95) - np.percentile(seg, 5)),
        1e-12,
    )
    offsets = np.arange(-radius, radius + 1)
    spatial = np.exp(-0.5 * (offsets.astype(float) / spatial_scale) ** 2)
    current = seg.copy()

    for _ in range(2):
        out = np.empty(n, dtype=float)
        for i in range(n):
            jj = i + offsets
            jj = np.where(jj < 0, -jj, jj)
            jj = np.where(jj >= n, 2 * n - 2 - jj, jj)
            jj = np.clip(jj, 0, n - 1)
            delta = seg[jj] - seg[i]
            w = spatial * np.exp(-0.5 * (delta / value_scale) ** 2)
            sw = float(np.sum(w))
            out[i] = float(np.sum(w * current[jj]) / sw) if sw > 1e-12 else current[i]
        out[0] = seg[0]
        out[-1] = seg[-1]
        current = out
    return current


def _full_denoise(x, window_size):
    x = _finite_signal(x)
    n = len(x)
    if n < 4:
        return x.copy()

    d = np.diff(x)
    dmed = float(np.median(d))
    dmad = float(np.median(np.abs(d - dmed))) + 1e-12
    sigma = 1.4826 * dmad / np.sqrt(2.0)
    spread = float(np.std(x)) + 1e-12

    threshold = max(6.0 * dmad, 2.5 * spread)
    cuts = np.flatnonzero(np.abs(d - dmed) > threshold) + 1
    bounds = np.r_[0, cuts, n]
    out = np.empty(n, dtype=float)

    radius = min(max(2, int(window_size) // 3), 6)
    spatial_scale = max(1.75, float(window_size) / 5.0)

    for left, right in zip(bounds[:-1], bounds[1:]):
        raw = x[left:right]
        pre = _bilateral_segment(raw, radius, spatial_scale, sigma)
        out[left:right] = _fit_segment(pre, sigma)

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
    clean = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean += 0.1 * t * np.sin(0.2 * t)
    clean += np.cumsum(rng.randn(length) * 0.05)
    return clean + rng.normal(0, noise_level, length), clean


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