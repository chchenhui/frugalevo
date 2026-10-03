"""
Offline endpoint-aligned reconstruction using a small deterministic
smoothness/tracking candidate bank and persistent polyline reconstruction.
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
    idx = np.arange(x.size)
    return np.interp(idx, idx[good], x[good])


def _median3(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x.copy()
    p = np.pad(x, (1, 1), mode="edge")
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


def _smooth_piece(piece, requested_frame=21):
    """Apply bounded EMD-style noise-IMF rejection, with Savitzky-Golay fallback."""
    clean = np.asarray(_median3(piece), dtype=float)
    n = clean.size

    def sg_fallback(value):
        value = np.asarray(value, dtype=float)
        if savgol_filter is None or value.size < 5:
            return value.copy()
        frame = min(int(requested_frame), value.size if value.size % 2 else value.size - 1)
        if frame < 5:
            return value.copy()
        try:
            result = np.asarray(
                savgol_filter(
                    value,
                    window_length=frame,
                    polyorder=min(3, frame - 2),
                    mode="interp",
                ),
                dtype=float,
            )
            return result if np.all(np.isfinite(result)) else value.copy()
        except Exception:
            return value.copy()

    if n < 32 or PchipInterpolator is None:
        return sg_fallback(clean)

    def extrema_envelope(value, maxima):
        """Construct a finite PCHIP envelope through extrema and endpoints."""
        value = np.asarray(value, dtype=float)
        if maxima:
            points = np.flatnonzero(
                (value[1:-1] >= value[:-2]) &
                (value[1:-1] > value[2:])
            ) + 1
        else:
            points = np.flatnonzero(
                (value[1:-1] <= value[:-2]) &
                (value[1:-1] < value[2:])
            ) + 1

        anchors = np.unique(np.concatenate((
            np.array([0, value.size - 1], dtype=int),
            points.astype(int),
        )))
        if anchors.size < 2:
            return None

        # Endpoint anchors prevent envelope extrapolation and preserve segment length.
        try:
            envelope = np.asarray(
                PchipInterpolator(
                    anchors.astype(float),
                    value[anchors].astype(float),
                    extrapolate=False,
                )(np.arange(value.size, dtype=float)),
                dtype=float,
            )
        except Exception:
            return None
        return envelope if np.all(np.isfinite(envelope)) else None

    def sift_mode(value):
        """Perform at most four bounded EMD sifts for one candidate IMF."""
        mode = np.asarray(value, dtype=float).copy()
        for _ in range(4):
            upper = extrema_envelope(mode, True)
            lower = extrema_envelope(mode, False)
            if upper is None or lower is None:
                return None
            mean_envelope = 0.5 * (upper + lower)
            updated = mode - mean_envelope
            if not np.all(np.isfinite(updated)):
                return None
            if np.max(np.abs(updated - mode)) <= 0.01 * (
                np.std(mode) + 1e-12
            ):
                mode = updated
                break
            mode = updated
        return mode

    residual = clean.copy()
    retained = []
    difference = np.diff(clean)
    diff_center = float(np.median(difference))
    sigma_d = 1.4826 * float(
        np.median(np.abs(difference - diff_center))
    ) + 1e-12
    half_period_limit = max(4, int(requested_frame) // 2)

    # Extract no more than three modes; rejected fast modes are intentionally
    # omitted, while retained modes and the final residual reconstruct the trend.
    for _ in range(3):
        mode = sift_mode(residual)
        if mode is None:
            return sg_fallback(clean)

        crossings = np.flatnonzero(
            (mode[:-1] * mode[1:] <= 0.0)
        )
        amplitude = 1.4826 * float(
            np.median(np.abs(mode - np.median(mode)))
        )
        if crossings.size >= 2:
            periods = np.diff(crossings).astype(float) * 2.0
            median_half_period = float(np.median(periods))
        else:
            median_half_period = float(n)

        reject = (
            amplitude < 1.35 * sigma_d * np.sqrt(float(n)) and
            median_half_period < float(half_period_limit)
        )
        if not reject:
            retained.append(mode.copy())
        residual = residual - mode

        # A nearly monotone residual has no additional IMF to extract.
        if np.count_nonzero(np.diff(np.sign(np.diff(residual)))) < 2:
            break

    result = residual.copy()
    for mode in retained:
        result += mode
    if not np.all(np.isfinite(result)):
        return sg_fallback(clean)
    return np.asarray(result, dtype=float)


def _persistent_shape(y, sigma, spread, epsilon_scale=0.95):
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return y.copy()

    epsilon = max(float(epsilon_scale) * float(sigma), 0.008 * float(spread), 1e-12)
    max_interior = min(96, n // 2)
    selected = [0, n - 1]
    pending = []

    def best_split(left, right):
        if right - left < 2:
            return 0.0, -1
        ii = np.arange(left + 1, right)
        frac = (ii - left) / float(right - left)
        chord = y[left] + frac * (y[right] - y[left])
        residual = np.abs(y[ii] - chord)
        pos = int(np.argmax(residual))
        return float(residual[pos]), int(ii[pos])

    dev, at = best_split(0, n - 1)
    if at >= 0 and dev > epsilon:
        pending.append((-dev, 0, n - 1, at))

    while pending and len(selected) - 2 < max_interior:
        pending.sort(key=lambda z: z[0])
        _, left, right, at = pending.pop(0)
        if at in selected:
            continue
        selected.append(at)
        dev, split = best_split(left, at)
        if split >= 0 and dev > epsilon:
            pending.append((-dev, left, at, split))
        dev, split = best_split(at, right)
        if split >= 0 and dev > epsilon:
            pending.append((-dev, at, right, split))

    knots = np.asarray(sorted(set(selected)), dtype=int)
    grid = np.arange(n, dtype=float)
    if knots.size < 3 or PchipInterpolator is None:
        return np.interp(grid, knots.astype(float), y[knots])
    try:
        out = np.asarray(
            PchipInterpolator(knots.astype(float), y[knots], extrapolate=False)(grid),
            dtype=float,
        )
        if np.all(np.isfinite(out)):
            return out
    except Exception:
        pass
    return np.interp(grid, knots.astype(float), y[knots])


def _boundaries(x, window_size):
    n = x.size
    if n < 4:
        return [0, n]
    d = np.diff(x)
    dmed = float(np.median(d))
    sigma_d = 1.4826 * float(np.median(np.abs(d - dmed))) + 1e-12
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    threshold = max(5.0 * sigma_d, 0.30 * spread)
    candidates = np.flatnonzero(np.abs(d) > threshold) + 1
    result = [0]
    if candidates.size:
        group = [int(candidates[0])]
        for c in candidates[1:]:
            c = int(c)
            if c - group[-1] <= 4:
                group.append(c)
            else:
                result.append(max(group, key=lambda q: abs(d[q - 1])))
                group = [c]
        result.append(max(group, key=lambda q: abs(d[q - 1])))
    result.append(n)
    return sorted(set(int(v) for v in result))


def _reconstruct_candidate(x, window_size, frame, epsilon_scale):
    n = x.size
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    out = np.empty(n, dtype=float)
    bounds = _boundaries(x, window_size)

    for left, right in zip(bounds[:-1], bounds[1:]):
        piece = x[left:right]
        if piece.size == 0:
            continue
        smooth = _smooth_piece(piece, frame)
        residual = piece - smooth
        center = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-12
        out[left:right] = _persistent_shape(smooth, sigma, spread, epsilon_scale)

    emean = float(np.mean(out))
    xmean = float(np.mean(x))
    centered = out - emean
    variance = float(np.mean(centered * centered))
    if np.isfinite(variance) and variance > 1e-14:
        gain = float(np.mean(centered * (x - xmean))) / (variance + 1e-12)
        gain = float(np.clip(gain if np.isfinite(gain) else 1.0, 0.85, 1.20))
        out = gain * out + (xmean - gain * emean)
    else:
        out = out + (xmean - emean)
    return np.nan_to_num(out, nan=xmean, posinf=xmean, neginf=xmean)


def _candidate_cost(y, endpoint_x):
    """No-clean-reference proxy: retain endpoint tracking while pricing reversals."""
    if y.size < 3:
        return 0.0
    err = float(np.mean(np.abs(y - endpoint_x)))
    dref = np.diff(endpoint_x)
    scale = 1.4826 * float(np.median(np.abs(dref - np.median(dref)))) + 1e-12
    span = float(np.percentile(endpoint_x, 95) - np.percentile(endpoint_x, 5))
    denom = max(2.5 * scale, 0.025 * span, 1e-10)

    dy = np.diff(y)
    signs = np.sign(dy)
    nz = np.flatnonzero(signs)
    if nz.size < 2:
        reversals = 0
    else:
        signs[:nz[0]] = signs[nz[0]]
        for i in range(nz[0] + 1, signs.size):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return 0.72 * min(err / denom, 2.0) + 0.28 * min(reversals / 50.0, 2.0)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if x.size < window_size:
        raise ValueError(
            f"Input signal length ({x.size}) must be >= window_size ({window_size})"
        )

    x = _finite_signal(x)
    if x.size < 5:
        return x[window_size - 1:].copy()

    # A compact deterministic bank spans responsive, incumbent-like, and
    # reversal-resistant reconstructions.  Selection uses only observable data.
    settings = ((15, 0.78), (21, 0.95), (31, 1.25))
    delay = window_size - 1
    endpoint_x = x[delay:]
    best = None
    best_cost = np.inf
    for frame, eps in settings:
        candidate = _reconstruct_candidate(x, window_size, frame, eps)
        cropped = candidate[delay:]
        cost = _candidate_cost(cropped, endpoint_x)
        if cost < best_cost:
            best_cost = cost
            best = cropped

    return np.asarray(best, dtype=float)


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