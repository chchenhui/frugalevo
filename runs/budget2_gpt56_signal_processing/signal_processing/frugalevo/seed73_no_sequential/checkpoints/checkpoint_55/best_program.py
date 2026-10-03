"""
Offline endpoint-aligned reconstruction using a small deterministic
smoothness/tracking candidate bank and persistent polyline reconstruction.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter, butter, sosfiltfilt
except Exception:
    savgol_filter = None
    butter = None
    sosfiltfilt = None

try:
    from scipy.interpolate import PchipInterpolator, make_lsq_spline, BSpline
except Exception:
    PchipInterpolator = None
    make_lsq_spline = None
    BSpline = None


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


def _robust_spline(piece, window_size, frame):
    """Fit a robust cubic B-spline with curvature regularization and IRLS weighting."""
    value = np.asarray(piece, dtype=float)
    n = value.size
    if n < 8 or make_lsq_spline is None or BSpline is None:
        return _smooth_piece(value, frame)

    tgrid = np.arange(n, dtype=float)
    d = np.diff(value)
    scale = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2.0)
    scale = max(scale, 1e-8)
    spacing = max(4, int(window_size) // 2)
    interior = np.arange(spacing, n - spacing, spacing, dtype=float)
    knots = np.r_[
        np.repeat(0.0, 4),
        interior,
        np.repeat(float(n - 1), 4),
    ]
    ncoef = knots.size - 4
    if ncoef < 4:
        return _smooth_piece(value, frame)

    # Three fixed smoothness levels provide robust curvature control without
    # introducing an expensive optimizer or additional candidate portfolio.
    strengths = (0.4, 1.6, 6.4)
    best = None
    best_cost = np.inf
    weights = np.ones(n, dtype=float)

    for strength in strengths:
        current = value.copy()
        for _ in range(4):
            try:
                basis = np.empty((n, ncoef), dtype=float)
                for j in range(ncoef):
                    coeff = np.zeros(ncoef, dtype=float)
                    coeff[j] = 1.0
                    basis[:, j] = BSpline(knots, coeff, 3)(tgrid)

                # Penalize coefficient curvature by augmenting the weighted
                # least-squares system with second-difference pseudo-observations.
                reg = np.zeros((max(0, ncoef - 2), ncoef), dtype=float)
                if reg.size:
                    rows = np.arange(ncoef - 2)
                    reg[rows, rows] = 1.0
                    reg[rows, rows + 1] = -2.0
                    reg[rows, rows + 2] = 1.0
                lam = float(strength) * scale * scale * n
                aw = basis * weights[:, None]
                bw = current * weights
                if reg.size:
                    aw = np.vstack((aw, np.sqrt(lam) * reg))
                    bw = np.r_[bw, np.zeros(reg.shape[0])]
                coef = np.linalg.lstsq(aw, bw, rcond=None)[0]
                fitted = np.asarray(BSpline(knots, coef, 3)(tgrid), dtype=float)
                residual = value - fitted
                robust = np.abs(residual) / (2.5 * scale + 1e-12)
                weights = 1.0 / np.maximum(1.0, robust)
                current = fitted
            except Exception:
                current = _smooth_piece(value, frame)
                break

        curvature = np.diff(current, n=2)
        active = np.abs(curvature) > 0.35 * scale
        changes = int(np.count_nonzero(
            np.sign(curvature[1:][active[1:]]) !=
            np.sign(curvature[:-1][active[:-1]])
        )) if curvature.size > 1 else 0
        error = float(np.mean(np.abs(value - current)))
        cost = error / (scale + 1e-12) + 0.08 * changes
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best = current.copy()

    return np.asarray(best if best is not None else value, dtype=float)


def _reconstruct_candidate(x, window_size, frame, epsilon_scale):
    n = x.size
    spread = float(np.percentile(x, 95) - np.percentile(x, 5))
    out = np.empty(n, dtype=float)
    bounds = _boundaries(x, window_size)

    for left, right in zip(bounds[:-1], bounds[1:]):
        piece = x[left:right]
        if piece.size == 0:
            continue
        smooth = _robust_spline(piece, window_size, frame)
        residual = piece - smooth
        center = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-12
        out[left:right] = smooth

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
    """Score endpoint error, directional reversals, and sub-threshold chatter."""
    if y.size < 3:
        return 0.0
    err = float(np.mean(np.abs(y - endpoint_x)))
    dref = np.diff(endpoint_x)
    center = float(np.median(dref))
    scale = 1.4826 * float(np.median(np.abs(dref - center))) + 1e-12
    span = float(np.percentile(endpoint_x, 95) - np.percentile(endpoint_x, 5))
    threshold = max(2.5 * scale, 0.025 * span, 1e-10)

    dy = np.diff(y)
    signs = np.sign(dy).astype(float, copy=True)
    nz = np.flatnonzero(signs)
    if nz.size == 0:
        reversals = 0
    else:
        signs[:nz[0]] = signs[nz[0]]
        for i in range(nz[0] + 1, signs.size):
            if signs[i] == 0.0:
                signs[i] = signs[i - 1]
        reversals = int(np.count_nonzero(signs[1:] != signs[:-1]))

    quiet = np.abs(dy) <= threshold
    quiet_energy = float(np.mean(dy[quiet] * dy[quiet])) if np.any(quiet) else 0.0
    error_term = min(err / max(threshold, 1e-10), 2.0)
    reversal_term = min(reversals / max(0.08 * y.size, 1.0), 2.0)
    chatter_term = min(quiet_energy / (threshold * threshold), 2.0)
    return 0.50 * error_term + 0.35 * reversal_term + 0.15 * chatter_term


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Use jump-isolated zero-phase Butterworth envelopes with SG fallbacks."""
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

    # Four fixed spectral envelopes provide bounded attenuation choices.
    bounds = _boundaries(x, window_size)
    delay = window_size - 1
    endpoint_x = x[delay:]
    candidates = []

    for cutoff in (0.09, 0.14, 0.20, 0.28):
        candidate = np.empty_like(x, dtype=float)
        try:
            sos = butter(3, cutoff, output="sos")
        except Exception:
            sos = None

        for left, right in zip(bounds[:-1], bounds[1:]):
            piece = np.asarray(_median3(x[left:right]), dtype=float)
            if piece.size == 0:
                continue
            filtered = None
            if sos is not None and sosfiltfilt is not None:
                try:
                    filtered = np.asarray(
                        sosfiltfilt(sos, piece), dtype=float
                    )
                except Exception:
                    filtered = None
            if filtered is None:
                if savgol_filter is not None and piece.size >= 5:
                    frame = min(21, piece.size if piece.size % 2 else piece.size - 1)
                    try:
                        filtered = np.asarray(
                            savgol_filter(
                                piece, frame, min(3, frame - 2), mode="interp"
                            ),
                            dtype=float,
                        )
                    except Exception:
                        filtered = piece.copy()
                else:
                    filtered = piece.copy()
            candidate[left:right] = filtered
        candidates.append(candidate)

    # Retain one incumbent reconstruction as a numerically safe fallback.
    candidates.append(_reconstruct_candidate(x, window_size, 21, 0.95))
    best = None
    best_cost = np.inf
    for candidate in candidates:
        cropped = np.asarray(candidate[delay:], dtype=float)
        if cropped.size != endpoint_x.size or not np.all(np.isfinite(cropped)):
            continue
        cost = _candidate_cost(cropped, endpoint_x)
        if cost < best_cost:
            best_cost = cost
            best = cropped

    if best is None:
        best = endpoint_x.copy()
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