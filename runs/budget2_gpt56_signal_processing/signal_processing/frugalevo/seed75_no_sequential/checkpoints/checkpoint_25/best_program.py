"""
Offline endpoint-aligned robust trend estimator.

The enhanced path uses robust Whittaker smoothing followed by a
prominence-selected, shape-preserving turning-point skeleton.  The returned
suffix begins at window_size - 1 and therefore has exactly the required length.
"""
import numpy as np

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from scipy.signal import peak_prominences, savgol_filter
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def adaptive_filter(x, window_size=20):
    """Baseline trailing-window moving average."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.mean(np.lib.stride_tricks.sliding_window_view(x, window_size), axis=1)


def _finite_signal(x):
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if not np.all(good):
        ind = np.arange(x.size)
        x[~good] = np.interp(ind[~good], ind[good], x[good])
    return x


def _fallback_smoother(x, span):
    if span <= 1:
        return x.copy()
    kernel = np.ones(span, dtype=float) / span
    left = span // 2
    right = span - 1 - left
    return np.convolve(np.pad(x, (left, right), mode="reflect"), kernel, mode="valid")


def _turning_point_skeleton(trend):
    """Remove completed shallow excursions and interpolate retained extrema."""
    n = len(trend)
    if n < 5 or np.ptp(trend) <= 1e-12:
        return trend.copy()

    d = np.diff(trend)
    signs = np.sign(d)
    # Carry the most recent nonzero direction over flat samples.
    for i in range(1, signs.size):
        if signs[i] == 0:
            signs[i] = signs[i - 1]
    for i in range(signs.size - 2, -1, -1):
        if signs[i] == 0:
            signs[i] = signs[i + 1]

    maxima = np.flatnonzero((signs[:-1] > 0) & (signs[1:] < 0)) + 1
    minima = np.flatnonzero((signs[:-1] < 0) & (signs[1:] > 0)) + 1
    if maxima.size + minima.size == 0:
        return trend.copy()

    scale = 1.4826 * np.median(np.abs(d - np.median(d)))
    q25, q75 = np.percentile(trend, [25.0, 75.0])
    threshold = max(4.0 * float(scale), 0.035 * float(q75 - q25), 1e-12)

    if _HAVE_SCIPY:
        keep_max = maxima[peak_prominences(trend, maxima)[0] >= threshold]
        keep_min = minima[
            peak_prominences(-trend, minima)[0] >= threshold
        ]
    else:
        # Conservative local excursion proxy if SciPy is unavailable.
        keep_max = maxima[
            (trend[maxima] - np.minimum(trend[maxima - 1], trend[maxima + 1]))
            >= threshold
        ]
        keep_min = minima[
            (np.maximum(trend[minima - 1], trend[minima + 1]) - trend[minima])
            >= threshold
        ]

    knots = np.unique(np.concatenate(([0], keep_max, keep_min, [n - 1])))
    if knots.size < 3:
        return np.interp(np.arange(n), knots, trend[knots])

    if _HAVE_SCIPY:
        candidate = PchipInterpolator(knots, trend[knots], extrapolate=False)(
            np.arange(n)
        )
        if np.all(np.isfinite(candidate)):
            return np.asarray(candidate, dtype=float)
    return np.interp(np.arange(n), knots, trend[knots])


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Estimate a robust trend with Whittaker reweighting, persistent extrema, PCHIP interpolation, and conservative affine calibration."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    x = _finite_signal(x)
    n = x.size
    if n <= 1:
        return x.copy()

    try:
        if not _HAVE_SCIPY:
            raise RuntimeError("SciPy unavailable")

        if n >= 3:
            dd = sparse.diags(
                (np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)),
                (0, 1, 2), shape=(n - 2, n), format="csr"
            )
            curvature = dd.T @ dd
        else:
            curvature = sparse.csr_matrix((n, n))

        # Use moderately relaxed curvature regularization so genuine turns are
        # retained without materially increasing noise-induced extrema.
        # The prominence skeleton below still suppresses shallow reversals.
        lam = max(16.0, 0.75 * (float(window_size) / 2.0) ** 4)
        weights = np.ones(n, dtype=float)
        trend = x.copy()

        for _ in range(3):
            system = sparse.diags(weights, format="csr") + lam * curvature
            trend = np.asarray(spsolve(system, weights * x), dtype=float)
            if trend.shape != x.shape or not np.all(np.isfinite(trend)):
                raise FloatingPointError("non-finite sparse solution")

            residual = x - trend
            centered = residual - np.median(residual)
            sigma = max(
                float(np.median(np.abs(centered)) / 0.6745),
                np.finfo(float).eps,
            )
            scaled = centered / (2.5 * sigma)
            weights = 1.0 / np.sqrt(1.0 + scaled * scaled)

        projected = _turning_point_skeleton(trend)

        z = projected - np.median(projected)
        energy = float(np.dot(z, z))
        if energy <= np.finfo(float).eps:
            smooth = projected
        else:
            xc = x - np.median(x)
            # Prevent calibration from re-amplifying noise removed by the
            # persistence-filtered skeleton.  The lower bound retains
            # responsiveness while the tighter upper bound limits spurious
            # slope reversals and false turning points.
            gain = float(np.clip(np.dot(z, xc) / energy, 0.85, 1.18))
            offset = float(np.median(x - gain * projected))
            smooth = gain * projected + offset

    except Exception:
        span = min(n if n % 2 else n - 1, max(3, int(window_size) | 1))
        if _HAVE_SCIPY and span >= 5:
            smooth = savgol_filter(x, span, min(3, span - 1), mode="interp")
        else:
            smooth = _fallback_smoother(x, max(1, span))

    return _finite_signal(smooth)[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Dispatch to the persistent-extrema enhanced estimator or trailing mean baseline."""
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


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level
        )
    else:
        clean_signal = None

    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if clean_signal is not None:
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = np.asarray(noisy_signal, dtype=float)[delay:]
        m = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:m]
        aligned_clean = aligned_clean[:m]
        aligned_noisy = aligned_noisy[:m]
        correlation = (
            np.corrcoef(filtered_signal, aligned_clean)[0, 1]
            if m > 1 and np.std(filtered_signal) > 0
            and np.std(aligned_clean) > 0 else 0.0
        )
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (
            (noise_before - noise_after) / noise_before
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