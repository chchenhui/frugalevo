"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

The public output convention is intentionally retained: every filtering function
returns samples aligned to input indices window_size-1 through len(x)-1.
"""
import numpy as np


def _as_finite_1d(x):
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        return a
    if np.all(np.isfinite(a)):
        return a
    good = np.isfinite(a)
    if not np.any(good):
        return np.zeros_like(a)
    idx = np.arange(a.size)
    return np.interp(idx, idx[good], a[good])


def adaptive_filter(x, window_size=20):
    """Simple causal moving-average baseline with endpoint alignment."""
    x = _as_finite_1d(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    # valid convolution is naturally aligned to the final input sample in a window.
    return np.convolve(x, kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Estimate a robust zero-phase trend with bounded spline continuation.

    Four noise-calibrated spline budgets are evaluated, each with two Huber
    reweighting passes using a robust outlier threshold. The candidate with the
    best residual, reversal, and curvature proxy is selected, then reversal
    hysteresis suppresses brief noise-induced direction changes.
    """
    x = _as_finite_1d(x)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n <= 2:
        return x[window_size - 1:].copy()

    d = np.diff(x)
    dmed = np.median(d)
    diff_scale = 1.4826 * np.median(np.abs(d - dmed))
    signal_scale = 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-12
    # Difference noise has sqrt(2) larger standard deviation for white noise.
    noise_ratio = max(0.0, diff_scale / (np.sqrt(2.0) * signal_scale))

    # Normalized to Nyquist.  This range favors reversal suppression while
    # retaining medium-frequency content on low-noise records.
    cutoff = float(np.clip(0.12 / (1.0 + 1.8 * noise_ratio), 0.055, 0.12))

    """Estimate a robust global trend with three deterministic spline fits."""
    try:
        from scipy.interpolate import UnivariateSpline

        if n < 4:
            from scipy.signal import savgol_filter

            wl = n if n % 2 else n - 1
            full = (
                savgol_filter(x, wl, min(2, wl - 1), mode="interp")
                if wl >= 3 else x.copy()
            )
        else:
            t = np.arange(n, dtype=float)
            d = np.diff(x)
            dmed = np.median(d)
            sigma = (
                np.median(np.abs(d - dmed))
                / (0.67448975 * np.sqrt(2.0))
            )
            sigma = max(float(sigma), 1e-12)
            order = min(3, n - 1)
            base_budget = float(n * sigma * sigma)

            # Select flexibility by blocked validation rather than in-sample
            # residuals, which otherwise reward over-fitted turning structure.
            multipliers = (0.45, 0.8, 1.35, 2.3)
            validation = []
            for multiplier in multipliers:
                smooth_budget = base_budget * multiplier
                errors = []
                try:
                    for fold in range(5):
                        valid = (np.arange(n) % 5) == fold
                        train = ~valid
                        if np.count_nonzero(train) <= order:
                            continue
                        fit = UnivariateSpline(
                            t[train], x[train],
                            w=np.ones(np.count_nonzero(train), dtype=float),
                            s=smooth_budget, k=order
                        )
                        prediction = np.asarray(fit(t[valid]), dtype=float)
                        if np.all(np.isfinite(prediction)):
                            errors.extend(
                                np.abs(x[valid] - prediction).tolist()
                            )
                    if errors:
                        validation.append(
                            (float(np.median(errors)), multiplier)
                        )
                except Exception:
                    continue

            if validation:
                validation.sort(key=lambda item: item[0])
                selected = validation[0][1]
                smooth_budget = base_budget * selected
                weights = np.ones(n, dtype=float)
                estimate = None
                for _ in range(2):
                    fit = UnivariateSpline(
                        t, x, w=weights, s=smooth_budget, k=order
                    )
                    estimate = np.asarray(fit(t), dtype=float)
                    residual = x - estimate
                    rmed = np.median(residual)
                    robust_scale = 1.4826 * np.median(
                        np.abs(residual - rmed)
                    )
                    threshold = 1.345 * max(sigma, float(robust_scale))
                    weights = np.minimum(
                        1.0, threshold / (np.abs(residual) + 1e-12)
                    )
                if estimate is None or not np.all(np.isfinite(estimate)):
                    raise RuntimeError("selected spline fit failed")
                full = estimate.copy()
            else:
                raise RuntimeError("blocked spline validation failed")
    except Exception:
        # Retain a deterministic centered fallback if spline construction is
        # unavailable or fails on an ill-conditioned short record.
        try:
            from scipy.signal import butter, sosfiltfilt

            sos = butter(3, cutoff, btype="lowpass", output="sos")
            padlen = min(n - 1, 12)
            full = sosfiltfilt(sos, x, padlen=padlen)
        except Exception:
            span = min(
                n if n % 2 else n - 1,
                max(3, 2 * (window_size // 2) + 1)
            )
            if span < 3:
                full = x.copy()
            else:
                kernel = np.ones(span, dtype=float) / span
                pad = span // 2
                xp = np.pad(x, (pad, pad), mode="edge")
                full = np.convolve(xp, kernel, mode="valid")

    full = np.asarray(full, dtype=float)[:n]
    if not np.all(np.isfinite(full)):
        full = x.copy()

    # Blend a bounded residual channel into the spline trend, then select the
    # best deterministic surrogate and apply reversal hysteresis.
    spline = np.asarray(full, dtype=float).copy()
    spline_scale = float(np.std(spline))
    if not np.isfinite(spline_scale):
        spline_scale = 0.0

    def _hysteresis(candidate, deadband_factor):
        """Suppress short opposite excursions while preserving sustained turns."""
        result = np.asarray(candidate, dtype=float).copy()
        delta = np.diff(result)
        center = np.median(delta) if delta.size else 0.0
        scale = 1.4826 * np.median(np.abs(delta - center)) if delta.size else 0.0
        if not np.isfinite(scale) or scale <= 0.0:
            return result
        deadband = deadband_factor * scale
        direction = 0
        pending = 0.0
        for i in range(1, n):
            step = result[i] - result[i - 1]
            if not np.isfinite(step):
                step = 0.0
            step_direction = 1 if step > 0.0 else (-1 if step < 0.0 else 0)
            if direction == 0:
                if step_direction:
                    direction = step_direction
                continue
            if step_direction == 0 or step_direction == direction:
                pending = 0.0
            else:
                pending += step
                if abs(pending) >= deadband:
                    direction = step_direction
                    pending = 0.0
                else:
                    result[i] = result[i - 1]
        return result

    candidates = []
    for alpha in (0.0, 0.08, 0.16, 0.24, 0.32):
        blended = spline + alpha * (x - spline)
        if not np.all(np.isfinite(blended)):
            continue
        for deadband_factor in (0.70, 0.90, 1.10):
            candidate = _hysteresis(blended, deadband_factor)
            if not np.all(np.isfinite(candidate)):
                continue
            if spline_scale > 0.0 and np.std(candidate) < 0.35 * spline_scale:
                continue
            residual = candidate - x
            rd = np.diff(candidate)
            rcenter = np.median(residual)
            dcenter = np.median(rd) if rd.size else 0.0
            mad_residual = np.median(np.abs(residual - rcenter))
            mad_slope = (
                np.median(np.abs(rd - dcenter)) if rd.size else 0.0
            )
            reversal_rate = (
                np.mean(np.sign(rd[1:]) != np.sign(rd[:-1]))
                if rd.size > 1 else 0.0
            )
            surrogate = (
                0.55 * mad_residual
                + 0.30 * mad_slope
                + 0.15 * reversal_rate
            )
            candidates.append((float(surrogate), candidate))

    if candidates:
        candidates.sort(key=lambda item: item[0])
        full = candidates[0][1]
    else:
        full = spline

    return full[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply the selected valid-length filter."""
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """Generate the original synthetic signal."""
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
    """Run filtering and retain the original result-dictionary interface."""
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
        noisy_signal = _as_finite_1d(noisy_signal)
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
        np.corrcoef(filtered_signal, aligned_clean)[0, 1]
        if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0
        else 0
    )
    noise_before = np.var(aligned_noisy - aligned_clean)
    noise_after = np.var(filtered_signal - aligned_clean)
    noise_reduction = (
        (noise_before - noise_after) / noise_before if noise_before > 0 else 0
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