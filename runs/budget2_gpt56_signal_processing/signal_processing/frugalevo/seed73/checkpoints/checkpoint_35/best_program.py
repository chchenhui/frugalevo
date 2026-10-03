"""
Batch-capable adaptive processing for non-stationary one-dimensional signals.

The output corresponds to the input samples at indices window_size-1 onward.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAVE_SCIPY = True
except Exception:
    savgol_filter = None
    _HAVE_SCIPY = False


def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1 or x.size < window_size:
        raise ValueError(
            f"Input signal length ({x.size}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size, dtype=float) / window_size, "valid")


def _finite_signal(x):
    z = np.asarray(x, dtype=float).copy()
    good = np.isfinite(z)
    if not np.any(good):
        return np.zeros_like(z)
    if not np.all(good):
        ind = np.arange(z.size)
        z[~good] = np.interp(ind[~good], ind[good], z[good])
    return z


def _robust_sigma(v):
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return np.finfo(float).eps
    med = float(np.median(v))
    return max(1.4826 * float(np.median(np.abs(v - med))),
               np.finfo(float).eps)


def _median3(z):
    if z.size < 3:
        return z.copy()
    p = np.pad(z, (1, 1), mode="edge")
    return np.median(np.stack((p[:-2], p[1:-1], p[2:])), axis=0)


def _savgol_estimate(z, span):
    half = span // 2
    if _HAVE_SCIPY and savgol_filter is not None:
        return np.asarray(savgol_filter(z, span, 3, mode="mirror"), dtype=float)

    u = np.arange(-half, half + 1, dtype=float)
    design = np.column_stack((np.ones(span), u, u * u, u * u * u))
    weights = np.linalg.pinv(design)[0]
    padded = np.pad(z, (half, half), mode="reflect")
    return np.convolve(padded, weights[::-1], mode="valid")


def _triangular_fallback(z, window_size):
    """Multi-span centered cubic local-polynomial trend estimate."""
    z = np.asarray(z, dtype=float)
    n = z.size
    if n < 5:
        return z.copy()

    maximum = n if n % 2 else n - 1
    base = max(5, 2 * (int(window_size) // 2) + 1)
    requested = (base, max(7, 2 * int(window_size) + 1),
                 max(9, 3 * int(window_size) + 1))
    spans = sorted(set(min(maximum, s if s % 2 else s - 1) for s in requested))
    spans = [s for s in spans if s >= 5]
    if not spans:
        return z.copy()

    noise = _robust_sigma(np.diff(z))
    estimates = []
    for span in spans:
        y = _savgol_estimate(z, span)
        estimates.append((_robust_sigma(z - y), y))

    acceptable = [y for residual, y in estimates if residual >= 0.40 * noise]
    return acceptable[-1] if acceptable else estimates[0][1]


def _turn_indices(y):
    d = np.diff(y)
    if d.size < 2:
        return []
    s = np.sign(d).astype(np.int8)

    last = np.int8(0)
    for i in range(s.size):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]

    last = np.int8(0)
    for i in range(s.size - 1, -1, -1):
        if s[i] == 0:
            s[i] = last
        else:
            last = s[i]

    return [i for i in range(1, s.size) if s[i - 1] * s[i] < 0]


def _hysteretic_projection(candidate, residual_sigma, window_size):
    y = np.asarray(candidate, dtype=float).copy()
    n = y.size
    if n < 5 or not np.all(np.isfinite(y)):
        return y

    short_span = max(5, window_size // 2 + 1)
    sigma = max(float(residual_sigma), np.finfo(float).eps)

    for _ in range(n):
        turns = _turn_indices(y)
        if not turns:
            break
        knots = [0] + turns + [n - 1]
        removed = False

        for j in range(1, len(knots) - 1):
            a, k, b = knots[j - 1], knots[j], knots[j + 1]
            span = b - a
            if span < 2:
                continue
            fraction = (k - a) / float(span)
            baseline = y[a] + fraction * (y[b] - y[a])
            prominence = abs(y[k] - baseline)
            threshold = 1.15 * sigma * (1.0 + 0.15 * np.sqrt(span))

            if ((span <= short_span and prominence < threshold) or
                    prominence < 0.45 * sigma):
                y[a:b + 1] = np.linspace(y[a], y[b], span + 1)
                removed = True
                break

        if not removed:
            break

    y[0] = candidate[0]
    y[-1] = candidate[-1]
    return y


def _harmonic_candidate(z, robust, local, step_dominant):
    """Select a robust Fourier or variable-projection quadratic-phase trend."""
    n = z.size
    if step_dominant or n < 24:
        return None

    try:
        from scipy.optimize import minimize
    except Exception:
        minimize = None

    t = np.arange(n, dtype=float)
    u = (t - 0.5 * (n - 1)) / max(n - 1, 1)
    raw_sigma = _robust_sigma(z - robust)
    eps = np.finfo(float).eps

    # Fit a small fundamental-frequency lattice, selecting order by BIC.
    # This represents harmonically related components without assuming one
    # cycle per record, while the turn surcharge rejects noise-like fits.
    max_order = min(10, max(1, n // 16))
    centered = robust - np.mean(robust)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(spectrum[1:]))
    best = None
    best_bic = np.inf
    if max_order >= 1 and np.isfinite(total) and total > eps:
        ranked = np.argsort(spectrum[1:])[-4:] + 1
        bases = sorted(set([1, 2, 3, 4, 5, 6, 7, 8] +
                           [int(q) for q in ranked if int(q) <= 8]))
        for base in bases:
            base_best = None
            base_bic = np.inf
            phase = 2.0 * np.pi * float(base) * t / float(n)
            for order in range(1, max_order + 1):
                columns = [np.ones(n), u]
                for h in range(1, order + 1):
                    columns.extend((np.cos(h * phase), np.sin(h * phase)))
                design = np.column_stack(columns)
                try:
                    coef, _, _, _ = np.linalg.lstsq(design, robust, rcond=None)
                    fitted = design @ coef
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    continue
                if not np.all(np.isfinite(fitted)):
                    continue
                residual = robust - fitted
                scale = _robust_sigma(residual)
                if scale < 0.25 * raw_sigma:
                    continue
                rss = float(np.dot(residual, residual))
                bic = (n * np.log(max(rss / n, eps)) +
                       design.shape[1] * np.log(n))
                if bic < base_bic:
                    base_bic, base_best = bic, fitted
            if base_best is not None:
                turns = len(_turn_indices(base_best))
                penalized = base_bic + 0.5 * np.log(n) * turns
                if penalized < best_bic:
                    best_bic, best = penalized, base_best

    if minimize is None:
        return best

    # Variable projection: nonlinear phase parameters, linear amplitudes.
    def projected(theta, return_fit=False):
        f, chirp = float(theta[0]), float(theta[1])
        phase = 2.0 * np.pi * (f * u + 0.5 * chirp * u * u)
        design = np.column_stack((
            np.ones(n), u, np.sin(phase), np.cos(phase)
        ))
        weights = np.ones(n, dtype=float)
        coef = None
        for _ in range(3):
            try:
                coef, _, _, _ = np.linalg.lstsq(
                    design * weights[:, None], robust * weights, rcond=None
                )
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                return np.inf if not return_fit else None
            residual = robust - design @ coef
            scale = max(_robust_sigma(residual), eps)
            q = np.abs(residual) / (1.5 * scale)
            weights = np.where(q <= 1.0, 1.0, 1.0 / np.maximum(q, 1.0))
        fitted = design @ coef
        residual = robust - fitted
        # Huber objective makes frequency selection resistant to outliers.
        scale = max(_robust_sigma(residual), eps)
        q = np.abs(residual) / scale
        loss = np.where(q <= 1.5, 0.5 * q * q, 1.5 * q - 1.125)
        value = float(np.sum(loss))
        return fitted if return_fit else value

    power = spectrum[1:max_order + 1] if total > eps else np.array([])
    bins = (np.argsort(power)[-3:] + 1) if power.size else np.array([1])
    starts = [(float(k), 0.0) for k in bins]
    starts.extend(((0.5, -1.5), (max(1.0, n / 8.0), 1.5)))
    chirp_fit = None
    chirp_bic = np.inf
    for start in starts[:5]:
        try:
            result = minimize(
                projected, np.asarray(start, dtype=float), method="L-BFGS-B",
                bounds=((0.25, max(2.0, n / 3.0)), (-3.0, 3.0)),
                options={"maxiter": 30, "ftol": 1e-9}
            )
            fitted = projected(result.x, return_fit=True)
        except (ValueError, TypeError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if fitted is None or not np.all(np.isfinite(fitted)):
            continue
        residual = robust - fitted
        scale = _robust_sigma(residual)
        if scale < 0.25 * raw_sigma:
            continue
        rss = float(np.dot(residual, residual))
        bic = n * np.log(max(rss / n, eps)) + 4.0 * np.log(n)
        if bic < chirp_bic:
            chirp_bic, chirp_fit = bic, fitted

    if chirp_fit is not None:
        local_turns = len(_turn_indices(local))
        chirp_turns = len(_turn_indices(chirp_fit))
        if (chirp_bic + 2.0 * np.log(n) < best_bic and
                chirp_turns <= local_turns):
            return chirp_fit

    if best is None:
        return None
    local_mae = float(np.median(np.abs(z - local)))
    global_mae = float(np.median(np.abs(z - best)))
    if (global_mae <= 1.35 * max(local_mae, eps) and
            len(_turn_indices(best)) < len(_turn_indices(local))):
        return best
    return None


def enhanced_filter_with_trend_preservation(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 1 or x.size < window_size:
        raise ValueError(
            f"Input signal length ({x.size}) must be >= window_size ({window_size})"
        )

    z = _finite_signal(x)
    if z.size <= 3:
        return z[window_size - 1:].copy()

    robust = _median3(z)
    candidate = _triangular_fallback(robust, window_size)

    dz = np.diff(robust)
    scale = _robust_sigma(dz)
    step_dominant = (
        dz.size > 4 and
        np.max(np.abs(dz)) > 7.0 * scale and
        np.mean(np.abs(dz) > 3.0 * scale) < 0.08
    )

    if step_dominant and z.size >= 5:
        jump = int(np.argmax(np.abs(np.diff(robust)))) + 1
        left = _triangular_fallback(z[:jump], window_size)
        right = _triangular_fallback(z[jump:], window_size)
        if left.size and right.size:
            candidate = np.concatenate((left, right))
    else:
        harmonic = _harmonic_candidate(z, robust, candidate, step_dominant)
        if harmonic is not None:
            candidate = harmonic

    if candidate.size != z.size or not np.all(np.isfinite(candidate)):
        candidate = _triangular_fallback(z, window_size)

    residual_sigma = _robust_sigma(z - candidate)
    filtered = _hysteretic_projection(candidate, residual_sigma, window_size)
    if not np.all(np.isfinite(filtered)):
        filtered = candidate

    return filtered[window_size - 1:].copy()


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


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
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
    aligned_clean = clean_signal[delay:delay + len(filtered_signal)]
    aligned_noisy = noisy_signal[delay:delay + len(filtered_signal)]
    m = min(len(filtered_signal), len(aligned_clean))

    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0:
        correlation = float(np.corrcoef(filtered_signal, aligned_clean)[0, 1])
    else:
        correlation = 0.0

    noise_before = float(np.var(aligned_noisy - aligned_clean))
    noise_after = float(np.var(filtered_signal - aligned_clean))
    noise_reduction = (
        (noise_before - noise_after) / noise_before if noise_before > 0 else 0.0
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