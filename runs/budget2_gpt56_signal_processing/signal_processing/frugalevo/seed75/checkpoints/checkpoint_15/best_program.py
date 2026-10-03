"""
Batch-capable adaptive filtering for volatile non-stationary signals.

The public output is aligned with the right edge of the requested window:
output[j] estimates input sample j + window_size - 1.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter, butter, sosfiltfilt
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


def _finite_signal(x):
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    if x.size == 0:
        return x
    good = np.isfinite(x)
    if not np.any(good):
        return np.zeros_like(x)
    if not np.all(good):
        indices = np.arange(x.size)
        x[~good] = np.interp(indices[~good], indices[good], x[good])
    return x


def adaptive_filter(x, window_size=20):
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(x, kernel, mode="valid")


def _odd_span(value, n):
    span = int(round(value))
    span = max(5, span | 1)
    largest = n if n % 2 else n - 1
    return min(span, largest)


def _reversals(z):
    d = np.diff(z)
    if d.size < 2:
        return 0
    # Ignore numerical near-zero slopes rather than manufacturing a sign change.
    eps = 1e-10 * (np.std(z) + 1.0)
    s = np.sign(d)
    for i in range(1, len(s)):
        if abs(d[i]) <= eps:
            s[i] = s[i - 1]
    return int(np.sum(s[1:] * s[:-1] < 0))


def _fallback_smooth(x, span):
    kernel = np.ones(span, dtype=float) / span
    pad = span // 2
    return np.convolve(np.pad(x, pad, mode="edge"), kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Select a zero-phase, multiscale smoother using a reversal/fit proxy."""
    x = _finite_signal(x)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n < 5:
        return x[window_size - 1:].copy()

    # Project onto a sparse harmonic/chirp family, with robust reweighting.
    t = np.linspace(-1.0, 1.0, n)
    tc = np.arange(n, dtype=float) / max(1, n - 1)
    scale = 1.4826 * np.median(np.abs(x - np.median(x))) + 1e-8
    line = np.polyfit(t, x, 1)
    detrended = x - np.polyval(line, t)
    spectrum = np.abs(np.fft.rfft(detrended))
    freqs = np.fft.rfftfreq(n)
    eligible = np.arange(1, len(spectrum))
    if eligible.size:
        order = eligible[np.argsort(spectrum[eligible])[::-1]]
        chosen = []
        for k in order:
            if all(abs(k - q) >= 2 for q in chosen):
                chosen.append(int(k))
            if len(chosen) == 12:
                break
    else:
        chosen = []

    def _robust_fit(A, y):
        """Fit a finite design matrix using three Huber IRLS passes."""
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        for _ in range(3):
            residual = y - A @ coef
            s = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-8
            weights = np.minimum(1.0, 1.5 * s / (np.abs(residual) + 1e-12))
            coef = np.linalg.lstsq(A * weights[:, None], y * weights, rcond=None)[0]
        return A @ coef

    def _quality(z):
        """Rank reconstructions by residual error, reversals, and whiteness."""
        residual = x - z
        rscale = np.std(x) + 1e-8
        ac = np.corrcoef(residual[:-1], residual[1:])[0, 1] if n > 2 else 0.0
        return (np.mean(np.abs(residual)) / rscale +
                0.018 * _reversals(z) + 0.08 * abs(np.nan_to_num(ac)))

    candidates = []
    if chosen:
        harmonic = 2.0 * np.pi * freqs[chosen]
        A = np.column_stack([np.ones(n), t] +
                            [f(t * 0 + harmonic[j] * tc) for j in range(len(chosen))
                             for f in (np.sin, np.cos)])
        candidates.append(_robust_fit(A, x))

        lo, hi = freqs[chosen].min(), freqs[chosen].max()
        initials = np.linspace(lo, hi, 12)
        chirps = np.linspace(-0.25, 0.25, 8)
        atoms = []
        for f0 in initials:
            for rate in chirps:
                phase = 2 * np.pi * (f0 * tc + 0.5 * rate * tc * tc)
                atom = np.sin(phase)
                atoms.append((abs(np.dot(atom, detrended)), atom))
        atoms.sort(key=lambda q: q[0], reverse=True)
        C = np.column_stack([np.ones(n), t] +
                            [a for _, a in atoms[:12]])
        candidates.append(_robust_fit(C, x))

    spans = [_odd_span(window_size, n), _odd_span(2 * window_size, n)]
    if _HAVE_SCIPY:
        candidates.extend(savgol_filter(x, s, 2 if s < 7 else 3, mode="interp")
                         for s in spans)
    else:
        candidates.extend(_fallback_smooth(x, s) for s in spans)

    xstd = np.std(x)
    valid = []
    for candidate in candidates:
        candidate = np.asarray(candidate, dtype=float)
        corr = (np.corrcoef(candidate, x)[0, 1]
                if xstd > 1e-12 and np.std(candidate) > 1e-12 else 1.0)
        if np.isfinite(corr) and corr >= 0.35:
            valid.append((candidate, _quality(candidate)))
    best = min(valid, key=lambda q: q[1])[0] if valid else x.copy()

    """Apply bounded ADMM second-order trend filtering to the selected suffix."""
    raw = np.asarray(best[window_size - 1:], dtype=float).copy()
    if raw.size < 3:
        return raw

    def _soft_threshold(v, threshold):
        """Apply elementwise soft thresholding to the second-difference split."""
        return np.sign(v) * np.maximum(np.abs(v) - threshold, 0.0)

    m = raw.size
    second = np.zeros((max(0, m - 2), m), dtype=float)
    if m >= 3:
        rows = np.arange(m - 2)
        second[rows, rows] = 1.0
        second[rows, rows + 1] = -2.0
        second[rows, rows + 2] = 1.0

    differences = np.diff(raw)
    sigma = (1.4826 * np.median(
        np.abs(differences - np.median(differences))
    ) / np.sqrt(2.0))
    sigma = max(float(sigma), 0.02 * float(np.std(raw)) + 1e-8)
    weights = np.ones(m, dtype=float)
    weights[[0, -1]] = 25.0
    rho = 1.0
    normal = np.diag(weights) + rho * (second.T @ second)
    rhs_data = weights * raw

    try:
        factor = np.linalg.cholesky(normal)
        def _solve(rhs):
            """Solve the fixed positive-definite ADMM normal system."""
            y = np.linalg.solve(factor, rhs)
            return np.linalg.solve(factor.T, y)

        z = raw.copy()
        d = second @ z
        dual = np.zeros_like(d)
        candidates = []
        for multiplier in (0.5, 1.0, 2.0, 4.0, 8.0):
            lam = multiplier * sigma * np.sqrt(float(m))
            for _ in range(35):
                z = _solve(rhs_data + rho * second.T @ (d - dual))
                proposed = second @ z + dual
                new_d = _soft_threshold(proposed, lam / rho)
                dual += second @ z - new_d
                d = new_d
            z = np.asarray(z, dtype=float)
            z[0] = raw[0]
            z[-1] = raw[-1]
            corr = (np.corrcoef(z, raw)[0, 1]
                    if np.std(z) > 1e-12 and np.std(raw) > 1e-12 else 1.0)
            if np.isfinite(corr) and corr >= 0.35:
                fidelity = np.mean(np.abs(z - raw)) / (np.std(raw) + 1e-8)
                score = (0.018 * _reversals(z) + 0.08 * fidelity -
                         0.04 * corr)
                candidates.append((score, z.copy()))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass
    return raw


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