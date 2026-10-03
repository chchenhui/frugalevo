"""
Batch zero-phase adaptive signal filtering with robust local smoothing.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def adaptive_filter(x, window_size=20):
    """Baseline trailing-window mean, retained for the public interface."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return np.convolve(x, np.ones(window_size, dtype=float) / window_size,
                       mode="valid")


def _odd_at_most(value, n):
    value = int(min(value, n))
    if value % 2 == 0:
        value -= 1
    return value


def _fallback_smooth(x, width):
    """Dependency-free centered triangular smoother used only without SciPy."""
    n = len(x)
    width = _odd_at_most(width, n)
    if width < 3:
        return x.copy()
    half = width // 2
    weights = np.arange(1, half + 2, dtype=float)
    weights = np.r_[weights, weights[-2::-1]]
    weights /= weights.sum()
    padded = np.pad(x, (half, half), mode="edge")
    return np.convolve(padded, weights, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Reconstruct jump-separated regimes using two-pass reweighted curvature TV.

    The first ADMM pass estimates sparse second-difference knots.  A second
    pass increases the penalty on weak curvature while preserving persistent,
    high-amplitude bends, reducing noise-induced trend reversals.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    n = len(x)
    if not np.all(np.isfinite(x)):
        finite = x[np.isfinite(x)]
        fill = float(np.median(finite)) if finite.size else 0.0
        x = np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)
    else:
        x = x.copy()

    dx = np.diff(x)
    if dx.size:
        median_dx = float(np.median(dx))
        sigma_d = float(np.median(np.abs(dx - median_dx)) / 0.67448975)
        signal_range = float(np.max(x) - np.min(x))
        threshold = max(6.0 * sigma_d, 0.12 * signal_range)
        boundaries = np.flatnonzero(np.abs(dx) > threshold).astype(int) + 1
        boundaries = np.unique(boundaries[(boundaries > 0) & (boundaries < n)])
    else:
        sigma_d = 0.0
        boundaries = np.empty(0, dtype=int)

    rho = 4.0
    lam = 2.2 * sigma_d * np.sqrt(max(window_size, 2))
    tolerance = 1e-4
    y = x.copy()

    for start, stop in zip(np.r_[0, boundaries], np.r_[boundaries, n]):
        segment = np.asarray(x[start:stop], dtype=float)
        length = len(segment)
        if length < 3:
            y[start:stop] = _fallback_smooth(segment, length)
            continue

        # Factor I + rho * D2.T @ D2 as a banded LDL.T system once per regime.
        diagonal = np.ones(length, dtype=float)
        diagonal[:-2] += rho
        diagonal[1:-1] += 4.0 * rho
        diagonal[2:] += rho
        # Lower bands of I + rho * D2.T @ D2.  The first subdiagonal
        # has -2*rho endpoints and -4*rho interior coefficients.
        first_band = np.zeros(length, dtype=float)
        first_band[1] = -2.0 * rho
        if length > 3:
            first_band[2:-1] = -4.0 * rho
        first_band[-1] = -2.0 * rho
        second_band = np.zeros(length, dtype=float)
        second_band[2:] = rho

        lower1 = np.zeros(length, dtype=float)
        lower2 = np.zeros(length, dtype=float)
        pivots = np.empty(length, dtype=float)
        for i in range(length):
            if i >= 2:
                lower2[i] = second_band[i] / pivots[i - 2]
            if i >= 1:
                cross = (
                    lower2[i] * pivots[i - 2] * lower1[i - 1]
                    if i >= 2 else 0.0
                )
                lower1[i] = (first_band[i] - cross) / pivots[i - 1]
            pivots[i] = diagonal[i]
            if i >= 1:
                pivots[i] -= lower1[i] * lower1[i] * pivots[i - 1]
            if i >= 2:
                pivots[i] -= lower2[i] * lower2[i] * pivots[i - 2]

        def solve_curvature_tv(weights, iterations, initial_estimate=None,
                               initial_curvature=None):
            """
            Solve weighted curvature-TV ADMM, optionally warm-started from a
            prior unweighted trend estimate to refine persistent bends.
            """
            curvature = (
                np.asarray(initial_curvature, dtype=float).copy()
                if initial_curvature is not None
                else np.zeros(length - 2, dtype=float)
            )
            dual = np.zeros(length - 2, dtype=float)
            estimate = (
                np.asarray(initial_estimate, dtype=float).copy()
                if initial_estimate is not None
                else np.zeros(length, dtype=float)
            )

            for _ in range(iterations):
                rhs = segment.copy()
                work = curvature - dual
                rhs[:-2] += rho * work
                rhs[1:-1] -= 2.0 * rho * work
                rhs[2:] += rho * work

                forward = np.empty(length, dtype=float)
                for i in range(length):
                    forward[i] = rhs[i]
                    if i >= 1:
                        forward[i] -= lower1[i] * forward[i - 1]
                    if i >= 2:
                        forward[i] -= lower2[i] * forward[i - 2]
                forward /= pivots

                estimate[-1] = forward[-1]
                estimate[-2] = forward[-2] - lower1[-1] * estimate[-1]
                for i in range(length - 3, -1, -1):
                    estimate[i] = (
                        forward[i] - lower1[i + 1] * estimate[i + 1]
                        - lower2[i + 2] * estimate[i + 2]
                    )

                second_difference = (
                    estimate[:-2] - 2.0 * estimate[1:-1] + estimate[2:]
                )
                previous_curvature = curvature.copy()
                relaxed_difference = (
                    1.60 * second_difference - 0.60 * previous_curvature
                )
                shrink_input = relaxed_difference + dual
                curvature = np.sign(shrink_input) * np.maximum(
                    np.abs(shrink_input) - (lam * weights) / rho, 0.0
                )
                residual = second_difference - curvature
                dual += relaxed_difference - curvature

                curvature_change = curvature - previous_curvature
                dual_residual = np.zeros(length, dtype=float)
                dual_residual[:-2] += curvature_change
                dual_residual[1:-1] -= 2.0 * curvature_change
                dual_residual[2:] += curvature_change
                if (np.linalg.norm(residual) <= tolerance * np.sqrt(length)
                        and rho * np.linalg.norm(dual_residual)
                        <= tolerance * np.sqrt(length)):
                    break
            return estimate, curvature

        # First pass establishes which curvature knots persist above noise.
        first_estimate, first_curvature = solve_curvature_tv(
            np.ones(length - 2, dtype=float), 55
        )
        epsilon = max(0.15 * sigma_d, 1e-8)
        curvature_scale = float(np.median(np.abs(first_curvature) + epsilon))
        weights = np.clip(
            curvature_scale / (np.abs(first_curvature) + epsilon), 0.35, 3.0
        )
        # Warm-start the MM-style weighted pass so its fixed iteration budget
        # is spent refining weak bends instead of reconstructing the trend.
        estimate, _ = solve_curvature_tv(
            weights.astype(float, copy=False), 45,
            initial_estimate=first_estimate,
            initial_curvature=first_curvature,
        )
        y[start:stop] = estimate

    # A detected discontinuity must not be redistributed into adjacent ramps.
    for boundary in boundaries:
        y[boundary - 1] = x[boundary - 1]
        y[boundary] = x[boundary]

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y[window_size - 1:]


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


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
    else:
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
    aligned_clean = np.asarray(clean_signal, dtype=float)[delay:]
    aligned_noisy = np.asarray(noisy_signal, dtype=float)[delay:]
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