# EVOLVE-BLOCK-START
"""
Trend-adaptive quadratic filter with low lag for non-stationary time series.

Pipeline: (1) vectorized Hampel despike via scipy median_filter (removes
outliers, the dominant cause of false reversals), (2) vectorized quadratic
fit per sliding window evaluated at t=0.6 (lag/noise compromise), (3)
scale-invariant trend-strength blend gating raw passthrough for genuine
trends, (4) binomial interior smoothing, (5) iterative slope-sign
hysteresis damping in weak-trend regions to collapse noise-induced
directional reversals while leaving genuine trends untouched.
"""
import numpy as np
from scipy.ndimage import median_filter


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Trend-adaptive quadratic filter with low lag and noise suppression.

    Despikes outliers (Hampel), fits quadratics over sliding windows
    evaluated at t=0.6, blends with raw signal via squared trend strength,
    and applies binomial smoothing on the interior. Fully vectorized.
    """
    x = np.asarray(input_signal, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    output_length = n - window_size + 1

    # Hampel-style despike (vectorized via scipy): replace samples
    # deviating > 3 MAD from the rolling median with the median itself.
    k = 3
    med = median_filter(x, size=2 * k + 1, mode="nearest")
    mad = np.median(np.abs(x - med)) + 1e-9
    x = np.where(np.abs(x - med) > 3.0 * mad, med, x)

    # Sliding window matrix: rows = windows, columns = samples within window
    idx = np.arange(window_size)
    starts = np.arange(output_length)
    windows = x[starts[:, None] + idx[None, :]]  # shape (m, W)

    # Fit quadratic on t in [-1, 1] for all windows at once
    t = np.linspace(-1.0, 1.0, window_size)
    V = np.vander(t, 3, increasing=True)  # columns: 1, t, t^2
    coeffs = np.linalg.lstsq(V, windows.T, rcond=None)[0]  # (3, m)

    # Evaluate fit at t = 0.6 (lag/noise compromise between center and
    # edge; te=0.5 measurably increased lag_error without noise benefit)
    te = 0.6
    fit = coeffs[0] + coeffs[1] * te + coeffs[2] * te * te
    slope = coeffs[1] + 2.0 * coeffs[2] * te

    # Scale-invariant trend strength: |slope| relative to local variability
    local_std = np.std(windows, axis=1)
    trend_strength = np.clip(np.abs(slope) / (local_std + 1e-9), 0.0, 1.0)

    # Blend: strong trend -> despiked recent sample; weak trend -> smooth
    # fit. 4th power gating is very conservative: raw signal passes only
    # for genuinely strong trends, sharply cutting false reversals.
    recent = windows[:, -1]
    w = trend_strength ** 4
    y = w * recent + (1.0 - w) * fit

    # 5-point binomial smoothing on interior only (endpoints preserved).
    # Binomial kernels preserve linear trends better than box filters.
    if output_length > 6:
        k5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        y[2:-2] = (
            k5[0] * y[:-4] + k5[1] * y[1:-3] + k5[2] * y[2:-2]
            + k5[3] * y[3:-1] + k5[4] * y[4:]
        )
        y[1] = (y[0] + 2.0 * y[1] + y[2]) / 4.0
        y[-2] = (y[-3] + 2.0 * y[-2] + y[-1]) / 4.0

        # Trend-adaptive second pass: smooth weak-trend regions more
        # aggressively, leave strong-trend regions nearly untouched. This
        # targets false reversals (which occur where trend is weak/flat)
        # without adding lag during genuine directional moves.
        w2 = trend_strength[2:-2] ** 2  # 0 = full smooth, 1 = keep as-is
        smoothed = (
            k5[0] * y[:-4] + k5[1] * y[1:-3] + k5[2] * y[2:-2]
            + k5[3] * y[3:-1] + k5[4] * y[4:]
        )
        y[2:-2] = w2 * y[2:-2] + (1.0 - w2) * smoothed

        # Iterative reversal suppression with hysteresis: repeatedly damp
        # sign-flip points in weak-trend regions toward their neighbors.
        # Iterating collapses chains of noise flips that a single pass
        # merely shifts by one sample; strong trends are never touched.
        for _ in range(3):
            d = np.diff(y)
            flips = (d[:-1] * d[1:]) < 0
            weak = trend_strength[1:-1] < 0.35
            targets = np.where(flips & weak)[0] + 1
            if len(targets) == 0:
                break
            y[targets] = (
                0.2 * y[targets - 1] + 0.6 * y[targets] + 0.2 * y[targets + 1]
            )
    elif output_length > 4:
        y[1:-1] = (y[:-2] + 2.0 * y[1:-1] + y[2:]) / 4.0

    return y


# EVOLVE-BLOCK-END


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    """
    Generate synthetic test signal with known characteristics.

    Args:
        length: Length of the signal
        noise_level: Standard deviation of noise to add
        seed: Random seed for reproducibility

    Returns:
        Tuple of (noisy_signal, clean_signal)
    """
    np.random.seed(seed)
    t = np.linspace(0, 10, length)

    # Create a complex signal with multiple components
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)  # Low frequency component
        + 1.5 * np.sin(2 * np.pi * 2 * t)  # Medium frequency component
        + 0.5 * np.sin(2 * np.pi * 5 * t)  # Higher frequency component
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)  # Decaying oscillation
    )

    # Add non-stationary behavior
    trend = 0.1 * t * np.sin(0.2 * t)  # Slowly varying trend
    clean_signal += trend

    # Add random walk component for non-stationarity
    random_walk = np.cumsum(np.random.randn(length) * 0.05)
    clean_signal += random_walk

    # Add noise
    noise = np.random.normal(0, noise_level, length)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000, noise_level=0.3, window_size=20):
    """
    Run the signal processing algorithm on a test signal.

    Args:
        noisy_signal: Input signal to filter (if provided, use this; otherwise generate)
        signal_length: Length if generating signal (for backward compatibility)
        noise_level: Noise level if generating signal (for backward compatibility)
        window_size: Window size for processing

    Returns:
        Dictionary containing results and metrics
    """
    # Use provided signal or generate test signal (for backward compatibility)
    if noisy_signal is not None:
        # Filter the provided signal
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")
        clean_signal = None  # Not available when using provided signal
    else:
        # Generate test signal (for __main__ and backward compatibility)
        noisy_signal, clean_signal = generate_test_signal(signal_length, noise_level)
        filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    # Calculate basic metrics (only if we have clean_signal from generation)
    if len(filtered_signal) > 0 and clean_signal is not None:
        # Align signals for comparison (account for processing delay)
        delay = window_size - 1
        aligned_clean = clean_signal[delay:]
        aligned_noisy = noisy_signal[delay:]

        # Ensure same length
        min_length = min(len(filtered_signal), len(aligned_clean))
        filtered_signal = filtered_signal[:min_length]
        aligned_clean = aligned_clean[:min_length]
        aligned_noisy = aligned_noisy[:min_length]

        # Calculate correlation with clean signal
        correlation = np.corrcoef(filtered_signal, aligned_clean)[0, 1] if min_length > 1 else 0

        # Calculate noise reduction
        noise_before = np.var(aligned_noisy - aligned_clean)
        noise_after = np.var(filtered_signal - aligned_clean)
        noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0

        return {
            "filtered_signal": filtered_signal,
            "clean_signal": aligned_clean,
            "noisy_signal": aligned_noisy,
            "correlation": correlation,
            "noise_reduction": noise_reduction,
            "signal_length": min_length,
        }
    elif len(filtered_signal) > 0:
        # When using provided signal (no clean_signal available), just return filtered signal
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }
    else:
        return {
            "filtered_signal": [],
            "clean_signal": [],
            "noisy_signal": [],
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": 0,
        }


if __name__ == "__main__":
    # Test the algorithm
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")
