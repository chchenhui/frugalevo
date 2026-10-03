# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Trend-adaptive quadratic filter with near-zero lag.

    Fits a degree-2 polynomial to each sliding window (single vectorized
    lstsq) and evaluates it at t=0.7 — between the window center (smooth but
    laggy) and the leading edge (zero lag but noise-amplifying) — giving a
    low-lag, low-noise estimate. The fit is blended with the most recent raw
    sample based on scale-invariant local trend strength: strong trends pass
    through raw signal (responsiveness), weak trends use the smooth fit
    (noise suppression, fewer false reversals). A final 3-point smoothing on
    the interior removes residual spurious reversals while preserving
    endpoints, adding negligible lag.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    output_length = n - window_size + 1

    # Hampel-style despike: replace samples deviating strongly from a
    # rolling median with the median. Outliers are the dominant cause of
    # false reversals in the polynomial fit, so removing them first
    # improves both smoothness and tracking accuracy.
    k = 3
    pad = np.pad(x, k, mode="edge")
    med = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(n)])
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

    # Evaluate fit at t = 0.6 (lag/noise compromise between center and edge)
    te = 0.6
    fit = coeffs[0] + coeffs[1] * te + coeffs[2] * te * te
    slope = coeffs[1] + 2.0 * coeffs[2] * te

    # Scale-invariant trend strength: |slope| relative to local variability
    local_std = np.std(windows, axis=1)
    trend_strength = np.abs(slope) / (local_std + 1e-9)
    trend_strength = np.clip(trend_strength, 0.0, 1.0)

    # Blend: strong trend -> raw recent sample; weak trend -> smooth fit.
    # Squaring trend_strength makes the blend more conservative, passing
    # raw signal through only for genuinely strong trends (fewer false
    # reversals) while keeping responsiveness on real moves.
    raw_recent = windows[:, -1]
    w = trend_strength * trend_strength
    y = w * raw_recent + (1.0 - w) * fit

    # 5-point binomial smoothing on interior only (endpoints preserved).
    # Binomial kernels preserve linear trends better than box filters at
    # the same cutoff, cutting residual slope reversals with minimal lag.
    if output_length > 6:
        k5 = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        y[2:-2] = (
            k5[0] * y[:-4] + k5[1] * y[1:-3] + k5[2] * y[2:-2]
            + k5[3] * y[3:-1] + k5[4] * y[4:]
        )
        # Smooth the two edge-adjacent points with the 3-point kernel
        y[1] = (y[0] + 2.0 * y[1] + y[2]) / 4.0
        y[-2] = (y[-3] + 2.0 * y[-2] + y[-1]) / 4.0
    elif output_length > 4:
        y[1:-1] = (y[:-2] + 2.0 * y[1:-1] + y[2:]) / 4.0

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic" or "enhanced")

    Returns:
        Filtered signal
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    else:
        return adaptive_filter(input_signal, window_size)


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
