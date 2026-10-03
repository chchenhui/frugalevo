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
    Causal robust local-trend Kalman filter.

    The state contains signal level and slope.  Unlike a trailing moving
    average, the slope prediction compensates for most of the smoothing lag.
    Measurement uncertainty is learned from innovations and large innovations
    are clipped before updating, preventing isolated noise spikes from
    creating false trend reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Number of initial samples used for noise calibration

    Returns:
        y: Filtered output aligned with x[window_size - 1:]
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite values")

    output_length = len(x) - window_size + 1
    y = np.empty(output_length)

    # Robustly estimate observation noise from first differences.  Differencing
    # avoids interpreting a slowly changing signal level as measurement noise.
    initial_differences = np.diff(x[:window_size])
    diff_mad = np.median(np.abs(initial_differences - np.median(initial_differences)))
    measurement_variance = max((diff_mad / 0.6745) ** 2 * 0.5, 1e-8)

    # Constant-velocity state: [level, slope].  A modest acceleration variance
    # smooths short fluctuations while allowing real turns to be tracked.
    level = x[0]
    slope = 0.0
    covariance = np.array([[measurement_variance, 0.0],
                           [0.0, measurement_variance]], dtype=float)
    process_variance = max(measurement_variance * 0.035, 1e-9)
    process_noise = process_variance * np.array([[0.25, 0.5],
                                                  [0.5, 1.0]])

    output_index = 0
    for i, observation in enumerate(x):
        if i > 0:
            level += slope
            covariance = covariance + np.array(
                [[covariance[1, 0] + covariance[0, 1] + covariance[1, 1], covariance[1, 1]],
                 [covariance[1, 1], covariance[1, 1]]]
            ) + process_noise

        innovation = observation - level
        innovation_limit = 3.0 * np.sqrt(measurement_variance + covariance[0, 0])
        clipped_innovation = np.clip(innovation, -innovation_limit, innovation_limit)

        innovation_variance = covariance[0, 0] + measurement_variance
        gain = covariance[:, 0] / innovation_variance
        level += gain[0] * clipped_innovation
        slope += gain[1] * clipped_innovation
        covariance -= np.outer(gain, covariance[0, :])
        covariance = 0.5 * (covariance + covariance.T)

        # Slow adaptation is stable enough for non-stationary noise but does
        # not let one genuine sharp transition inflate the noise estimate.
        bounded_residual = min(innovation * innovation, innovation_limit * innovation_limit)
        measurement_variance = max(
            0.97 * measurement_variance + 0.03 * bounded_residual,
            1e-8,
        )

        if i >= window_size - 1:
            y[output_index] = level
            output_index += 1

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