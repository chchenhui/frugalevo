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
    Causal robust alpha-beta tracker with adaptive noise rejection.

    The state contains a signal level and its per-sample slope.  Predicting the
    level before incorporating a measurement substantially reduces the lag of a
    moving average.  A rolling MAD estimate robustly limits impulsive innovations,
    while a slope deadband prevents small noise fluctuations from repeatedly
    reversing the estimated trend.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the rolling noise-estimation window

    Returns:
        Filtered samples aligned with input samples window_size - 1 onward.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    level = x[0]
    slope = 0.0
    pending_direction = 0.0
    pending_reversal_count = 0
    estimates = np.empty(len(x), dtype=float)
    innovations = np.empty(len(x), dtype=float)
    estimates[0] = level
    innovations[0] = 0.0

    for i in range(1, len(x)):
        predicted_level = level + slope
        innovation = x[i] - predicted_level

        start = max(0, i - window_size)
        recent_innovations = innovations[start:i]
        median_innovation = np.median(recent_innovations)
        mad = np.median(np.abs(recent_innovations - median_innovation))

        # Innovation MAD is initially zero because the tracker has not yet
        # accumulated residuals.  Use a causal difference-based bootstrap during
        # acquisition so that the first genuine movement is not clipped away.
        if i < 4:
            recent_differences = np.diff(x[max(0, i - window_size) : i + 1])
            difference_median = np.median(recent_differences)
            difference_mad = np.median(np.abs(recent_differences - difference_median))
            bootstrap_scale = 1.4826 * difference_mad / np.sqrt(2.0)
        else:
            bootstrap_scale = 0.0
        noise_scale = max(1.4826 * mad, bootstrap_scale, 1e-8)

        # Robust gating rejects isolated spikes without suppressing a real move.
        clipped_innovation = np.clip(innovation, -3.0 * noise_scale, 3.0 * noise_scale)
        significance = abs(innovation) / noise_scale
        alpha = 0.24 + 0.28 * min(1.0, max(0.0, (significance - 1.0) / 2.0))
        beta = 0.035 + 0.070 * min(1.0, max(0.0, (significance - 1.0) / 2.0))

        level = predicted_level + alpha * clipped_innovation
        proposed_slope = slope + beta * clipped_innovation

        # A directional change must be both significant and persistent.  Level
        # correction remains immediate, but retaining the old predictive slope
        # for one counter-trend sample prevents noise from creating a reversal.
        slope_deadband = 0.14 * noise_scale
        old_direction = np.sign(slope)
        candidate_direction = np.sign(proposed_slope)
        reverses_trend = (
            old_direction != 0.0
            and candidate_direction != 0.0
            and candidate_direction != old_direction
            and abs(proposed_slope) >= slope_deadband
            and np.sign(clipped_innovation) == candidate_direction
        )

        if reverses_trend:
            if candidate_direction == pending_direction:
                pending_reversal_count += 1
            else:
                pending_direction = candidate_direction
                pending_reversal_count = 1

            if pending_reversal_count >= 2:
                # Confirmed turns receive a relatively fast slope correction.
                slope = 0.55 * slope + 0.45 * proposed_slope
                pending_direction = 0.0
                pending_reversal_count = 0
            else:
                slope *= 0.985
        else:
            pending_direction = 0.0
            pending_reversal_count = 0
            if abs(proposed_slope) < slope_deadband:
                proposed_slope = 0.0
            slope = 0.78 * slope + 0.22 * proposed_slope

        estimates[i] = level
        innovations[i] = innovation

    return estimates[window_size - 1:]


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