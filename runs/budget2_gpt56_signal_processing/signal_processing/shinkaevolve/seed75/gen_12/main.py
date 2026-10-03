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
    Causal adaptive level-and-trend Kalman filter.

    Recent first differences provide a robust local noise estimate.  The state
    includes a slope term, reducing moving-average phase lag, while clipped
    innovations suppress isolated spikes that would otherwise create false
    directional reversals.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Estimate a stable initial measurement variance from the available
    # startup window.  MAD of first differences is robust to impulses.
    startup_diff = np.diff(x[:window_size])
    startup_mad = np.median(np.abs(startup_diff - np.median(startup_diff)))
    measurement_var = max((startup_mad / (0.6745 * np.sqrt(2.0))) ** 2, 1e-8)

    level = x[0]
    slope = 0.0
    p00 = measurement_var
    p01 = 0.0
    p11 = measurement_var

    # The state estimate may still make small noise-scale reversals.  Keep a
    # separate causal emitted value with direction hysteresis so that output
    # slope changes represent persistent trends rather than innovations.
    emitted_level = None
    emitted_direction = 0
    pending_direction = 0
    reversal_count = 0

    for t in range(len(x)):
        start = max(0, t - window_size + 1)
        recent_diff = np.diff(x[start : t + 1])
        if len(recent_diff):
            diff_mad = np.median(np.abs(recent_diff - np.median(recent_diff)))
            local_r = max((diff_mad / (0.6745 * np.sqrt(2.0))) ** 2, 1e-8)
            # Avoid abrupt gain changes caused by a single unusual window.
            measurement_var = 0.9 * measurement_var + 0.1 * local_r

        predicted_level = level + slope
        predicted_slope = slope

        raw_innovation = x[t] - predicted_level
        innovation_limit = 3.0 * np.sqrt(measurement_var + p00)
        innovation = np.clip(raw_innovation, -innovation_limit, innovation_limit)

        # Small baseline acceleration noise keeps the estimate smooth.  A
        # bounded innovation term lets genuine changes increase responsiveness.
        process_var = min(
            0.002 * measurement_var + 0.01 * innovation * innovation,
            0.25 * measurement_var,
        )

        predicted_p00 = p00 + 2.0 * p01 + p11 + 0.25 * process_var
        predicted_p01 = p01 + p11 + 0.5 * process_var
        predicted_p11 = p11 + process_var

        residual_var = predicted_p00 + measurement_var
        gain_level = predicted_p00 / residual_var
        gain_slope = predicted_p01 / residual_var

        level = predicted_level + gain_level * innovation
        slope = predicted_slope + gain_slope * innovation

        p00 = (1.0 - gain_level) * predicted_p00
        p01 = (1.0 - gain_level) * predicted_p01
        p11 = predicted_p11 - gain_slope * predicted_p01

        if t >= window_size - 1:
            if emitted_level is None:
                emitted_level = level
            else:
                delta = level - emitted_level
                # This is deliberately below a typical genuine local move,
                # but above residual one-sample measurement jitter.
                deadband = 0.12 * np.sqrt(measurement_var)
                direction = 1 if delta > deadband else (-1 if delta < -deadband else 0)

                if direction == 0:
                    # Retain the previous output through insignificant moves.
                    pending_direction = 0
                    reversal_count = 0
                elif emitted_direction == 0 or direction == emitted_direction:
                    # Do not delay an already established trend.
                    emitted_level = level
                    emitted_direction = direction
                    pending_direction = 0
                    reversal_count = 0
                else:
                    # A reversal must persist for two noise-significant
                    # estimates; this rejects isolated false turning points.
                    if direction == pending_direction:
                        reversal_count += 1
                    else:
                        pending_direction = direction
                        reversal_count = 1

                    if reversal_count >= 2:
                        emitted_level = level
                        emitted_direction = direction
                        pending_direction = 0
                        reversal_count = 0

            y[t - window_size + 1] = emitted_level

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