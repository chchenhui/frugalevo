# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if window_size < 1:
        raise ValueError("window_size must be at least 1")

    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal robust endpoint trend tracker.

    A recency-weighted local linear fit produces an endpoint-aligned measurement
    with substantially less lag than a moving average. A robust alpha-beta tracker
    then smooths that estimate using residual MAD noise scaling, bounded
    innovations, slope deadbanding, and persistent-turn confirmation.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        Filtered output signal aligned to input samples window_size - 1 onward.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")

    output_length = len(x) - window_size + 1
    y = np.empty(output_length, dtype=float)

    # The newest sample is t=0, so the fitted intercept estimates the present
    # value rather than the centre of the window.
    time = np.arange(window_size, dtype=float) - (window_size - 1)
    weights = np.exp(np.linspace(-2.45, 0.0, window_size))
    weights /= np.sum(weights)

    weighted_time = np.sum(weights * time)
    centered_time = time - weighted_time
    time_variance = np.sum(weights * centered_time * centered_time)
    slope_operator = weights * centered_time / max(time_variance, 1e-12)

    position = 0.0
    velocity = 0.0
    confirmed_direction = 0.0
    pending_direction = 0.0
    pending_count = 0

    for i in range(output_length):
        window = x[i:i + window_size]

        weighted_mean = np.dot(weights, window)
        local_slope = np.dot(slope_operator, window)
        measurement = weighted_mean - local_slope * weighted_time

        residuals = window - (measurement + local_slope * time)
        residual_median = np.median(residuals)
        noise_scale = max(
            1.4826 * np.median(np.abs(residuals - residual_median)),
            1e-8,
        )

        if i == 0:
            position = measurement
            velocity = local_slope
            confirmed_direction = np.sign(velocity)
            y[i] = position
            continue

        prediction = position + velocity
        innovation = measurement - prediction
        bounded_innovation = np.clip(
            innovation,
            -3.25 * noise_scale,
            3.25 * noise_scale,
        )

        activity = min(abs(innovation) / (2.35 * noise_scale), 1.0)

        # Conservative normal gains suppress jitter; sustained large changes
        # automatically receive stronger updates to preserve responsiveness.
        alpha = 0.15 + 0.46 * activity
        velocity_gain = 0.075 + 0.185 * activity

        # Blend independently measured local slope with innovation-derived slope.
        velocity_measurement = (
            0.72 * local_slope
            + 0.28 * (velocity + 0.11 * bounded_innovation)
        )
        candidate_velocity = velocity + velocity_gain * (
            velocity_measurement - velocity
        )

        velocity_deadband = 0.10 * noise_scale
        if abs(candidate_velocity) < velocity_deadband:
            candidate_direction = 0.0
        else:
            candidate_direction = np.sign(candidate_velocity)

        significant_turn = (
            abs(innovation) > 1.30 * noise_scale
            and abs(local_slope) > 0.055 * noise_scale
        )

        # Reversal hysteresis: a new slope direction must persist for three
        # meaningful endpoint measurements before replacing an established trend.
        if (
            confirmed_direction != 0.0
            and candidate_direction != 0.0
            and candidate_direction != confirmed_direction
        ):
            if significant_turn and candidate_direction == pending_direction:
                pending_count += 1
            elif significant_turn:
                pending_direction = candidate_direction
                pending_count = 1
            else:
                pending_direction = 0.0
                pending_count = 0

            if pending_count < 3:
                candidate_velocity = 0.88 * velocity
            else:
                confirmed_direction = candidate_direction
                pending_direction = 0.0
                pending_count = 0
                # Confirmed turns are allowed to adapt promptly.
                candidate_velocity = (
                    0.45 * velocity + 0.55 * candidate_velocity
                )
        else:
            pending_direction = 0.0
            pending_count = 0
            if candidate_direction != 0.0:
                confirmed_direction = candidate_direction

        position = prediction + alpha * bounded_innovation
        velocity = candidate_velocity
        y[i] = position

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