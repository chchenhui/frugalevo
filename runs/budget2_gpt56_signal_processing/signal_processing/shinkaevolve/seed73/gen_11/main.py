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
    Causal robust local-trend filter with adaptive reversal suppression.

    Each output is aligned to the newest sample of its window.  A robust weighted
    local-linear fit supplies a low-lag measurement, then an alpha-beta tracker
    smooths measurement noise while retaining persistent trend velocity.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    if window_size < 3:
        raise ValueError("window_size must be at least 3")

    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)

    # Robustly clip isolated outliers before fitting.  MAD adapts the clipping
    # range to local volatility rather than assuming stationary noise.
    medians = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - medians[:, None]), axis=1)
    local_noise = np.maximum(1.4826 * mad, np.finfo(float).eps)
    clipped = np.clip(
        windows,
        (medians - 3.5 * local_noise)[:, None],
        (medians + 3.5 * local_noise)[:, None],
    )

    # A weighted line evaluated at t=0 (the newest sample) has substantially
    # less phase delay than an average, while exponential weighting limits the
    # effect of old non-stationary observations.
    t = np.arange(window_size, dtype=float) - (window_size - 1)
    weights = np.exp(np.linspace(-3.0, 0.0, window_size))
    design = np.column_stack((np.ones(window_size), t))
    fit_matrix = np.linalg.solve(
        design.T @ (weights[:, None] * design),
        design.T * weights,
    )
    measurements = clipped @ fit_matrix[0]
    local_slopes = clipped @ fit_matrix[1]

    # Window amplitude is not measurement noise for an oscillating signal.
    # Estimate noise from deviations around the fitted local trend instead.
    coefficients = clipped @ fit_matrix.T
    fitted_windows = coefficients[:, :1] + coefficients[:, 1:] * t[None, :]
    residual_mad = np.median(np.abs(clipped - fitted_windows), axis=1)
    tracker_noise = np.maximum(1.4826 * residual_mad, 0.08 * local_noise)

    y = np.empty(len(measurements))
    level = measurements[0]
    velocity = local_slopes[0]
    pending_direction = 0
    pending_count = 0
    y[0] = level

    for i in range(1, len(measurements)):
        prediction = level + velocity
        noise_scale = tracker_noise[i]

        # Bound isolated innovations while allowing the bound to expand in
        # volatile regions.  This is a lightweight robust Kalman-style update.
        innovation = measurements[i] - prediction
        innovation = np.clip(innovation, -4.0 * noise_scale, 4.0 * noise_scale)

        slope_evidence = abs(local_slopes[i]) / (abs(local_slopes[i]) + noise_scale)
        alpha = 0.38 + 0.34 * slope_evidence
        candidate = prediction + alpha * innovation
        step = candidate - level

        # A Schmitt-style confirmation gate rejects one-sample direction flips.
        # Strong fitted-slope evidence bypasses confirmation at real turns.
        reversal_band = 0.22 * noise_scale + 0.10 * abs(velocity)
        opposite = step * velocity < 0.0
        strong_turn = abs(local_slopes[i]) > 0.55 * noise_scale and abs(step) > reversal_band

        if opposite and not strong_turn:
            direction = 1 if step > 0.0 else -1
            if direction == pending_direction:
                pending_count += 1
            else:
                pending_direction = direction
                pending_count = 1

            if pending_count < 2 or abs(step) < reversal_band:
                candidate = level
                step = 0.0
            else:
                pending_count = 0
        else:
            pending_direction = 0
            pending_count = 0

        beta = 0.18 + 0.16 * slope_evidence
        velocity = (1.0 - beta) * velocity + beta * step
        level = candidate
        y[i] = level

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