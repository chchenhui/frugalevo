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
    Causal robust alpha-beta tracker with endpoint-aligned local trend estimation.

    The output remains aligned to the newest sample in each window.  A weighted
    local linear fit reduces average-filter lag, while adaptive gains and a
    reversal gate suppress noise-driven changes in direction.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    if window_size < 3:
        raise ValueError("window_size must be at least 3")

    # The latest sample is at t=0.  A lightly regularized quadratic fit retains
    # local curvature at real turning points while still being fully causal.
    time = np.arange(window_size, dtype=float) - (window_size - 1)
    weights = np.exp(np.linspace(-2.7, 0.0, window_size))
    weights /= np.sum(weights)
    design = np.column_stack((np.ones(window_size), time, time * time))
    normal_matrix = design.T @ (weights[:, None] * design)
    normal_matrix += np.diag((1e-10, 1e-10, 2e-4))
    fit_operator = np.linalg.solve(normal_matrix, design.T * weights)

    position = 0.0
    velocity = 0.0
    confirmed_direction = 0.0
    pending_direction = 0.0
    pending_count = 0

    for i in range(output_length):
        window = x[i : i + window_size]
        # Initial causal fit supplies a trend reference for robust endpoint
        # weighting. This protects the high-leverage newest observation from
        # isolated counter-trend impulses without delaying sustained turns.
        coefficients = fit_operator @ window
        residuals = window - design @ coefficients
        residual_center = np.median(residuals)
        noise_scale = max(
            1.4826 * np.median(np.abs(residuals - residual_center)), 1e-8
        )

        residual_magnitude = np.abs(residuals - residual_center)
        robust_limit = np.full(window_size, 2.5 * noise_scale)
        trend_direction = np.sign(coefficients[1])

        # Only tighten the final few samples when both independent trend
        # estimates agree that their residual is counter-trend. A genuine turn
        # soon establishes a new local slope and therefore is not persistently
        # penalized by this endpoint-specific gate.
        if trend_direction != 0.0 and np.sign(velocity) == trend_direction:
            newest = slice(max(0, window_size - 3), window_size)
            counter_trend = (
                np.sign(residuals[newest] - residual_center) == -trend_direction
            )
            robust_limit[newest] = np.where(
                counter_trend, 1.75 * noise_scale, robust_limit[newest]
            )

        robust_weights = weights * np.minimum(
            1.0, robust_limit / (residual_magnitude + 1e-12)
        )
        robust_normal = design.T @ (robust_weights[:, None] * design)
        robust_normal += np.diag((1e-10, 1e-10, 2e-4))
        coefficients = np.linalg.solve(
            robust_normal, design.T @ (robust_weights * window)
        )

        measurement = coefficients[0]
        local_slope = coefficients[1]
        residuals = window - design @ coefficients
        residual_center = np.median(residuals)
        noise_scale = max(
            1.4826 * np.median(np.abs(residuals - residual_center)), 1e-8
        )

        if i == 0:
            position = measurement
            velocity = local_slope
            y[i] = position
            confirmed_direction = np.sign(velocity)
            continue

        prediction = position + velocity
        innovation = measurement - prediction
        # A bounded innovation gives outliers limited influence on both states.
        bounded_innovation = np.clip(innovation, -3.5 * noise_scale, 3.5 * noise_scale)
        activity = min(abs(innovation) / (2.2 * noise_scale), 1.0)
        alpha = 0.16 + 0.48 * activity

        # Blend a level-derived velocity correction with the independently
        # estimated local derivative.  This stabilizes slope without adding
        # a delayed averaging stage.
        velocity_measurement = 0.65 * local_slope + 0.35 * (velocity + 0.10 * bounded_innovation)
        velocity_gain = 0.10 + 0.20 * activity
        candidate_velocity = velocity + velocity_gain * (velocity_measurement - velocity)

        velocity_deadband = 0.12 * noise_scale
        candidate_direction = np.sign(candidate_velocity) if abs(candidate_velocity) > velocity_deadband else 0.0
        significant_turn = abs(innovation) > 1.15 * noise_scale

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

            if pending_count < 2:
                # Preserve the established direction until a turn persists.
                candidate_velocity = 0.82 * velocity
            else:
                confirmed_direction = candidate_direction
                pending_direction = 0.0
                pending_count = 0
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