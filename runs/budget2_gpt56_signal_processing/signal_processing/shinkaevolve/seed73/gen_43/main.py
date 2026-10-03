# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust adaptive Kalman/window consensus filter.

    A constant-velocity Kalman tracker supplies a low-lag level estimate.  A
    causal exponentially weighted local estimate is used as a second, smoother
    consensus signal.  The final alpha-beta stage follows persistent trends
    predictively while damping short alternating deviations that cause false
    directional reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Sliding-window length used for local context and output
            alignment.

    Returns:
        Filtered signal aligned with x[window_size - 1:].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    n = len(x)
    initial = x[:window_size]
    differences = np.diff(initial)

    if differences.size:
        diff_median = np.median(differences)
        diff_scale = np.median(np.abs(differences - diff_median)) / 0.6745
    else:
        diff_scale = 0.0

    measurement_var = max(0.5 * diff_scale * diff_scale, 1e-8)
    innovation_var = measurement_var
    process_var = max(0.05 * measurement_var, 1e-10)

    # Robust initial slope avoids allowing one endpoint outlier to establish a
    # spurious initial direction.
    if window_size > 1:
        half = max(1, window_size // 2)
        initial_slope = (
            np.median(initial[half:] if half < window_size else initial[-1:])
            - np.median(initial[:half])
        ) / max(window_size - half, 1)
    else:
        initial_slope = 0.0

    level = float(x[0])
    velocity = float(initial_slope)
    p00 = measurement_var * 4.0
    p01 = 0.0
    p11 = measurement_var

    # Keep a small causal context for a robust local-linear estimate.  Raw
    # spread is only used to clip outliers; fitted residual spread is used for
    # confidence because a clean ramp naturally has a large raw window MAD.
    history = []
    regression_decay = np.exp(-2.2 / max(window_size - 1, 1))

    output = np.empty(n - window_size + 1, dtype=float)
    consensus_level = float(x[0])
    consensus_velocity = 0.0

    for index, sample in enumerate(x):
        # Constant-velocity Kalman prediction with white acceleration noise.
        predicted_level = level + velocity
        predicted_velocity = velocity
        q = process_var

        pp00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pp01 = p01 + p11 + 0.5 * q
        pp11 = p11 + q

        innovation = sample - predicted_level
        residual_var = max(pp00 + measurement_var, 1e-12)
        innovation_limit = 3.0 * np.sqrt(residual_var)
        bounded_innovation = np.clip(innovation, -innovation_limit, innovation_limit)

        k0 = pp00 / residual_var
        k1 = pp01 / residual_var
        level = predicted_level + k0 * bounded_innovation
        velocity = predicted_velocity + k1 * bounded_innovation

        # Scalar Joseph-equivalent covariance update, kept symmetric.
        p00 = max(pp00 - k0 * pp00, 1e-12)
        p01 = pp01 - k0 * pp01
        p11 = max(pp11 - k1 * pp01, 1e-12)

        innovation_var = 0.95 * innovation_var + 0.05 * bounded_innovation * bounded_innovation
        measurement_var = max(0.985 * measurement_var + 0.015 * innovation_var, 1e-8)
        process_var = max(0.99 * process_var + 0.01 * innovation_var * 0.05, 1e-10)

        # Robust, exponentially weighted local line.  Clipping uses raw MAD
        # before fitting, whereas residual MAD measures noise left after the
        # local trend has been explained.
        history.append(float(sample))
        if len(history) > window_size:
            history.pop(0)
        values = np.asarray(history, dtype=float)
        center = np.median(values)
        raw_mad = 1.4826 * np.median(np.abs(values - center))
        clip_scale = max(raw_mad, np.sqrt(measurement_var), 1e-8)
        clipped_values = np.clip(values, center - 3.0 * clip_scale, center + 3.0 * clip_scale)

        age = np.arange(clipped_values.size, dtype=float) - (clipped_values.size - 1)
        weights = np.exp(age * 2.2 / max(window_size - 1, 1))
        weight_sum = np.sum(weights)
        time_mean = np.sum(weights * age) / weight_sum
        value_mean = np.sum(weights * clipped_values) / weight_sum
        time_centered = age - time_mean
        denominator = np.sum(weights * time_centered * time_centered)
        local_slope = (
            np.sum(weights * time_centered * (clipped_values - value_mean)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        local_level = value_mean + local_slope * ((clipped_values.size - 1) - time_mean)
        fitted_values = value_mean + local_slope * (age - time_mean)
        residuals = clipped_values - fitted_values
        residual_noise = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
        scale = max(residual_noise, 0.12 * np.sqrt(max(innovation_var, 1e-10)), 1e-8)

        # A coherent local slope is reliable even when raw window variation is
        # high.  Conversely, disagreement without slope evidence is treated as
        # noise rather than a reason to create a new direction.
        disagreement = abs(level - local_level)
        slope_evidence = abs(local_slope) / (scale + 1e-12)
        local_weight = (0.18 + 0.20 * min(1.0, slope_evidence / 2.0)) / (
            1.0 + (disagreement / (2.0 * scale + 1e-12)) ** 2
        )
        target = (1.0 - local_weight) * level + local_weight * local_level

        # Predictive post-stage: smoothing is performed around a velocity
        # prediction instead of a stationary average, minimizing phase delay.
        # Update its velocity from both the Kalman trend estimate and the
        # realized consensus step.  Bounding acceleration prevents one noisy
        # residual from creating a sequence of false directional reversals.
        previous_consensus = consensus_level
        predicted_consensus = consensus_level + consensus_velocity
        residual = target - predicted_consensus
        alpha = 0.66 + 0.14 * min(1.0, disagreement / (2.5 * scale + 1e-12)) + 0.10 * min(1.0, slope_evidence / 2.5)
        consensus_level = predicted_consensus + alpha * residual

        step = consensus_level - previous_consensus
        velocity_target = 0.55 * velocity + 0.45 * step
        acceleration_limit = (0.075 + 0.045 * min(1.0, slope_evidence / 2.0)) * scale + 0.025 * abs(velocity)
        velocity_change = np.clip(
            velocity_target - consensus_velocity,
            -acceleration_limit,
            acceleration_limit,
        )
        consensus_velocity += 0.48 * velocity_change

        velocity_deadband = (0.035 - 0.018 * min(1.0, slope_evidence / 2.0)) * scale
        if abs(consensus_velocity) < velocity_deadband:
            consensus_velocity = 0.0

        if index >= window_size - 1:
            output[index - window_size + 1] = consensus_level

    return output


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