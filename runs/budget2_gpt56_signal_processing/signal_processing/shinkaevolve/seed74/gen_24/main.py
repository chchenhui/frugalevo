# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving-average baseline.

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
        raise ValueError("window_size must be at least 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust low-lag adaptive trend filter.

    Every output is aligned to the newest sample in its causal window. A robust
    exponentially weighted local-linear fit produces an endpoint observation.
    An adaptive alpha-beta tracker then combines prediction and observation,
    clipping isolated innovations and requiring confirmation for weak direction
    reversals. This reduces noise-induced slope changes while preserving
    sustained non-stationary trends.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    output_length = len(x) - window_size + 1
    y = np.empty(output_length, dtype=float)

    # Endpoint-centered time basis: zero represents the newest window sample.
    time = np.arange(1 - window_size, 1, dtype=float)

    # Moderately recent weighting retains responsiveness without making the
    # endpoint estimate excessively sensitive to a single latest sample.
    base_weights = np.exp(time / max(0.42 * window_size, 1.0))
    base_weights /= base_weights.sum()

    base_time_mean = np.dot(base_weights, time)
    base_centered_time = time - base_time_mean
    base_time_variance = np.dot(base_weights, base_centered_time * base_centered_time)
    eps = 1e-12

    level = 0.0
    velocity = 0.0
    pending_sign = 0
    pending_count = 0

    for i in range(output_length):
        window = x[i : i + window_size]

        # Fast initial weighted local-linear endpoint estimate.
        value_mean = np.dot(base_weights, window)
        slope = (
            np.dot(base_weights, base_centered_time * (window - value_mean))
            / max(base_time_variance, eps)
        )
        observation = value_mean - slope * base_time_mean

        # One robust Huber pass prevents impulses from creating artificial
        # endpoint movement or false directional reversals.
        residuals = window - (observation + slope * time)
        median_residual = np.median(residuals)
        scale = 1.4826 * np.median(np.abs(residuals - median_residual)) + 1e-8

        huber_limit = 2.35 * scale
        huber = np.minimum(1.0, huber_limit / (np.abs(residuals) + eps))

        # The newest point has the greatest leverage in an endpoint fit. When
        # both the local fit and the persistent tracker agree on direction,
        # discount only recent residuals that contradict that direction more
        # aggressively. A sustained real turn soon changes the fitted slope,
        # so it is not permanently treated as an endpoint impulse.
        trend_floor = 0.08 * scale
        if (
            i > 0
            and abs(slope) > trend_floor
            and abs(velocity) > trend_floor
            and slope * velocity > 0.0
        ):
            trend_sign = 1.0 if velocity > 0.0 else -1.0
            recent_count = min(3, window_size)
            contradictory = (
                residuals[-recent_count:] * trend_sign < -0.35 * scale
            )
            if np.any(contradictory):
                recent_huber_limit = 1.65 * scale
                recent_residuals = np.abs(residuals[-recent_count:])
                huber[-recent_count:] = np.where(
                    contradictory,
                    np.minimum(
                        huber[-recent_count:],
                        recent_huber_limit / (recent_residuals + eps),
                    ),
                    huber[-recent_count:],
                )

        weights = base_weights * huber
        weight_sum = weights.sum()

        if weight_sum > eps:
            weights /= weight_sum
            time_mean = np.dot(weights, time)
            centered_time = time - time_mean
            value_mean = np.dot(weights, window)
            denominator = np.dot(weights, centered_time * centered_time)

            if denominator > eps:
                slope = np.dot(weights, centered_time * (window - value_mean)) / denominator
                observation = value_mean - slope * time_mean

        if i == 0:
            level = observation
            velocity = slope
            y[i] = level
            continue

        prediction = level + velocity
        innovation = observation - prediction

        # Bound individual measurement shocks while allowing larger movements
        # when supported by the fitted local slope.
        innovation_limit = 2.8 * scale + 1.35 * abs(slope)
        clipped_innovation = np.clip(innovation, -innovation_limit, innovation_limit)

        # Adaptive alpha gain: quiet regions are smoothed aggressively, whereas
        # coherent local motion receives a higher tracking gain.
        motion = abs(slope) + 0.30 * abs(clipped_innovation)
        alpha = motion / (motion + 1.75 * scale + eps)
        alpha = np.clip(alpha, 0.09, 0.64)

        level = prediction + alpha * clipped_innovation

        # Blend long-lived velocity with robust local slope. The beta component
        # corrects accumulated prediction error with less noise than direct
        # differentiation of the output.
        candidate_velocity = (
            0.78 * velocity
            + 0.22 * slope
            + 0.12 * alpha * clipped_innovation
        )

        # Weak sign changes need consecutive confirmation. Strong reversals,
        # indicated by a slope beyond the noise deadband, are accepted directly.
        deadband = 0.16 * scale + 0.025 * abs(velocity)
        old_sign = 1 if velocity > deadband else (-1 if velocity < -deadband else 0)
        new_sign = (
            1 if candidate_velocity > deadband
            else (-1 if candidate_velocity < -deadband else 0)
        )

        if old_sign != 0 and new_sign != 0 and old_sign != new_sign:
            strong_reversal = abs(candidate_velocity) > (0.42 * scale + 0.55 * abs(velocity))

            if strong_reversal:
                pending_sign = 0
                pending_count = 0
            elif pending_sign == new_sign:
                pending_count += 1
            else:
                pending_sign = new_sign
                pending_count = 1

            if not strong_reversal and pending_count < 2:
                candidate_velocity = 0.55 * velocity
            else:
                pending_sign = 0
                pending_count = 0
        else:
            pending_sign = 0
            pending_count = 0

        # Remove sub-noise velocity jitter, which otherwise appears as frequent
        # one-sample slope reversals in volatile signals.
        if abs(candidate_velocity) < 0.055 * scale:
            candidate_velocity = 0.0

        velocity = candidate_velocity
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