# EVOLVE-BLOCK-START
"""
Multi-scale robust real-time filtering for volatile non-stationary signals.

Pipeline:
    1. Robust short- and long-scale causal endpoint regressions
    2. Adaptive blend based on trend agreement and residual noise
    3. Predictive alpha-beta trend tracking
    4. Hysteretic, confirmation-based reversal acceptance

The output is causally aligned with the newest input sample in each trailing
window and has length len(x) - window_size + 1.
"""
import numpy as np


def _validate_signal(x, window_size, minimum_window=2):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < minimum_window:
        raise ValueError(f"window_size must be at least {minimum_window}")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return x


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window.

    Returns:
        Moving-average output of length len(x) - window_size + 1.
    """
    x = _validate_signal(x, window_size, minimum_window=1)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def _robust_endpoint_fit(
    window,
    time,
    base_weights,
    persistent_velocity=0.0,
    apply_asymmetric_weighting=False,
):
    """
    One-pass robust weighted local-linear regression evaluated at time zero.

    Returns:
        endpoint observation, local slope, robust residual scale
    """
    eps = 1e-12

    weight_sum = np.sum(base_weights)
    time_mean = np.dot(base_weights, time) / weight_sum
    centered_time = time - time_mean
    value_mean = np.dot(base_weights, window) / weight_sum
    denominator = np.dot(base_weights, centered_time * centered_time)

    if denominator <= eps:
        return float(window[-1]), 0.0, 1e-8

    slope = np.dot(base_weights, centered_time * (window - value_mean)) / denominator
    observation = value_mean - slope * time_mean

    residuals = window - (observation + slope * time)
    residual_center = np.median(residuals)
    scale = max(1.4826 * np.median(np.abs(residuals - residual_center)), 1e-8)

    # Standard Huber threshold.  It limits broad impulsive contamination
    # without discarding legitimate local structure.
    huber_limit = 2.45 * scale
    robust_weights = np.minimum(1.0, huber_limit / (np.abs(residuals) + eps))

    # Endpoint samples have greatest leverage in a causal regression.  Tighten
    # their threshold only if they oppose both the local and persistent trend.
    trend_floor = 0.08 * scale
    if (
        apply_asymmetric_weighting
        and abs(slope) > trend_floor
        and abs(persistent_velocity) > trend_floor
        and slope * persistent_velocity > 0.0
    ):
        direction = 1.0 if persistent_velocity > 0.0 else -1.0
        recent_count = min(3, len(window))
        recent_residuals = residuals[-recent_count:]
        contradictory = recent_residuals * direction < -0.30 * scale

        if np.any(contradictory):
            tighter_limit = 1.70 * scale
            current = robust_weights[-recent_count:]
            tightened = tighter_limit / (np.abs(recent_residuals) + eps)
            robust_weights[-recent_count:] = np.where(
                contradictory, np.minimum(current, tightened), current
            )

    weights = base_weights * robust_weights
    weight_sum = np.sum(weights)
    if weight_sum <= eps:
        return observation, slope, scale

    weights /= weight_sum
    time_mean = np.dot(weights, time)
    centered_time = time - time_mean
    value_mean = np.dot(weights, window)
    denominator = np.dot(weights, centered_time * centered_time)

    if denominator > eps:
        slope = np.dot(weights, centered_time * (window - value_mean)) / denominator
        observation = value_mean - slope * time_mean

    return float(observation), float(slope), float(scale)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Multi-scale robust low-lag causal trend filter.

    A short regression tracks curvature and sustained transitions. A longer
    regression supplies a stable reference during noisy intervals. Their
    adaptive blend feeds a predictive level/velocity tracker with explicit
    reversal hysteresis.

    Args:
        x: Input signal (1D array of real-valued samples).
        window_size: Trailing robust-statistics window size.

    Returns:
        Filtered samples aligned with input samples window_size - 1 onward.
    """
    x = _validate_signal(x, window_size, minimum_window=2)

    output_length = len(x) - window_size + 1
    y = np.empty(output_length, dtype=float)
    time = np.arange(1 - window_size, 1, dtype=float)
    eps = 1e-12

    # Short scale: responsive endpoint estimate. Long scale: noise-resistant
    # reference trend. Both remain causal and endpoint aligned.
    short_weights = np.exp(time / max(0.26 * window_size, 1.0))
    long_weights = np.exp(time / max(0.62 * window_size, 1.0))
    short_weights /= np.sum(short_weights)
    long_weights /= np.sum(long_weights)

    level = 0.0
    velocity = 0.0
    pending_sign = 0
    pending_count = 0

    for i in range(output_length):
        window = x[i : i + window_size]

        short_observation, short_slope, short_scale = _robust_endpoint_fit(
            window,
            time,
            short_weights,
            persistent_velocity=velocity,
            apply_asymmetric_weighting=(i > 0),
        )
        long_observation, long_slope, long_scale = _robust_endpoint_fit(
            window,
            time,
            long_weights,
            persistent_velocity=velocity,
            apply_asymmetric_weighting=(i > 0),
        )

        scale = max(0.55 * short_scale + 0.45 * long_scale, 1e-8)

        # Short-scale evidence is trusted when both scales support the same
        # trend. Disagreement shifts weight toward the smoother long estimate.
        slope_agreement = short_slope * long_slope
        coherent_motion = abs(short_slope) / (abs(short_slope) + 1.35 * scale)
        if slope_agreement < 0.0:
            coherent_motion *= 0.42

        short_blend = np.clip(0.26 + 0.52 * coherent_motion, 0.26, 0.78)
        observation = (
            short_blend * short_observation
            + (1.0 - short_blend) * long_observation
        )
        slope = short_blend * short_slope + (1.0 - short_blend) * long_slope

        if i == 0:
            level = observation
            velocity = slope
            y[i] = level
            continue

        prediction = level + velocity
        innovation = observation - prediction

        # Innovation clipping prevents one window from changing level and
        # direction simultaneously. Coherent fitted motion expands the gate.
        innovation_limit = 2.55 * scale + 1.15 * abs(slope)
        clipped_innovation = float(
            np.clip(innovation, -innovation_limit, innovation_limit)
        )

        motion = abs(slope) + 0.24 * abs(clipped_innovation)
        alpha = motion / (motion + 1.55 * scale + eps)
        alpha = float(np.clip(alpha, 0.10, 0.68))
        level = prediction + alpha * clipped_innovation

        candidate_velocity = (
            0.84 * velocity
            + 0.16 * slope
            + 0.10 * alpha * clipped_innovation
        )

        deadband = 0.19 * scale + 0.035 * abs(velocity)
        old_sign = 1 if velocity > deadband else (-1 if velocity < -deadband else 0)
        new_sign = (
            1
            if candidate_velocity > deadband
            else (-1 if candidate_velocity < -deadband else 0)
        )

        if old_sign != 0 and new_sign != 0 and old_sign != new_sign:
            # A genuine turn has support from both local scales and sufficient
            # magnitude. Otherwise require three consecutive observations.
            coherent_turn = short_slope * long_slope > 0.0 and short_slope * velocity < 0.0
            strong_turn = (
                coherent_turn
                and abs(slope) > 0.34 * scale + 0.48 * abs(velocity)
            )

            if strong_turn:
                pending_sign = 0
                pending_count = 0
            elif pending_sign == new_sign:
                pending_count += 1
            else:
                pending_sign = new_sign
                pending_count = 1

            if not strong_turn and pending_count < 3:
                candidate_velocity = 0.70 * velocity
            else:
                pending_sign = 0
                pending_count = 0
        else:
            pending_sign = 0
            pending_count = 0

        # Suppress residual sub-noise jitter before it can produce output
        # slope flicker on subsequent predictions.
        if abs(candidate_velocity) < 0.070 * scale:
            candidate_velocity = 0.0

        velocity = candidate_velocity
        y[i] = level

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected filtering algorithm.

    Args:
        input_signal: Input time series data.
        window_size: Sliding window size.
        algorithm_type: "enhanced" for robust tracker; other values use basic.

    Returns:
        Filtered signal.
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
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