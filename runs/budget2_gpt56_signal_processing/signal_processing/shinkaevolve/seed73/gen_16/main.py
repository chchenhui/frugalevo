# EVOLVE-BLOCK-START
"""
Multiscale robust consensus filtering for volatile non-stationary signals.

The enhanced mode combines two causal robust endpoint regressions with a
bounded-velocity tracker.  A short regression horizon supplies early turn
evidence, while a longer horizon supplies stable trend confirmation.  The
tracker only accepts weak reversals after directional persistence, reducing
noise-induced slope changes without imposing moving-average phase delay.
"""
import numpy as np


def _validate_signal_and_window(x, window_size):
    """Validate compatible public API inputs."""
    signal = np.asarray(x, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array of real-valued samples")
    if int(window_size) != window_size or window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    window_size = int(window_size)
    if len(signal) < window_size:
        raise ValueError(
            f"Input signal length ({len(signal)}) must be >= window_size ({window_size})"
        )
    if not np.all(np.isfinite(signal)):
        raise ValueError("Input signal must contain only finite real-valued samples")

    return signal, window_size


def adaptive_filter(x, window_size=20):
    """
    Baseline trailing moving-average filter.

    Returns:
        Filtered signal with length len(x) - window_size + 1.
    """
    signal, window_size = _validate_signal_and_window(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def _robust_endpoint_features(signal, horizon):
    """
    Vectorized causal robust local-linear endpoint estimates.

    A preliminary exponentially weighted fit establishes a local line. One
    Huber reweighting pass then limits samples that deviate sharply from that
    local trajectory.  The returned intercept represents the newest sample of
    every trailing horizon window.
    """
    windows = np.lib.stride_tricks.sliding_window_view(signal, horizon)
    time = np.arange(-(horizon - 1), 1, dtype=float)

    # A moderate decay preserves endpoint responsiveness without allowing the
    # final noisy sample to dominate the regression.
    decay = max(2.0, horizon / 2.8)
    base_weight = np.exp(time / decay)
    base_weight /= np.sum(base_weight)

    def weighted_line(weights):
        sum_w = np.sum(weights, axis=1)
        sum_t = np.sum(weights * time, axis=1)
        sum_tt = np.sum(weights * time * time, axis=1)
        sum_x = np.sum(weights * windows, axis=1)
        sum_tx = np.sum(weights * windows * time, axis=1)

        denominator = sum_w * sum_tt - sum_t * sum_t
        denominator = np.maximum(denominator, 1e-12)

        slope = (sum_w * sum_tx - sum_t * sum_x) / denominator
        level = (sum_x - slope * sum_t) / np.maximum(sum_w, 1e-12)
        return level, slope

    initial_weights = np.broadcast_to(base_weight, windows.shape)
    level, slope = weighted_line(initial_weights)

    residual = windows - level[:, None] - slope[:, None] * time[None, :]
    residual_median = np.median(residual, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(residual - residual_median), axis=1)
    scale = np.maximum(scale, 1e-8)

    # Huber weighting preserves samples on the inferred local trend while
    # reducing the influence of impulsive noise.
    huber = np.minimum(1.0, 2.2 * scale[:, None] / (np.abs(residual) + 1e-12))
    level, slope = weighted_line(initial_weights * huber)

    final_residual = windows - level[:, None] - slope[:, None] * time[None, :]
    final_center = np.median(final_residual, axis=1, keepdims=True)
    residual_scale = 1.4826 * np.median(
        np.abs(final_residual - final_center), axis=1
    )
    residual_scale = np.maximum(residual_scale, 1e-8)

    return level, slope, residual_scale


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal multiscale trend-consensus filter.

    Output i is aligned with input sample x[i + window_size - 1].  The filter
    uses short/long robust local trends for low-lag evidence, then applies a
    bounded velocity-consensus tracker to smooth noise and reject false turns.
    """
    signal, window_size = _validate_signal_and_window(x, window_size)

    if window_size == 1:
        return signal.copy()

    # The short scale responds quickly to real curvature.  The long scale
    # provides a more conservative estimate used to validate direction.
    short_horizon = min(
        window_size,
        max(2, int(round(0.42 * window_size)))
    )

    long_level, long_slope, long_noise = _robust_endpoint_features(
        signal, window_size
    )
    short_level_all, short_slope_all, short_noise_all = _robust_endpoint_features(
        signal, short_horizon
    )

    # Align short-window estimates to the endpoints represented by long windows.
    endpoint_offset = window_size - short_horizon
    short_level = short_level_all[endpoint_offset:]
    short_slope = short_slope_all[endpoint_offset:]
    short_noise = short_noise_all[endpoint_offset:]

    count = len(long_level)
    output = np.empty(count, dtype=float)

    # Initial trend is taken from the conservative horizon to avoid startup
    # overreaction to a single short-window fluctuation.
    level = float(long_level[0])
    velocity = float(long_slope[0])
    pending_direction = 0
    pending_count = 0

    output[0] = level

    for i in range(1, count):
        noise = max(
            0.55 * long_noise[i] + 0.45 * short_noise[i],
            1e-8,
        )

        fast_slope = short_slope[i]
        slow_slope = long_slope[i]

        # Agreement measures whether the faster estimator is observing genuine
        # motion or merely reacting to local noise.
        slope_gap = abs(fast_slope - slow_slope)
        agreement = 1.0 / (1.0 + slope_gap / noise)
        same_direction = fast_slope * slow_slope >= 0.0

        # Fast endpoint estimates get more influence only when supported by
        # long-scale direction evidence. This keeps turns responsive but limits
        # high-frequency endpoint jitter.
        fast_weight = 0.22 + 0.38 * agreement
        if not same_direction:
            fast_weight *= 0.55

        measurement = (
            fast_weight * short_level[i]
            + (1.0 - fast_weight) * long_level[i]
        )

        consensus_slope = (
            (0.62 if same_direction else 0.40) * fast_slope
            + (0.38 if same_direction else 0.60) * slow_slope
        )

        prediction = level + velocity
        raw_innovation = measurement - prediction
        innovation_limit = 3.2 * noise + 0.35 * abs(velocity)
        innovation = np.clip(
            raw_innovation,
            -innovation_limit,
            innovation_limit,
        )

        trend_evidence = abs(consensus_slope) / (
            abs(consensus_slope) + noise
        )
        alpha = 0.28 + 0.30 * trend_evidence + 0.10 * agreement
        candidate = prediction + alpha * innovation
        step = candidate - level

        # Velocity consensus explicitly combines direct local slope evidence
        # with the realized tracker step. This is less reactive than updating
        # velocity only from innovations, yet avoids fixed-window lag.
        target_velocity = 0.55 * consensus_slope + 0.45 * step

        opposite = velocity * target_velocity < 0.0
        direction = 1 if target_velocity > 0.0 else -1
        reversal_strength = abs(target_velocity) / (noise + 0.20 * abs(velocity) + 1e-12)
        strong_reversal = (
            reversal_strength > 0.60
            and abs(consensus_slope) > 0.18 * noise
            and (same_direction or abs(fast_slope) > 0.75 * noise)
        )

        # Require two weak opposite-direction observations. Strong multiscale
        # evidence bypasses the delay, which preserves response at real turns.
        if opposite and not strong_reversal:
            if direction == pending_direction:
                pending_count += 1
            else:
                pending_direction = direction
                pending_count = 1

            if pending_count < 2:
                target_velocity = 0.45 * velocity
                candidate = level + target_velocity
                step = candidate - level
            else:
                pending_count = 0
        else:
            pending_direction = 0
            pending_count = 0

        # Bound acceleration according to local residual volatility.  This
        # prevents isolated endpoint excursions from injecting a large velocity
        # reversal into subsequent predictions.
        velocity_change = target_velocity - velocity
        acceleration_limit = 0.34 * noise + 0.55 * abs(velocity) + 1e-10
        velocity_change = np.clip(
            velocity_change,
            -acceleration_limit,
            acceleration_limit,
        )

        velocity_gain = 0.30 + 0.28 * trend_evidence + 0.12 * agreement
        velocity += velocity_gain * velocity_change

        level = candidate
        output[i] = level

    return output


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected compatible filter mode.

    Args:
        input_signal: Input one-dimensional time series.
        window_size: Trailing window and output-alignment size.
        algorithm_type: "enhanced" selects multiscale trend tracking.

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
