# EVOLVE-BLOCK-START
"""
Robust low-lag adaptive filtering for volatile non-stationary signals.

The enhanced filter combines a clipped causal endpoint-regression bank with a
constant-velocity Kalman tracker and a direction-persistent predictive output
stage.  It is causal internally and returns samples aligned to x[window_size-1:].
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving-average baseline.

    Args:
        x: Input signal, a one-dimensional real-valued array.
        window_size: Number of samples per moving-average window.

    Returns:
        Filtered output aligned to x[window_size - 1:].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust causal trend-preserving adaptive filter.

    Parameters controlled internally:
        * Regression recency strengths: 1.6 and 4.2.
        * Window outlier clipping: +/- 3.0 robust window MAD.
        * Kalman innovation clipping: +/- 3.25 innovation standard deviations.
        * Trend confidence midpoint: slope displacement / residual MAD = 1.35.
        * Reversal evidence threshold: 0.18 residual MAD.
        * Standard reversal persistence: 3 samples.
        * Decisive-turn threshold: 0.46 residual MAD.

    Args:
        x: Input signal, a one-dimensional real-valued array.
        window_size: Causal local-context window size.

    Returns:
        Filtered signal with length len(x) - window_size + 1, aligned to
        x[window_size - 1:].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    n = len(x)
    if window_size == 1:
        return x.copy()

    initial = x[:window_size]
    initial_diff = np.diff(initial)
    diff_center = np.median(initial_diff) if initial_diff.size else 0.0
    diff_mad = (
        1.4826 * np.median(np.abs(initial_diff - diff_center))
        if initial_diff.size else 0.0
    )
    initial_noise = max(diff_mad, 1e-5)

    # ---- Robust causal endpoint-regression bank --------------------------
    # Window MAD is intentionally used only for clipping.  Trend confidence
    # below uses post-fit residual MAD, which does not falsely label a clean
    # long ramp as noisy solely because its endpoints are far apart.
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    time = np.arange(window_size, dtype=float) - float(window_size - 1)
    design = np.column_stack((np.ones(window_size, dtype=float), time))

    window_median = np.median(windows, axis=1, keepdims=True)
    window_mad = 1.4826 * np.median(
        np.abs(windows - window_median), axis=1, keepdims=True
    )
    clipping_scale = np.maximum(window_mad, initial_noise * 0.35)
    clipped = np.clip(
        windows,
        window_median - 3.0 * clipping_scale,
        window_median + 3.0 * clipping_scale,
    )

    endpoint_levels = []
    endpoint_slopes = []
    fitted_medium = None

    for profile_index, strength in enumerate((1.6, 4.2)):
        weights = np.exp(strength * time / max(window_size - 1, 1))
        normal = design.T @ (weights[:, None] * design)
        regression_map = np.linalg.solve(normal, design.T * weights)
        coefficients = clipped @ regression_map.T

        endpoint_levels.append(coefficients[:, 0])
        endpoint_slopes.append(coefficients[:, 1])

        if profile_index == 0:
            fitted_medium = (
                coefficients[:, :1]
                + coefficients[:, 1:] * time[None, :]
            )

    endpoint_levels = np.asarray(endpoint_levels)
    endpoint_slopes = np.asarray(endpoint_slopes)

    # Centered post-fit residual MAD: this is the relevant scale for deciding
    # whether endpoint slope and reversal evidence are genuine.
    residuals = clipped - fitted_medium
    residual_center = np.median(residuals, axis=1, keepdims=True)
    residual_noise = 1.4826 * np.median(
        np.abs(residuals - residual_center), axis=1
    )
    residual_noise = np.maximum(residual_noise, initial_noise * 0.12 + 1e-6)

    # Slope displacement is assessed across the local horizon rather than
    # comparing a one-sample slope to a full-window residual amplitude.
    horizon = float(window_size - 1)
    medium_slope = endpoint_slopes[0]
    recent_slope = endpoint_slopes[1]
    slope_evidence = np.abs(medium_slope) * horizon / residual_noise
    trend_confidence = slope_evidence / (slope_evidence + 1.35)

    same_direction = (
        np.sign(medium_slope) == np.sign(recent_slope)
    ) & (np.sign(medium_slope) != 0.0)
    slope_ratio = np.minimum(
        np.abs(medium_slope),
        np.abs(recent_slope),
    ) / (np.maximum(np.abs(medium_slope), np.abs(recent_slope)) + 1e-10)

    coherent_confidence = trend_confidence * same_direction * (
        0.45 + 0.55 * slope_ratio
    )

    # In quiet regions use the gentle fit.  Strong, coherent directional
    # evidence admits the more recent endpoint fit to reduce phase delay.
    recent_weight = 0.58 * coherent_confidence
    endpoint_reference = (
        (1.0 - recent_weight) * endpoint_levels[0]
        + recent_weight * endpoint_levels[1]
    )

    # ---- Adaptive constant-velocity state tracker ------------------------
    measurement_var = max(0.65 * initial_noise * initial_noise, 1e-8)
    innovation_var = measurement_var
    process_var = max(0.035 * measurement_var, 1e-10)

    split = max(1, window_size // 2)
    initial_slope = (
        np.median(initial[split:]) - np.median(initial[:split])
    ) / max(window_size - split, 1)

    level = float(x[0])
    velocity = float(initial_slope)
    p00 = 4.0 * measurement_var
    p01 = 0.0
    p11 = measurement_var

    warm_decay = np.exp(-2.3 / max(window_size - 1, 1))
    warm_reference = float(x[0])

    output = np.empty(n - window_size + 1, dtype=float)
    consensus_level = float(x[0])
    consensus_velocity = 0.0

    previous_step = 0.0
    accepted_direction = 0.0
    reversal_direction = 0.0
    reversal_evidence = 0

    for index, sample in enumerate(x):
        # Constant-velocity Kalman prediction.
        predicted_level = level + velocity
        q = process_var
        pp00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pp01 = p01 + p11 + 0.5 * q
        pp11 = p11 + q

        innovation = sample - predicted_level
        total_var = max(pp00 + measurement_var, 1e-12)
        innovation_limit = 3.25 * np.sqrt(total_var)
        bounded_innovation = np.clip(
            innovation, -innovation_limit, innovation_limit
        )

        k0 = pp00 / total_var
        k1 = pp01 / total_var
        level = predicted_level + k0 * bounded_innovation
        velocity = velocity + k1 * bounded_innovation

        p00 = max(pp00 - k0 * pp00, 1e-12)
        p01 = pp01 - k0 * pp01
        p11 = max(pp11 - k1 * pp01, 1e-12)

        innovation_var = (
            0.965 * innovation_var + 0.035 * bounded_innovation ** 2
        )
        measurement_var = max(
            0.992 * measurement_var + 0.008 * innovation_var,
            1e-8,
        )
        process_var = max(
            0.992 * process_var + 0.008 * innovation_var * 0.040,
            1e-10,
        )

        warm_reference = (
            warm_decay * warm_reference + (1.0 - warm_decay) * sample
        )

        if index >= window_size - 1:
            local_index = index - window_size + 1
            local_reference = endpoint_reference[local_index]
            local_noise = residual_noise[local_index]
            confidence = coherent_confidence[local_index]
            local_slope = medium_slope[local_index]
        else:
            local_reference = warm_reference
            local_noise = max(initial_noise, 1e-6)
            confidence = 0.0
            local_slope = 0.0

        innovation_scale = np.sqrt(max(innovation_var, 1e-10))
        scale = max(local_noise, 0.65 * innovation_scale, 1e-6)
        disagreement = abs(level - local_reference)

        # Local endpoint fitting is most useful when it agrees with the
        # tracker; coherence increases its contribution only for real trends.
        agreement = 1.0 / (1.0 + (disagreement / (2.0 * scale)) ** 2)
        local_weight = (0.20 + 0.30 * confidence) * agreement
        target = (1.0 - local_weight) * level + local_weight * local_reference

        # Predict around velocity, rather than around a stationary mean, to
        # retain responsiveness and avoid phase delay on sustained movement.
        previous_level = consensus_level
        prediction = consensus_level + consensus_velocity
        residual = target - prediction
        alpha = 0.66 + 0.18 * min(
            1.0, disagreement / (2.6 * scale + 1e-12)
        )
        candidate = prediction + alpha * residual
        step = candidate - previous_level
        step_direction = np.sign(step)

        # A turn must be supported by more than a single target crossing.
        # Endpoint slope is included as independent evidence, allowing a
        # genuine sharp turn to release before the Kalman velocity fully flips.
        opposing = (
            accepted_direction != 0.0
            and step_direction != 0.0
            and step_direction == -accepted_direction
        )
        slope_support = (
            np.sign(local_slope) == step_direction
            and abs(local_slope) * horizon >= 0.20 * scale
        )
        velocity_support = (
            np.sign(velocity) == step_direction
            and abs(velocity) >= 0.050 * scale
        )
        evidence_floor = 0.18 * scale
        strong_step = abs(step) >= evidence_floor
        decisive_turn = (
            abs(step) >= 0.46 * scale
            and slope_support
            and velocity_support
        )

        if opposing and not decisive_turn:
            if strong_step and (slope_support or velocity_support):
                if step_direction == reversal_direction:
                    reversal_evidence += 1
                else:
                    reversal_direction = step_direction
                    reversal_evidence = 1
            else:
                reversal_direction = 0.0
                reversal_evidence = 0

            if reversal_evidence < 3:
                consensus_level = previous_level
                step = 0.0
            else:
                consensus_level = candidate
                reversal_direction = 0.0
                reversal_evidence = 0
        else:
            consensus_level = candidate
            reversal_direction = 0.0
            reversal_evidence = 0

        if step != 0.0:
            previous_step = step
            accepted_direction = np.sign(step)

        velocity_target = 0.62 * velocity + 0.38 * step
        acceleration_limit = 0.085 * scale + 0.030 * abs(velocity)
        velocity_delta = np.clip(
            velocity_target - consensus_velocity,
            -acceleration_limit,
            acceleration_limit,
        )
        consensus_velocity += 0.44 * velocity_delta

        deadband = 0.035 * scale
        if abs(consensus_velocity) < deadband:
            consensus_velocity = 0.0

        if index >= window_size - 1:
            output[index - window_size + 1] = consensus_level

    return output


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected filtering algorithm.

    Args:
        input_signal: Input time series.
        window_size: Sliding-window length.
        algorithm_type: "basic" or "enhanced".

    Returns:
        Filtered time series.
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(
            input_signal, window_size
        )
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