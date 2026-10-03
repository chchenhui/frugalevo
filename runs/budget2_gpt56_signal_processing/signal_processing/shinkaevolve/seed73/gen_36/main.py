# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

The enhanced implementation combines a robust adaptive Kalman tracker with
causal multi-scale endpoint regressions.  All estimates are aligned to the
newest sample in each trailing window, minimizing additional phase delay.
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
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def _endpoint_regression_coefficients(window_size, decay):
    """
    Construct causal weighted local-linear FIR coefficients.

    Coefficients are ordered oldest to newest and estimate both the level and
    slope at the newest sample, rather than at the center of the window.
    """
    time = np.arange(window_size, dtype=float) - float(window_size - 1)
    weights = np.exp(decay * time / float(max(window_size - 1, 1)))

    s0 = np.sum(weights)
    s1 = np.sum(weights * time)
    s2 = np.sum(weights * time * time)
    determinant = max(s0 * s2 - s1 * s1, 1e-12)

    level_coefficients = weights * (s2 - s1 * time) / determinant
    slope_coefficients = weights * (s0 * time - s1) / determinant
    return level_coefficients, slope_coefficients


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal adaptive Kalman and multi-scale endpoint-regression filter.

    A robust constant-velocity tracker gives a responsive state estimate. Three
    endpoint-aligned weighted regressions provide independent slow, balanced,
    and recent local measurements. Their confidence-weighted fusion reduces
    noisy directional changes without introducing centered-window lag.

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
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size == 1:
        return x.copy()

    n = len(x)
    n_out = n - window_size + 1
    initial = x[:window_size]
    differences = np.diff(initial)

    if differences.size:
        difference_median = np.median(differences)
        difference_scale = np.median(
            np.abs(differences - difference_median)
        ) / 0.6745
    else:
        difference_scale = 0.0

    measurement_var = max(0.50 * difference_scale * difference_scale, 1e-8)
    innovation_var = measurement_var
    process_var = max(0.05 * measurement_var, 1e-10)

    half = max(1, window_size // 2)
    initial_slope = (
        np.median(initial[half:]) - np.median(initial[:half])
    ) / float(max(window_size - half, 1))

    # Precompute endpoint measurements.  These convolutions are vectorized and
    # correspond exactly to samples x[window_size - 1:].
    regression_levels = []
    regression_slopes = []
    for decay in (1.5, 3.0, 5.0):
        level_coefficients, slope_coefficients = _endpoint_regression_coefficients(
            window_size, decay
        )
        regression_levels.append(
            np.convolve(x, level_coefficients[::-1], mode="valid")
        )
        regression_slopes.append(
            np.convolve(x, slope_coefficients[::-1], mode="valid")
        )

    slow_level, medium_level, fast_level = regression_levels
    slow_slope, medium_slope, fast_slope = regression_slopes

    # Difference-derived noise scale is causal, cheap, and less susceptible to
    # slow local trends than a level variance estimate.
    absolute_difference = np.abs(np.diff(x))
    difference_kernel = np.ones(window_size - 1, dtype=float) / float(window_size - 1)
    local_difference = np.convolve(
        absolute_difference, difference_kernel, mode="valid"
    )
    regression_noise = np.maximum(local_difference / 1.128379167, 1e-8)

    level = float(x[0])
    velocity = float(initial_slope)
    p00 = measurement_var * 4.0
    p01 = 0.0
    p11 = measurement_var

    exponential_decay = np.exp(-2.2 / float(max(window_size - 1, 1)))
    fallback_window_level = float(x[0])

    consensus_level = float(x[0])
    consensus_velocity = 0.0
    emitted_level = float(x[0])
    emitted_velocity = 0.0
    established_direction = 0
    reversal_evidence = 0

    output = np.empty(n_out, dtype=float)

    for index, sample in enumerate(x):
        # Adaptive constant-velocity Kalman prediction and robust update.
        predicted_level = level + velocity
        predicted_velocity = velocity
        q = process_var

        pp00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pp01 = p01 + p11 + 0.5 * q
        pp11 = p11 + q

        innovation = sample - predicted_level
        residual_var = max(pp00 + measurement_var, 1e-12)
        innovation_limit = 3.2 * np.sqrt(residual_var)
        bounded_innovation = np.clip(
            innovation, -innovation_limit, innovation_limit
        )

        gain_level = pp00 / residual_var
        gain_velocity = pp01 / residual_var
        level = predicted_level + gain_level * bounded_innovation
        velocity = predicted_velocity + gain_velocity * bounded_innovation

        p00 = max(pp00 - gain_level * pp00, 1e-12)
        p01 = pp01 - gain_level * pp01
        p11 = max(pp11 - gain_velocity * pp01, 1e-12)

        innovation_var = (
            0.955 * innovation_var + 0.045 * bounded_innovation * bounded_innovation
        )
        measurement_var = max(
            0.987 * measurement_var + 0.013 * innovation_var, 1e-8
        )
        process_var = max(
            0.992 * process_var + 0.008 * innovation_var * 0.055, 1e-10
        )

        fallback_window_level = (
            exponential_decay * fallback_window_level
            + (1.0 - exponential_decay) * sample
        )

        if index < window_size - 1:
            target = 0.78 * level + 0.22 * fallback_window_level
            scale = np.sqrt(max(innovation_var, 1e-10))
            trend_strength = abs(velocity) / (scale + 1e-12)
            slopes_agree = True
        else:
            position = index - window_size + 1
            scale = max(
                regression_noise[position],
                np.sqrt(max(innovation_var, 1e-10)),
                1e-8,
            )

            # Endpoint fits may use the recent estimator aggressively only when
            # all scales agree on direction. This avoids following isolated
            # endpoint noise while maintaining low lag on sustained motion.
            slopes_agree = (
                slow_slope[position] * medium_slope[position] > 0.0
                and medium_slope[position] * fast_slope[position] > 0.0
            )
            coherent_slope = min(
                abs(slow_slope[position]),
                abs(medium_slope[position]),
                abs(fast_slope[position]),
            )
            trend_strength = coherent_slope / (scale + 1e-12)

            endpoint_spread = abs(fast_level[position] - slow_level[position])
            fit_confidence = 1.0 / (
                1.0 + endpoint_spread / (2.0 * scale + 1e-12)
            )

            if slopes_agree:
                fast_weight = 0.14 + 0.42 * (
                    trend_strength / (trend_strength + 0.10)
                ) * fit_confidence
            else:
                fast_weight = 0.06 * fit_confidence

            fast_weight = float(np.clip(fast_weight, 0.04, 0.56))
            medium_weight = 0.25 + 0.10 * fit_confidence
            slow_weight = 1.0 - fast_weight - medium_weight

            regression_target = (
                slow_weight * slow_level[position]
                + medium_weight * medium_level[position]
                + fast_weight * fast_level[position]
            )

            kalman_disagreement = abs(level - regression_target)
            regression_weight = 0.22 + 0.25 * fit_confidence
            regression_weight /= (
                1.0 + kalman_disagreement / (3.0 * scale + 1e-12)
            )
            if slopes_agree and trend_strength > 0.10:
                regression_weight += 0.08 * fit_confidence
            regression_weight = float(np.clip(regression_weight, 0.14, 0.48))

            target = (
                (1.0 - regression_weight) * level
                + regression_weight * regression_target
            )

        # Predictive consensus tracking reduces lag relative to a level-only
        # smoother while still rejecting high-frequency measurement movement.
        predicted_consensus = consensus_level + consensus_velocity
        consensus_residual = target - predicted_consensus
        confidence = min(1.0, trend_strength / 0.18)
        alpha = 0.64 + 0.18 * confidence
        if not slopes_agree:
            alpha *= 0.90
        beta = 0.042 + 0.050 * min(
            1.0, abs(consensus_residual) / (2.5 * scale + 1e-12)
        )

        consensus_level = predicted_consensus + alpha * consensus_residual
        consensus_velocity += beta * consensus_residual

        proposed_delta = consensus_level - emitted_level
        proposed_direction = (
            1 if proposed_delta > 0.0 else (-1 if proposed_delta < 0.0 else 0)
        )
        adaptive_band = 0.075 * scale + 0.13 * abs(emitted_velocity)

        opposing = (
            established_direction != 0
            and proposed_direction != 0
            and proposed_direction != established_direction
        )

        accepted_delta = proposed_delta
        if abs(proposed_delta) <= adaptive_band:
            accepted_delta = 0.0
            reversal_evidence = 0
        elif opposing:
            velocity_support = (
                np.sign(consensus_velocity) == proposed_direction
                or abs(consensus_velocity) < 0.025 * scale
            )
            reversal_evidence = reversal_evidence + 1 if velocity_support else 0
            strong_turn = abs(proposed_delta) >= max(
                2.5 * adaptive_band, 0.34 * scale
            )

            if reversal_evidence < 2 and not strong_turn:
                accepted_delta = 0.0
            else:
                established_direction = proposed_direction
                reversal_evidence = 0
        else:
            reversal_evidence = 0
            if proposed_direction != 0:
                established_direction = proposed_direction

        # Do not jump directly to a delayed held target. The gain keeps release
        # from hysteresis smooth, reducing artificial reversal amplitudes.
        output_gain = 0.78 if slopes_agree else 0.68
        emitted_step = output_gain * accepted_delta
        emitted_level += emitted_step

        if abs(emitted_step) > 0.012 * scale:
            emitted_velocity = (
                emitted_step
                if abs(emitted_velocity) < 1e-12
                else 0.68 * emitted_velocity + 0.32 * emitted_step
            )
        else:
            emitted_velocity *= 0.68

        if index >= window_size - 1:
            output[index - window_size + 1] = emitted_level

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