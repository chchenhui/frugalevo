# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

The enhanced filter uses robust multi-scale local regression followed by a
causal alpha-beta tracker with hysteretic trend modes and reversal confirmation.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Sliding-window moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        Filtered output with length len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def _endpoint_fit_matrix(window_size, decay):
    """Return weighted least-squares coefficients for endpoint and slope."""
    t = np.arange(window_size, dtype=float) - (window_size - 1)
    weights = np.exp(np.linspace(-decay, 0.0, window_size))
    design = np.column_stack((np.ones(window_size), t))
    normal = design.T @ (weights[:, None] * design)
    return np.linalg.solve(normal, design.T * weights)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal robust low-lag filter for volatile non-stationary signals.

    The newest sample of each trailing window is the output time reference.
    Three exponentially weighted line fits provide robust slow, medium, and
    recent trend measurements.  Their combination feeds an alpha-beta state
    tracker with hysteretic flat/trend modes and confirmed reversal handling.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite real-valued samples")

    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    n_output = len(windows)
    eps = np.finfo(float).eps

    # Robust local scale and clipping ensure isolated spikes do not cause a
    # large fitted endpoint or an artificial slope reversal.
    median = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - median[:, None]), axis=1)
    noise = np.maximum(1.4826 * mad, eps)
    clipped = np.clip(
        windows,
        (median - 3.2 * noise)[:, None],
        (median + 3.2 * noise)[:, None],
    )

    # Slow fit stabilizes flat/noisy intervals; recent fit is used only after
    # multi-scale directional agreement confirms a genuine active trend.
    slow_fit = _endpoint_fit_matrix(window_size, 1.45)
    medium_fit = _endpoint_fit_matrix(window_size, 2.7)
    recent_fit = _endpoint_fit_matrix(window_size, 4.4)

    endpoint_slow = clipped @ slow_fit[0]
    endpoint_medium = clipped @ medium_fit[0]
    endpoint_recent = clipped @ recent_fit[0]

    slope_slow = clipped @ slow_fit[1]
    slope_medium = clipped @ medium_fit[1]
    slope_recent = clipped @ recent_fit[1]

    sign_medium = np.sign(slope_medium)
    agreement = (
        (np.sign(slope_slow) == sign_medium)
        & (np.sign(slope_recent) == sign_medium)
        & (sign_medium != 0.0)
    )

    # Slope is normalized by the approximate accumulated change across the
    # useful recent part of the window, not just by one-sample noise.
    slope_snr = (
        np.abs(slope_medium) * max(0.42 * window_size, 1.0) / (noise + 1e-12)
    )
    confidence = np.clip((slope_snr - 0.24) / 1.05, 0.0, 1.0)
    confidence *= agreement.astype(float)

    # Conservative blend is default.  The endpoint moves toward the recent
    # estimate only with directional multi-scale agreement.
    recent_weight = 0.34 * confidence
    slow_weight = 0.30 * (1.0 - confidence)
    medium_weight = 1.0 - slow_weight - recent_weight

    measurements = (
        slow_weight * endpoint_slow
        + medium_weight * endpoint_medium
        + recent_weight * endpoint_recent
    )
    local_slopes = (
        slow_weight * slope_slow
        + medium_weight * slope_medium
        + recent_weight * slope_recent
    )

    y = np.empty(n_output, dtype=float)
    level = measurements[0]
    velocity = local_slopes[0]
    residual_scale = max(noise[0], eps)

    # Discrete gain mode prevents gain jitter from directly becoming output
    # direction jitter. Enter trend mode quickly enough for low lag, but leave
    # it only after sustained low-confidence evidence.
    trend_mode = False
    high_confidence_count = 0
    low_confidence_count = 0

    active_direction = np.sign(velocity)
    reversal_direction = 0.0
    reversal_count = 0
    y[0] = level

    for i in range(1, n_output):
        slope = local_slopes[i]
        slope_direction = np.sign(slope)
        confidence_i = confidence[i]

        if confidence_i > 0.52 and slope_direction != 0.0:
            high_confidence_count += 1
            low_confidence_count = 0
        elif confidence_i < 0.27:
            low_confidence_count += 1
            high_confidence_count = max(high_confidence_count - 1, 0)
        else:
            high_confidence_count = max(high_confidence_count - 1, 0)
            low_confidence_count = max(low_confidence_count - 1, 0)

        if not trend_mode and high_confidence_count >= 2:
            trend_mode = True
            low_confidence_count = 0
        elif trend_mode and low_confidence_count >= 4:
            trend_mode = False
            high_confidence_count = 0

        prediction = level + velocity
        innovation = measurements[i] - prediction
        gate = max(2.55 * residual_scale, 0.62 * noise[i], 1e-10)
        bounded_innovation = np.clip(innovation, -2.35 * gate, 2.35 * gate)
        activity = min(abs(innovation) / (gate + 1e-12), 1.0)

        # Gain sets are intentionally discrete. Flat mode favors smoothness;
        # trend mode gains responsiveness after independently confirmed slope.
        if trend_mode:
            alpha = 0.54 + 0.11 * confidence_i + 0.06 * activity
            beta = 0.115 + 0.075 * confidence_i
            slope_blend = 0.24 + 0.12 * confidence_i
        else:
            alpha = 0.31 + 0.08 * confidence_i + 0.035 * activity
            beta = 0.055 + 0.045 * confidence_i
            slope_blend = 0.11 + 0.10 * confidence_i

        candidate = prediction + min(alpha, 0.73) * bounded_innovation
        step = candidate - level
        step_direction = np.sign(step)

        velocity_direction = np.sign(velocity)
        countertrend = (
            velocity_direction != 0.0
            and step_direction != 0.0
            and step_direction != velocity_direction
        )

        # A reversal must persist in local slope estimates. Large innovations
        # can still pass quickly, preserving response to genuine sharp turns.
        reversal_band = max(
            0.15 * noise[i] + 0.12 * abs(velocity),
            0.16 * residual_scale,
        )
        strong_counter_move = abs(step) > 1.15 * gate

        if (
            countertrend
            and slope_direction == step_direction
            and confidence_i > 0.30
        ):
            if reversal_direction == step_direction:
                reversal_count = min(reversal_count + 1, 5)
            else:
                reversal_direction = step_direction
                reversal_count = 1
        elif slope_direction == velocity_direction or slope_direction == 0.0:
            reversal_count = max(reversal_count - 1, 0)
            if reversal_count == 0:
                reversal_direction = 0.0

        required_confirmations = 2 if trend_mode else 3
        turn_confirmed = (
            reversal_count >= required_confirmations
            or (strong_counter_move and confidence_i > 0.60)
        )

        if countertrend and not turn_confirmed:
            # Hold weak unconfirmed opposing movement rather than leaking a
            # succession of tiny direction changes into the output.
            if abs(step) < reversal_band:
                candidate = level
                step = 0.0
            else:
                candidate = level + 0.18 * step
                step = candidate - level

        innovation_velocity = velocity + beta * bounded_innovation
        new_velocity = (
            (1.0 - slope_blend) * innovation_velocity
            + slope_blend * slope
        )

        # Keep velocity direction stable until the same evidence that permits
        # a level reversal is present.
        if (
            velocity_direction != 0.0
            and np.sign(new_velocity) != 0.0
            and np.sign(new_velocity) != velocity_direction
            and not turn_confirmed
        ):
            new_velocity = 0.55 * velocity

        velocity_deadband = 0.035 * noise[i] + 0.018 * residual_scale
        if abs(new_velocity) < velocity_deadband and not turn_confirmed:
            new_velocity = 0.0

        level = candidate
        velocity = new_velocity
        if abs(velocity) > velocity_deadband:
            active_direction = np.sign(velocity)

        residual_scale = 0.95 * residual_scale + 0.05 * min(
            abs(innovation), 2.35 * gate
        )
        residual_scale = max(residual_scale, 0.32 * noise[i], 1e-10)
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