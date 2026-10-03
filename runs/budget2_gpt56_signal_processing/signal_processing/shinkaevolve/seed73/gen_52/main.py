# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This implementation combines robust sliding-window local regression with a
causal adaptive alpha-beta trend tracker for low-lag noise suppression.
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
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    x = np.asarray(x, dtype=float)
    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(x, kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust low-lag local-trend filter.

    A recency-weighted local linear fit produces an endpoint measurement and a
    slope estimate for every trailing window. A causal adaptive alpha-beta
    tracker then combines that low-lag measurement with a persistent velocity
    state. Innovation gating and directional hysteresis suppress isolated noise
    spikes and false reversals without excessively delaying real turns.
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
    output_length = len(windows)

    # Local robust clipping prevents a single volatile observation from
    # distorting either the fitted endpoint or the fitted slope.
    medians = np.median(windows, axis=1)
    deviations = np.abs(windows - medians[:, None])
    mad = np.median(deviations, axis=1)
    noise = np.maximum(1.4826 * mad, np.finfo(float).eps)

    clipped = np.clip(
        windows,
        (medians - 3.25 * noise)[:, None],
        (medians + 3.25 * noise)[:, None],
    )

    # Fit y = intercept + slope * t, where t=0 is the newest observation.
    # Exponential recency weighting limits non-stationary lag while retaining
    # enough history to distinguish a real local slope from sample noise.
    t = np.arange(window_size, dtype=float) - (window_size - 1)
    weights = np.exp(np.linspace(-2.7, 0.0, window_size))
    design = np.column_stack((np.ones(window_size), t))
    normal = design.T @ (weights[:, None] * design)
    fit_matrix = np.linalg.solve(normal, design.T * weights)

    measurements = clipped @ fit_matrix[0]
    local_slopes = clipped @ fit_matrix[1]

    y = np.empty(output_length, dtype=float)
    level = measurements[0]
    velocity = local_slopes[0]
    residual_scale = max(noise[0], np.finfo(float).eps)
    previous_direction = np.sign(velocity)
    reversal_evidence = 0

    # Store one suppressed countertrend move.  A reversal is released only
    # when the next window independently confirms its direction.
    pending_direction = 0.0
    pending_step = 0.0

    # Gain changes are mode based rather than sample-by-sample.  This avoids
    # transmitting noisy confidence fluctuations directly into output motion.
    trend_mode = False
    high_confidence_count = 0
    low_confidence_count = 0

    y[0] = level

    for i in range(1, output_length):
        prediction = level + velocity
        innovation = measurements[i] - prediction

        # Scale robustly tracks ordinary local residuals but does not inflate
        # permanently due to an isolated large jump.
        gate = max(2.8 * residual_scale, 0.65 * noise[i], 1e-10)
        bounded_innovation = np.clip(innovation, -2.5 * gate, 2.5 * gate)

        slope_strength = abs(local_slopes[i]) / (
            abs(local_slopes[i]) + noise[i] + 1e-12
        )
        activity = min(abs(innovation) / (gate + 1e-12), 1.0)

        measurement_direction = np.sign(local_slopes[i])
        if (
            measurement_direction != 0.0
            and previous_direction != 0.0
            and measurement_direction != previous_direction
        ):
            reversal_evidence = min(reversal_evidence + 1, 4)
        elif measurement_direction == previous_direction and measurement_direction != 0.0:
            reversal_evidence = max(reversal_evidence - 1, 0)
        else:
            reversal_evidence = max(reversal_evidence - 1, 0)

        # Normalize slope over an effective recent fitting horizon.  Entering
        # trend mode requires two independent high-confidence windows, whereas
        # leaving it requires a longer run of weak evidence.  The asymmetry
        # suppresses chatter around the confidence boundary.
        effective_horizon = max(window_size / 3.0, 1.0)
        slope_confidence = (
            abs(local_slopes[i]) * effective_horizon
            / (abs(local_slopes[i]) * effective_horizon + noise[i] + 1e-12)
        )
        if slope_confidence >= 0.34 and measurement_direction != 0.0:
            high_confidence_count = min(high_confidence_count + 1, 2)
            low_confidence_count = 0
        elif slope_confidence <= 0.18:
            low_confidence_count = min(low_confidence_count + 1, 4)
            high_confidence_count = 0
        else:
            high_confidence_count = 0
            low_confidence_count = max(low_confidence_count - 1, 0)

        if not trend_mode and high_confidence_count >= 2:
            trend_mode = True
        elif trend_mode and low_confidence_count >= 4:
            trend_mode = False

        # Stable discrete gains provide substantially smoother behavior in
        # flat/noisy regions.  Trend mode regains responsiveness only after
        # persistent directional evidence, with extra gain for a confirmed
        # directional transition.
        turn_strength = reversal_evidence / 4.0
        if trend_mode:
            alpha = 0.52 + 0.13 * activity + 0.06 * turn_strength
        else:
            alpha = 0.33 + 0.08 * activity + 0.03 * turn_strength
        alpha = min(alpha, 0.72)

        candidate = prediction + alpha * bounded_innovation
        step = candidate - level

        # Suppress a single weak countertrend step completely.  If the next
        # local fit and candidate agree with the stored direction and provide
        # a material move, release it as a confirmed turn.  This avoids the
        # small leaky opposing steps that otherwise create false reversals.
        reversal_band = 0.11 * noise[i] + 0.10 * abs(velocity)
        countertrend = step * velocity < 0.0
        step_direction = np.sign(step)
        slope_direction = np.sign(local_slopes[i])
        confirmation_band = max(0.30 * residual_scale, 0.18 * noise[i])

        confirmed_turn = (
            pending_direction != 0.0
            and step_direction == pending_direction
            and slope_direction == pending_direction
            and abs(step) >= confirmation_band
        )

        if countertrend and reversal_evidence < 2 and not confirmed_turn:
            pending_direction = step_direction
            pending_step = step
            candidate = level
            step = 0.0
        else:
            if confirmed_turn:
                # Preserve the responsive current estimate while requiring
                # direction agreement from two consecutive trailing windows.
                step = candidate - level
            pending_direction = 0.0
            pending_step = 0.0

        # Blend dynamic innovation velocity with robust local fitted slope.
        # Local slope provides anticipatory trend information; beta correction
        # prevents it from creating rapid noise-induced sign alternation.
        if trend_mode:
            beta = 0.13 + 0.10 * activity + 0.08 * slope_strength
            slope_blend = 0.25 + 0.14 * slope_strength
        else:
            beta = 0.07 + 0.05 * activity + 0.05 * slope_strength
            slope_blend = 0.12 + 0.10 * slope_strength
        innovation_velocity = velocity + beta * bounded_innovation
        new_velocity = (
            (1.0 - slope_blend) * innovation_velocity
            + slope_blend * local_slopes[i]
        )

        if (
            new_velocity * velocity < 0.0
            and abs(new_velocity) < 0.14 * noise[i]
            and reversal_evidence < 2
        ):
            new_velocity = 0.35 * velocity

        level = candidate
        velocity = new_velocity
        if abs(velocity) > 0.04 * noise[i]:
            previous_direction = np.sign(velocity)

        residual_scale = 0.97 * residual_scale + 0.03 * min(
            abs(innovation), 2.5 * gate
        )
        residual_scale = max(residual_scale, 0.35 * noise[i], 1e-10)
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