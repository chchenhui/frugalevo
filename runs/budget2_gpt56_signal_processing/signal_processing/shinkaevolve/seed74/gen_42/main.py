# EVOLVE-BLOCK-START
"""
Robust adaptive Kalman filtering for volatile non-stationary signals.

The enhanced path uses:
  1. A causal local-linear Kalman state model for low-lag tracking.
  2. Robust trailing noise estimation from first differences.
  3. Innovation clipping to reject isolated spikes.
  4. Adaptive process noise to track genuine fast motion.
  5. CUSUM-confirmed directional hysteresis to suppress false reversals.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples).
        window_size: Trailing window size.

    Returns:
        Moving-average output with length len(x) - window_size + 1.
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
    Causal robust adaptive Kalman filter with CUSUM reversal confirmation.

    A two-state model estimates instantaneous level and slope. The observation
    update is robustified by clipping innovations relative to a local MAD noise
    estimate. Process noise grows during coherent movement, allowing rapid
    tracking of legitimate non-stationary changes without permanently raising
    noise sensitivity.

    Displayed direction is controlled by evidence accumulation rather than by
    direct sample-by-sample slope signs. Small counter-trend fluctuations decay
    the accumulated evidence, whereas sustained movement or a decisive jump
    releases a reversal promptly.

    Args:
        x: Input signal, a one-dimensional real-valued array.
        window_size: Required causal trailing history size.

    Returns:
        Filtered endpoint-aligned signal of length len(x) - window_size + 1.
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
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite values")

    output_length = len(x) - window_size + 1
    y = np.empty(output_length, dtype=float)
    eps = 1e-12

    # The output starts at the newest sample of the first causal window.
    initial_window = x[:window_size]
    initial_tail = initial_window[-min(5, window_size):]
    level = float(np.median(initial_tail))

    if len(initial_tail) >= 3:
        initial_differences = np.diff(initial_tail)
        velocity = float(np.median(initial_differences))
    else:
        velocity = float(initial_tail[-1] - initial_tail[0])

    # Covariance of [level, velocity]. A broad initial covariance lets the
    # state settle quickly after the initial window.
    initial_scale = 1.4826 * np.median(
        np.abs(np.diff(initial_window) - np.median(np.diff(initial_window)))
    ) / np.sqrt(2.0)
    initial_scale = max(float(initial_scale), 1e-5)

    p00 = 3.0 * initial_scale * initial_scale
    p01 = 0.0
    p11 = 0.40 * initial_scale * initial_scale

    displayed_level = level
    displayed_direction = 0
    reversal_evidence = 0.0

    noise_memory = initial_scale

    for i in range(output_length):
        window = x[i:i + window_size]
        measurement = float(window[-1])

        differences = np.diff(window)
        diff_center = np.median(differences)
        robust_diff_scale = 1.4826 * np.median(
            np.abs(differences - diff_center)
        )
        local_noise = robust_diff_scale / np.sqrt(2.0)

        # Smooth the scale estimate so one unusually quiet or volatile window
        # does not cause unstable gain changes.
        local_noise = max(float(local_noise), 1e-5)
        noise_memory = 0.82 * noise_memory + 0.18 * local_noise
        noise_scale = max(noise_memory, 0.35 * local_noise, 1e-5)
        measurement_variance = noise_scale * noise_scale

        if i == 0:
            # Robust initial endpoint adjustment: preserve endpoint alignment
            # but prevent a single startup spike from setting the state.
            startup_limit = 2.8 * noise_scale
            level = level + np.clip(measurement - level, -startup_limit, startup_limit)
            displayed_level = level
            if abs(velocity) > 0.06 * noise_scale:
                displayed_direction = 1 if velocity > 0.0 else -1
            y[i] = displayed_level
            continue

        # Constant-velocity state prediction.
        predicted_level = level + velocity
        predicted_velocity = velocity

        predicted_p00 = p00 + 2.0 * p01 + p11
        predicted_p01 = p01 + p11
        predicted_p11 = p11

        raw_innovation = measurement - predicted_level

        # A bounded innovation makes the state resistant to impulses while
        # retaining enough correction range for real discontinuities.
        innovation_limit = 3.4 * noise_scale + 0.45 * abs(velocity)
        innovation = float(
            np.clip(raw_innovation, -innovation_limit, innovation_limit)
        )

        normalized_innovation = abs(raw_innovation) / (noise_scale + eps)
        activity = min(normalized_innovation / 2.5, 1.0)

        # Adaptive acceleration uncertainty. Quiet sections use a small q for
        # smoothness; sustained innovations increase q and reduce lag.
        coherent_motion = min(
            abs(np.median(differences)) / (noise_scale + eps),
            3.0,
        )
        process_factor = 0.018 + 0.070 * activity + 0.025 * coherent_motion
        q_velocity = process_factor * measurement_variance
        q_level = (0.08 + 0.20 * activity) * measurement_variance

        predicted_p00 += q_level + 0.25 * q_velocity
        predicted_p01 += 0.50 * q_velocity
        predicted_p11 += q_velocity

        innovation_variance = predicted_p00 + measurement_variance
        kalman_level_gain = predicted_p00 / max(innovation_variance, eps)
        kalman_velocity_gain = predicted_p01 / max(innovation_variance, eps)

        level = predicted_level + kalman_level_gain * innovation
        velocity = predicted_velocity + kalman_velocity_gain * innovation

        # Covariance update in compact Joseph-stable form for H=[1, 0].
        new_p00 = (1.0 - kalman_level_gain) * predicted_p00
        new_p01 = (1.0 - kalman_level_gain) * predicted_p01
        new_p11 = predicted_p11 - kalman_velocity_gain * predicted_p01

        p00 = max(new_p00, eps)
        p01 = new_p01
        p11 = max(new_p11, eps)

        # Predict slightly ahead only during strong, coherent movement. This
        # offsets causal filtering lag without exposing raw observation noise.
        predictive_fraction = 0.10 + 0.10 * activity
        target = level + predictive_fraction * velocity
        delta = target - displayed_level

        # Direction-dependent hysteresis: entering motion from neutral uses a
        # modest threshold, but opposite motion needs substantially stronger,
        # accumulated evidence before a reversal is accepted.
        enter_deadband = 0.075 * noise_scale
        oppose_deadband = 0.17 * noise_scale + 0.025 * abs(velocity)

        desired_direction = 0
        if delta > enter_deadband:
            desired_direction = 1
        elif delta < -enter_deadband:
            desired_direction = -1

        if displayed_direction == 0:
            if desired_direction != 0:
                displayed_direction = desired_direction
                reversal_evidence = 0.0

        elif desired_direction == displayed_direction:
            reversal_evidence *= 0.45

        elif desired_direction != 0:
            opposing_strength = max(0.0, abs(delta) - oppose_deadband)
            reversal_evidence = 0.70 * reversal_evidence + opposing_strength

            # A persistent counter-trend must move materially beyond the
            # expected local noise. Large discontinuities bypass accumulation.
            confirmation_threshold = (
                1.15 * noise_scale + 0.35 * abs(velocity)
            )
            decisive_turn = abs(delta) > (
                2.45 * noise_scale + 0.55 * abs(velocity)
            )

            if reversal_evidence >= confirmation_threshold or decisive_turn:
                displayed_direction = desired_direction
                reversal_evidence = 0.0
            else:
                delta = 0.0
        else:
            reversal_evidence *= 0.75
            delta = 0.0

        # Prevent displayed micro-oscillations. The state itself remains free
        # to evolve, so a genuine turn can be released once confirmed.
        step_limit = 2.6 * noise_scale + 0.70 * abs(velocity)
        step = float(np.clip(delta, -step_limit, step_limit))

        if displayed_direction > 0:
            step = max(step, 0.0)
        elif displayed_direction < 0:
            step = min(step, 0.0)

        if abs(step) < 0.030 * noise_scale:
            step = 0.0

        displayed_level += step
        y[i] = displayed_level

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the requested filtering algorithm.

    Args:
        input_signal: Input time series.
        window_size: Sliding causal window size.
        algorithm_type: "basic" or "enhanced".

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