# EVOLVE-BLOCK-START
"""
Adaptive innovation-gated Kalman filtering for volatile, non-stationary signals.

The public API is intentionally compatible with the original implementation:
    adaptive_filter(x, window_size=20)
    enhanced_filter_with_trend_preservation(x, window_size=20)
    process_signal(input_signal, window_size=20, algorithm_type="enhanced")
"""
import numpy as np


def _validate_signal(x, window_size):
    """Validate input while preserving the original error contract."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")

    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite real-valued samples")

    return x


def adaptive_filter(x, window_size=20):
    """
    Robust causal local-level filter.

    This is the conservative mode used for algorithm_type="basic". It is a
    robust exponentially adaptive smoother with innovation clipping.

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1.
           Output sample i corresponds to input sample i + window_size - 1.
    """
    x = _validate_signal(x, window_size)
    n = len(x)

    if n == 1:
        return x.copy()

    initial = x[:window_size]
    diff = np.diff(initial)

    # Robust scale estimate from adjacent differences.
    mad_diff = np.median(np.abs(diff - np.median(diff))) if len(diff) else 0.0
    noise_std = max(mad_diff / 0.67448975 / np.sqrt(2.0), 1e-6)

    y_full = np.empty(n, dtype=float)
    level = x[0]
    residual_scale = noise_std

    for t in range(n):
        observation = x[t]
        innovation = observation - level

        # Isolated impulses should not force a false directional reversal.
        clip_limit = max(3.0 * residual_scale, 1e-8)
        bounded_innovation = np.clip(innovation, -clip_limit, clip_limit)

        # Increase gain for sustained movement while keeping stable regions smooth.
        activity = min(abs(innovation) / (4.0 * residual_scale + 1e-12), 1.0)
        gain = 0.08 + 0.42 * activity

        level += gain * bounded_innovation
        y_full[t] = level

        # Robustly update the residual/noise scale.
        residual_scale = 0.97 * residual_scale + 0.03 * min(
            abs(innovation), 4.0 * residual_scale + 1e-8
        )
        residual_scale = max(residual_scale, noise_std * 0.25, 1e-8)

    return y_full[window_size - 1 :]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Innovation-gated adaptive Kalman filter with velocity-state trend tracking.

    A constant-velocity state model is updated causally for every observation:
        state = [signal level, local slope]

    Key mechanisms:
    - Robust innovation clipping rejects impulsive measurement noise.
    - Measurement variance adapts to local residual noise.
    - Process variance rises during coherent innovations, reducing lag at real
      turns and trend changes.
    - The velocity state prevents moving-average-style trend attenuation.

    Returns:
        y: Filtered output with length len(x) - window_size + 1.
    """
    x = _validate_signal(x, window_size)
    n = len(x)

    if n == 1:
        return x.copy()

    warmup = x[:window_size]
    differences = np.diff(warmup)

    # Robust initial observation noise estimate. Adjacent differencing makes
    # this much less sensitive to a slowly varying baseline.
    if len(differences):
        median_diff = np.median(differences)
        mad_diff = np.median(np.abs(differences - median_diff))
        measurement_std = mad_diff / (0.67448975 * np.sqrt(2.0))
    else:
        measurement_std = 0.0

    signal_scale = np.median(np.abs(warmup - np.median(warmup))) / 0.67448975
    measurement_std = max(measurement_std, 0.03 * signal_scale, 1e-5)
    measurement_var = measurement_std * measurement_std

    # State [position, velocity].  Velocity initialization from a robust
    # warm-up slope avoids a zero-slope startup bias.
    initial_slope = np.median(differences) if len(differences) else 0.0
    state = np.array([x[0], initial_slope], dtype=float)

    # Covariance state. Position uncertainty begins at measurement scale;
    # slope uncertainty is deliberately larger to permit early adaptation.
    covariance = np.array(
        [
            [measurement_var, 0.0],
            [0.0, max(measurement_var * 0.5, 1e-7)],
        ],
        dtype=float,
    )

    output_full = np.empty(n, dtype=float)
    residual_scale = measurement_std
    process_var = max(measurement_var * 0.03, 1e-8)
    previous_innovation_sign = 0.0
    coherent_count = 0

    for t in range(n):
        # Constant-velocity prediction:
        # [position_t] = [1 1] [position_(t-1)]
        # [velocity_t]   [0 1] [velocity_(t-1)]
        predicted_position = state[0] + state[1]
        predicted_velocity = state[1]

        # Adaptive process covariance for random acceleration.
        q = process_var
        predicted_covariance = np.array(
            [
                [
                    covariance[0, 0] + 2.0 * covariance[0, 1] + covariance[1, 1] + 0.25 * q,
                    covariance[0, 1] + covariance[1, 1] + 0.5 * q,
                ],
                [
                    covariance[0, 1] + covariance[1, 1] + 0.5 * q,
                    covariance[1, 1] + q,
                ],
            ],
            dtype=float,
        )

        innovation = x[t] - predicted_position
        innovation_sign = np.sign(innovation)

        # Consecutive same-direction innovations are evidence of a real trend
        # change. Alternating signs are more characteristic of measurement noise.
        if innovation_sign != 0.0 and innovation_sign == previous_innovation_sign:
            coherent_count = min(coherent_count + 1, 8)
        else:
            coherent_count = max(coherent_count - 1, 0)

        if innovation_sign != 0.0:
            previous_innovation_sign = innovation_sign

        # Robust gate: large isolated impulses update level only weakly.
        gate = max(3.5 * residual_scale, measurement_std * 2.0, 1e-8)
        bounded_innovation = np.clip(innovation, -gate, gate)

        # During coherent movement, increase process uncertainty so the filter
        # follows true acceleration instead of accumulating lag.
        normalized_activity = min(abs(innovation) / (gate + 1e-12), 1.0)
        coherence = coherent_count / 8.0
        target_process_var = measurement_var * (
            0.02 + 0.18 * normalized_activity + 1.20 * coherence * normalized_activity
        )
        process_var = 0.90 * process_var + 0.10 * max(target_process_var, 1e-9)

        # Adaptive measurement uncertainty rises modestly in noisy intervals,
        # but clipping prevents genuine jumps from being misclassified as noise.
        robust_abs_residual = min(abs(innovation), 2.5 * gate)
        residual_scale = 0.975 * residual_scale + 0.025 * robust_abs_residual
        residual_scale = max(residual_scale, measurement_std * 0.35, 1e-8)
        adaptive_measurement_var = max(residual_scale * residual_scale, measurement_var * 0.20)

        innovation_variance = predicted_covariance[0, 0] + adaptive_measurement_var
        gain_position = predicted_covariance[0, 0] / innovation_variance
        gain_velocity = predicted_covariance[1, 0] / innovation_variance

        # A gated innovation is used for the update, preserving resistance to
        # noise-induced reversals and one-sample spikes.
        state[0] = predicted_position + gain_position * bounded_innovation
        state[1] = predicted_velocity + gain_velocity * bounded_innovation

        # Joseph-form equivalent for scalar measurement, simplified for speed.
        p00 = (1.0 - gain_position) * predicted_covariance[0, 0]
        p01 = (1.0 - gain_position) * predicted_covariance[0, 1]
        p11 = predicted_covariance[1, 1] - gain_velocity * predicted_covariance[0, 1]

        covariance[0, 0] = max(p00, 1e-12)
        covariance[0, 1] = p01
        covariance[1, 0] = p01
        covariance[1, 1] = max(p11, 1e-12)

        output_full[t] = state[0]

    # Maintain original API alignment: each output represents the newest point
    # of its corresponding trailing input window.
    return output_full[window_size - 1 :]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected compatible filtering mode.

    Args:
        input_signal: Input 1D time series.
        window_size: Trailing-window alignment size.
        algorithm_type: "enhanced" selects adaptive Kalman trend tracking;
            all other values select the conservative robust filter.

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
