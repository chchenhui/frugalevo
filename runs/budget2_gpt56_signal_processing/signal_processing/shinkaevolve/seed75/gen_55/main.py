# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

The enhanced path is a causal, robust constant-velocity Kalman trend estimator.
It uses adaptive innovation gating and a separate persistent reversal gate to
reduce noise-driven turns without adding moving-average phase delay.
"""
import numpy as np


def _validate_signal(x, window_size):
    """Convert and validate a one-dimensional finite real-valued signal."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array of real-valued samples")
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite values")

    return x


def adaptive_filter(x, window_size=20):
    """
    Efficient trailing moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        Moving-average output with length len(x) - window_size + 1.
    """
    x = _validate_signal(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal adaptive level-and-trend Kalman filter with reversal persistence.

    A robust linear fit initializes the newest startup-window endpoint. The
    estimator then uses a scalar constant-velocity covariance update, clipped
    innovations, and bounded acceleration variance. Output reversal handling is
    deliberately decoupled from the state estimator: established trends have no
    extra confirmation delay, whereas weak opposing movements require both
    persistence and sufficient accumulated displacement.
    """
    x = _validate_signal(x, window_size)

    n = len(x)
    output_length = n - window_size + 1
    y = np.empty(output_length, dtype=float)

    if window_size == 1:
        y[0] = x[0]
        if n > 1:
            y[1:] = x[1:]
        return y

    # Endpoint least-squares initialization reduces startup lag compared with
    # beginning the recursive estimator at the first raw sample.
    t = np.arange(window_size, dtype=float)
    slope, intercept = np.polyfit(t, x[:window_size], 1)
    level = float(intercept + slope * (window_size - 1))

    residuals = x[:window_size] - (intercept + slope * t)
    residual_mad = np.median(np.abs(residuals - np.median(residuals)))
    residual_scale = 1.4826 * residual_mad

    startup_diff = np.diff(x[:window_size])
    diff_mad = np.median(np.abs(startup_diff - np.median(startup_diff)))
    diff_scale = diff_mad / (0.6745 * np.sqrt(2.0))

    noise_scale = max(residual_scale, 0.50 * diff_scale, np.std(residuals) * 0.20, 1e-6)
    measurement_var = noise_scale * noise_scale

    # Scalar covariance representation for state [level, slope].
    p00 = 4.0 * measurement_var
    p01 = measurement_var
    p11 = 0.50 * measurement_var

    y[0] = level
    emitted_level = level
    emitted_direction = 1 if slope > 0.0 else (-1 if slope < 0.0 else 0)
    pending_direction = 0
    pending_count = 0
    pending_distance = 0.0

    previous_increment = slope
    persistent_innovation = 0.0
    normalized_innovation_scale = 1.0

    for output_index, sample_index in enumerate(range(window_size, n), start=1):
        predicted_level = level + slope
        predicted_slope = slope

        raw_innovation = x[sample_index] - predicted_level
        predicted_measurement_var = max(p00 + measurement_var, 1e-12)
        normalized_raw = raw_innovation / np.sqrt(predicted_measurement_var)

        gate_sigma = np.clip(2.55 + 0.55 * normalized_innovation_scale, 2.8, 3.8)
        clip_limit = gate_sigma * np.sqrt(predicted_measurement_var)
        innovation = np.clip(raw_innovation, -clip_limit, clip_limit)

        # Require innovation persistence before substantially increasing process
        # noise. This avoids making the state chase isolated impulses.
        persistent_innovation = 0.76 * persistent_innovation + 0.24 * innovation
        process_var = min(
            0.0015 * measurement_var + 0.018 * persistent_innovation * persistent_innovation,
            0.26 * measurement_var,
        )

        # F=[[1,1],[0,1]], Q=q*[[1/4,1/2],[1/2,1]].
        predicted_p00 = p00 + 2.0 * p01 + p11 + 0.25 * process_var
        predicted_p01 = p01 + p11 + 0.50 * process_var
        predicted_p11 = p11 + process_var

        residual_var = max(predicted_p00 + measurement_var, 1e-12)
        gain_level = predicted_p00 / residual_var
        gain_slope = predicted_p01 / residual_var

        candidate_level = predicted_level + gain_level * innovation
        candidate_slope = predicted_slope + gain_slope * innovation

        # Joseph-equivalent scalar covariance form with positivity floors.
        p00 = max((1.0 - gain_level) * predicted_p00, 1e-12)
        p01 = (1.0 - gain_level) * predicted_p01
        p11 = max(predicted_p11 - gain_slope * predicted_p01, 1e-12)

        increment = candidate_level - level
        small_reverse_threshold = 0.16 * noise_scale
        if previous_increment * increment < 0.0 and abs(increment) < small_reverse_threshold:
            candidate_level = level
            candidate_slope *= 0.40
            increment = 0.0

        level = candidate_level
        slope = candidate_slope
        previous_increment = 0.78 * previous_increment + 0.22 * increment

        # Compensate part of the remaining causal smoothing lag only while the
        # instantaneous and slow trend estimates agree.  Prediction is reduced
        # as the fast slope decelerates and is removed when persistent
        # innovations oppose the current trend, which prevents extrapolation
        # beyond an approaching corner.
        consensus_slope = 0.65 * slope + 0.35 * previous_increment
        same_direction = slope * previous_increment > 0.0
        slope_ratio = abs(slope) / max(abs(previous_increment), 1e-12)
        deceleration_weight = np.clip(slope_ratio, 0.0, 1.0)
        opposing_evidence = max(
            0.0,
            -persistent_innovation * consensus_slope
            / max(noise_scale * abs(consensus_slope), 1e-12),
        )
        turn_weight = np.clip(1.0 - opposing_evidence, 0.0, 1.0)
        lead_weight = deceleration_weight * turn_weight if same_direction else 0.0
        predictive_lead = np.clip(
            0.30 * lead_weight * consensus_slope,
            -0.55 * noise_scale,
            0.55 * noise_scale,
        )
        published_level = level + predictive_lead

        # Publish only trend-significant output motion. Strong turns are
        # immediate; smaller turns require both repeated direction and distance.
        output_move = published_level - emitted_level
        output_deadband = 0.055 * noise_scale
        if output_move > output_deadband:
            output_direction = 1
        elif output_move < -output_deadband:
            output_direction = -1
        else:
            output_direction = 0

        reversal_distance = 0.62 * noise_scale
        strong_turn = 1.30 * noise_scale

        if output_direction == 0:
            pending_direction = 0
            pending_count = 0
            pending_distance = 0.0
        elif emitted_direction == 0 or output_direction == emitted_direction:
            emitted_level = published_level
            emitted_direction = output_direction
            pending_direction = 0
            pending_count = 0
            pending_distance = 0.0
        else:
            if output_direction != pending_direction:
                pending_direction = output_direction
                pending_count = 1
                pending_distance = abs(output_move)
            else:
                pending_count += 1
                pending_distance += abs(output_move)

            if (
                abs(output_move) >= strong_turn
                or (pending_count >= 2 and pending_distance >= reversal_distance)
            ):
                emitted_level = published_level
                emitted_direction = output_direction
                pending_direction = 0
                pending_count = 0
                pending_distance = 0.0

        y[output_index] = emitted_level

        # Slowly adapt measurement uncertainty using bounded innovations. This
        # tracks non-stationary volatility while preventing impulses from
        # permanently increasing gain or reversal thresholds.
        normalized_innovation_scale = np.clip(
            0.95 * normalized_innovation_scale
            + 0.05 * min(abs(normalized_raw), gate_sigma),
            0.5,
            2.2,
        )
        updated_scale = max(
            0.93 * noise_scale + 0.07 * min(abs(raw_innovation), clip_limit),
            1e-6,
        )
        noise_scale = updated_scale
        measurement_var = noise_scale * noise_scale

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