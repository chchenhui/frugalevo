# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

The enhanced filter uses a robust adaptive constant-velocity Kalman estimator.
It estimates both level and local slope, providing low-lag trend tracking while
adaptively suppressing noise-induced directional reversals.
"""
import numpy as np


def _validate_signal(x, window_size):
    """Convert and validate a one-dimensional input signal."""
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array of real-valued samples")

    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer")

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
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = _validate_signal(x, window_size)

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust adaptive Kalman trend filter.

    The estimator uses a two-state model:
        state = [signal_level, local_slope]

    A predicted level reduces phase delay, while adaptive noise estimation and
    robust innovation clipping reduce noise spikes and false reversals. Samples
    are returned at the same trailing-window alignment as the original filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = _validate_signal(x, window_size)
    n = len(x)

    if n == 1:
        return x.copy()

    # Estimate an initial noise level from first differences.  The median
    # estimator is resistant to local jumps and transient outliers.
    initial_span = min(n, max(4, window_size))
    initial_diff = np.diff(x[:initial_span])
    median_diff = np.median(initial_diff)
    mad_diff = np.median(np.abs(initial_diff - median_diff))
    noise_std = max(1e-6, 1.4826 * mad_diff / np.sqrt(2.0))

    # State: filtered level and its one-sample slope.
    level = float(x[0])
    slope = float(x[1] - x[0])

    # Covariance of [level, slope].
    p00 = max(noise_std * noise_std, 1e-6)
    p01 = 0.0
    p11 = max(0.25 * noise_std * noise_std, 1e-6)

    # Innovation variance is tracked slowly so the filter remains stable in
    # noisy regions while adapting to changes in signal volatility.
    measurement_var = max(noise_std * noise_std, 1e-6)
    innovation_var = measurement_var

    filtered = np.empty(n, dtype=float)
    filtered[0] = level

    # Noise time constants are expressed as EWMA gains.  The window length is
    # used as a practical adaptation horizon without introducing window delay.
    variance_rate = 2.0 / (max(window_size, 4) + 1.0)
    process_rate = 1.0 / (max(window_size, 6) + 1.0)

    # A small base acceleration variance preserves smoothness. Persistent,
    # coherent innovations temporarily increase it to retain genuine dynamics.
    base_q = max(0.0025 * measurement_var, 1e-8)
    adaptive_q = base_q
    previous_innovation_sign = 0.0
    persistence = 0.0

    for i in range(1, n):
        # Constant-velocity prediction.
        predicted_level = level + slope
        predicted_slope = slope

        # P = F P F' + Q, with F = [[1, 1], [0, 1]].
        q = adaptive_q
        pred_p00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pred_p01 = p01 + p11 + 0.5 * q
        pred_p11 = p11 + q

        innovation = x[i] - predicted_level
        predicted_measurement_var = max(pred_p00 + measurement_var, 1e-12)
        innovation_scale = np.sqrt(predicted_measurement_var)

        # Huber clipping prevents a single noise spike from making the slope
        # reverse. A generous limit avoids suppressing sustained real changes.
        clip_limit = 3.5 * innovation_scale
        robust_innovation = np.clip(innovation, -clip_limit, clip_limit)

        # Persistent innovations with the same sign usually indicate genuine
        # acceleration/trend change; alternating innovations are more likely
        # noise and therefore receive less process-noise increase.
        innovation_sign = np.sign(robust_innovation)
        if innovation_sign != 0.0 and innovation_sign == previous_innovation_sign:
            persistence = min(1.0, persistence + 0.20)
        else:
            persistence *= 0.55
        if innovation_sign != 0.0:
            previous_innovation_sign = innovation_sign

        # Robust residual power updates the local measurement-noise estimate.
        residual_power = robust_innovation * robust_innovation
        innovation_var = (
            (1.0 - variance_rate) * innovation_var + variance_rate * residual_power
        )
        measurement_var = max(0.15 * noise_std * noise_std, 0.55 * innovation_var)

        # Adapt process noise only after coherent movement is observed. This
        # balances responsiveness against false-reversal suppression.
        motion_power = max(0.0, residual_power - measurement_var)
        target_q = base_q + persistence * 0.18 * motion_power
        adaptive_q = (
            (1.0 - process_rate) * adaptive_q + process_rate * max(target_q, base_q)
        )

        # Kalman measurement update. The level gain controls denoising and the
        # slope gain provides predictive lag compensation.
        s = max(pred_p00 + measurement_var, 1e-12)
        k0 = pred_p00 / s
        k1 = pred_p01 / s

        level = predicted_level + k0 * robust_innovation
        slope = predicted_slope + k1 * robust_innovation

        # Joseph-equivalent scalar covariance update, simplified for H=[1, 0].
        p00 = max((1.0 - k0) * pred_p00, 1e-12)
        p01 = (1.0 - k0) * pred_p01
        p11 = max(pred_p11 - k1 * pred_p01, 1e-12)

        filtered[i] = level

    # Match the original trailing-window output convention: output sample zero
    # corresponds to the final sample of the first window.
    return filtered[window_size - 1 :]


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