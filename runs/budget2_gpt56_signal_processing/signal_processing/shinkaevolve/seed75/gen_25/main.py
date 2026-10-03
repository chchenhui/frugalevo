# EVOLVE-BLOCK-START
"""
Low-latency adaptive level-and-slope Kalman filtering for non-stationary signals.

The enhanced filter uses a causal two-state model containing signal level and
local slope. Process noise is bounded and innovation-adaptive, providing strong
smoothing during stable periods while increasing responsiveness at genuine
changes. A small display-level hysteresis controller reduces noise-induced
direction reversals without changing the internal Kalman predictor.
"""
import numpy as np


def _validate_signal(x, window_size):
    """Validate the common public signal-filter input contract."""
    signal = np.asarray(x, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array")

    if int(window_size) != window_size or window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    window_size = int(window_size)
    if signal.size < window_size:
        raise ValueError(
            f"Input signal length ({signal.size}) must be >= "
            f"window_size ({window_size})"
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("Input signal must contain only finite values")

    return signal, window_size


def adaptive_filter(x, window_size=20):
    """
    Basic trailing moving-average filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        Filtered output signal with length len(x) - window_size + 1.
    """
    signal, window_size = _validate_signal(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def _initial_measurement_variance(signal, window_size):
    """
    Estimate observation variance from first differences.

    Differencing removes local level and slow trend. MAD supplies robustness,
    while a limited standard-deviation fallback avoids unrealistically small
    variance estimates on short or quantized sequences.
    """
    initial = signal[:window_size]
    differences = np.diff(initial)

    if differences.size == 0:
        return 1e-12

    center = np.median(differences)
    mad = np.median(np.abs(differences - center))
    robust_std = mad / (0.6745 * np.sqrt(2.0))
    fallback_std = np.std(differences) / np.sqrt(2.0)

    noise_std = max(robust_std, 0.35 * fallback_std, 1e-6)
    return float(noise_std * noise_std)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal adaptive level-and-slope Kalman trend filter.

    The state transition is::

        level[t] = level[t - 1] + slope[t - 1]
        slope[t] = slope[t - 1] + acceleration_noise

    A bounded innovation-energy process-noise schedule permits rapid tracking
    of real turns but prevents isolated spikes from making the tracker noisy.
    Returned samples remain endpoint-aligned with input indices
    ``window_size - 1`` onward.

    Args:
        x: Input one-dimensional finite signal.
        window_size: Output alignment and initial-noise-estimation horizon.

    Returns:
        Filtered output with length len(x) - window_size + 1.
    """
    signal, window_size = _validate_signal(x, window_size)
    count = signal.size

    measurement_var = _initial_measurement_variance(signal, window_size)

    # State: level and one-sample level increment.
    level = float(signal[0])
    slope = 0.0

    # Symmetric covariance stored as independent elements.
    p00 = measurement_var
    p01 = 0.0
    p11 = max(0.10 * measurement_var, 1e-12)

    filtered = np.empty(count, dtype=float)
    filtered[0] = level

    # Output-only reversal state. It deliberately does not feed back into the
    # Kalman state, preserving predictive tracking after a rejected fluctuation.
    direction = 0
    pending_direction = 0
    pending_step = 0.0

    for index in range(1, count):
        # Constant-velocity prediction.
        predicted_level = level + slope
        predicted_slope = slope

        base_p00 = p00 + 2.0 * p01 + p11
        base_p01 = p01 + p11
        base_p11 = p11

        raw_innovation = signal[index] - predicted_level
        preliminary_variance = max(base_p00 + measurement_var, 1e-12)
        preliminary_scale = np.sqrt(preliminary_variance)

        # Robustly limit isolated observation spikes before adaptation.
        bounded_innovation = float(
            np.clip(
                raw_innovation,
                -3.5 * preliminary_scale,
                3.5 * preliminary_scale,
            )
        )

        # Conservative bounded acceleration schedule. The baseline preserves
        # smoothness, the innovation term reduces causal lag at true dynamics,
        # and the cap prevents an individual outlier from destabilizing slope.
        acceleration_var = np.clip(
            0.0020 * measurement_var + 0.0100 * bounded_innovation**2,
            0.0010 * measurement_var,
            0.2500 * measurement_var,
        )

        # Discrete white-acceleration covariance for unit sample interval.
        predicted_p00 = base_p00 + 0.25 * acceleration_var
        predicted_p01 = base_p01 + 0.50 * acceleration_var
        predicted_p11 = base_p11 + acceleration_var

        innovation_variance = max(predicted_p00 + measurement_var, 1e-12)
        gain_level = predicted_p00 / innovation_variance
        gain_slope = predicted_p01 / innovation_variance

        level = predicted_level + gain_level * bounded_innovation
        slope = predicted_slope + gain_slope * bounded_innovation

        # Joseph-equivalent scalar measurement covariance update.
        p00 = max(
            predicted_p00 - gain_level * predicted_p00,
            1e-12,
        )
        p01 = predicted_p01 - gain_level * predicted_p01
        p11 = max(
            predicted_p11 - gain_slope * predicted_p01,
            1e-12,
        )

        # Keep round-off from producing an invalid covariance correlation.
        covariance_limit = np.sqrt(p00 * p11) * (1.0 - 1e-12)
        p01 = float(np.clip(p01, -covariance_limit, covariance_limit))

        # Slow robust variance adaptation follows non-stationary noise but caps
        # transient innovation energy so genuine turning points do not cause
        # permanent loss of responsiveness.
        residual_energy = min(
            bounded_innovation * bounded_innovation,
            9.0 * measurement_var,
        )
        measurement_var = max(
            0.985 * measurement_var + 0.015 * residual_energy,
            1e-12,
        )

        candidate = level
        previous = filtered[index - 1]
        step = candidate - previous
        noise_scale = np.sqrt(measurement_var)

        # Soft deadband reduces tiny output jitter without creating a flat,
        # quantized trajectory.
        deadband = max(0.11 * noise_scale, 1e-9)

        if abs(step) <= deadband:
            output = previous + 0.24 * step
        else:
            move_direction = 1 if step > 0.0 else -1

            if direction == 0 or move_direction == direction:
                output = candidate
                direction = move_direction
                pending_direction = 0
                pending_step = 0.0
            else:
                # Large counter-trend displacement passes immediately. Weak
                # counter-motion requires one consistent following sample.
                immediate_turn = abs(step) >= max(
                    0.90 * noise_scale,
                    1.65 * abs(filtered[index - 1] - filtered[index - 2])
                    if index >= 2 else 0.0,
                )

                if immediate_turn:
                    output = candidate
                    direction = move_direction
                    pending_direction = 0
                    pending_step = 0.0
                elif pending_direction == move_direction:
                    # Release retained displacement when the reversal persists.
                    output = previous + step + pending_step
                    direction = move_direction
                    pending_direction = 0
                    pending_step = 0.0
                else:
                    pending_direction = move_direction
                    pending_step = step
                    output = previous + 0.08 * step

        filtered[index] = output

    return filtered[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing dispatcher.

    Args:
        input_signal: Input time series data
        window_size: Size of the processing window
        algorithm_type: "basic" or "enhanced"

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