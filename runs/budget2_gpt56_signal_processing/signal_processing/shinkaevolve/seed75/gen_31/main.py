# EVOLVE-BLOCK-START
"""
Robust real-time adaptive filtering for volatile non-stationary signals.

The enhanced filter is a causal, adaptive constant-velocity Kalman tracker with:
    * robust difference-based noise initialization,
    * innovation-clipped adaptive process noise,
    * Joseph-form covariance updates with PSD safeguards,
    * adaptive robust measurement-noise tracking, and
    * output-only reversal persistence control.

Output remains aligned to the final sample of each trailing API window.
"""
import numpy as np


def _validate_signal(x, window_size):
    """Validate and convert public filter inputs."""
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


def _initial_noise_variance(signal, window_size):
    """
    Estimate observation variance from first differences.

    MAD is reliable under impulsive noise, while a restrained standard-deviation
    fallback avoids a zero initial covariance for short or quantized signals.
    """
    count = min(signal.size, max(5, min(window_size, 32)))
    differences = np.diff(signal[:count])

    if differences.size == 0:
        return 1e-10

    center = np.median(differences)
    mad = np.median(np.abs(differences - center))
    mad_std = mad / (0.67448975 * np.sqrt(2.0))
    std_fallback = np.std(differences) / np.sqrt(2.0)

    scale = max(float(mad_std), 0.35 * float(std_fallback), 1e-5)
    return scale * scale


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
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal robust adaptive level-and-slope filter.

    The internal state is updated for every measurement even if output reversal
    hysteresis temporarily holds the reported value. This separation avoids
    treating a brief display hold as evidence that the physical trend stopped.

    Args:
        x: Input signal (1D array of finite real-valued samples)
        window_size: Trailing alignment window

    Returns:
        Filtered output with length len(x) - window_size + 1.
    """
    signal, window_size = _validate_signal(x, window_size)
    n_samples = signal.size

    # State x = [level, velocity].  Covariance is retained explicitly rather
    # than allocating small matrices for every streaming sample.
    measurement_var = _initial_noise_variance(signal, window_size)
    base_measurement_var = measurement_var

    level = float(signal[0])
    velocity = 0.0
    p00 = measurement_var
    p01 = 0.0
    p11 = max(0.10 * measurement_var, 1e-10)

    filtered = np.empty(n_samples, dtype=float)
    filtered[0] = level

    output_direction = 0
    pending_direction = 0
    pending_count = 0
    pending_displacement = 0.0

    for index in range(1, n_samples):
        # Constant-velocity prediction: F P F' before acceleration process Q.
        predicted_level = level + velocity
        predicted_velocity = velocity
        pp00 = p00 + 2.0 * p01 + p11
        pp01 = p01 + p11
        pp11 = p11

        innovation = float(signal[index] - predicted_level)
        innovation_std = np.sqrt(max(pp00 + measurement_var, 1e-12))
        normalized = abs(innovation) / innovation_std

        # A bounded residual rejects isolated gross outliers.  The limit remains
        # generous enough for a sustained turn to be learned on following data.
        clipped_innovation = float(
            np.clip(innovation, -3.75 * innovation_std, 3.75 * innovation_std)
        )

        # Process noise represents local acceleration. It rises smoothly for
        # surprising observations to reduce lag, but is capped for stability.
        transition_strength = max(0.0, normalized - 1.15)
        acceleration_var = measurement_var * (
            0.0022 + 0.0180 * min(6.0, transition_strength * transition_strength)
        )

        # Q for unit sample spacing under white acceleration.
        pp00 += 0.25 * acceleration_var
        pp01 += 0.50 * acceleration_var
        pp11 += acceleration_var

        innovation_var = max(pp00 + measurement_var, 1e-12)
        k0 = pp00 / innovation_var
        k1 = pp01 / innovation_var

        level = predicted_level + k0 * clipped_innovation
        velocity = predicted_velocity + k1 * clipped_innovation

        # Joseph covariance update, expanded for H=[1, 0].
        a00 = 1.0 - k0
        a10 = -k1
        new_p00 = (
            a00 * a00 * pp00
            + 2.0 * a00 * 0.0 * pp01
            + k0 * k0 * measurement_var
        )
        new_p01 = a00 * (a10 * pp00 + pp01) + k0 * k1 * measurement_var
        new_p11 = (
            a10 * a10 * pp00
            + 2.0 * a10 * pp01
            + pp11
            + k1 * k1 * measurement_var
        )

        p00 = max(float(new_p00), 1e-12)
        p11 = max(float(new_p11), 1e-12)
        correlation_limit = np.sqrt(p00 * p11) * (1.0 - 1e-12)
        p01 = float(np.clip(new_p01, -correlation_limit, correlation_limit))

        # Robust residual variance adaptation. Capping prevents a single spike
        # from making subsequent estimates unnecessarily sluggish.
        residual_var = min(
            clipped_innovation * clipped_innovation,
            8.0 * max(measurement_var, base_measurement_var * 0.15),
        )
        measurement_var = max(
            0.975 * measurement_var + 0.025 * residual_var,
            base_measurement_var * 0.04,
            1e-12,
        )

        noise_scale = np.sqrt(measurement_var)

        # A modest bounded lead offsets causal state-estimation delay while
        # avoiding overshoot around extrema.
        lead = float(np.clip(0.28 * velocity, -0.70 * noise_scale, 0.70 * noise_scale))
        candidate = level + lead

        previous = filtered[index - 1]
        step = candidate - previous
        tiny_step = max(0.055 * noise_scale, 1e-8)

        if abs(step) <= tiny_step:
            # Attenuate imperceptible jitter without introducing staircase jumps.
            output = previous + 0.18 * step
        else:
            move_direction = 1 if step > 0.0 else -1

            if output_direction == 0 or move_direction == output_direction:
                output = candidate
                output_direction = move_direction
                pending_direction = 0
                pending_count = 0
                pending_displacement = 0.0
            else:
                # Large coherent moves pass immediately. Smaller counter-motion
                # needs two confirmations and enough cumulative displacement.
                turn_margin = max(0.72 * noise_scale, 0.32 * abs(velocity))
                immediate_turn = abs(step) >= 1.85 * turn_margin

                if move_direction != pending_direction:
                    pending_direction = move_direction
                    pending_count = 1
                    pending_displacement = abs(step)
                else:
                    pending_count += 1
                    pending_displacement += abs(step)

                confirmed_turn = (
                    pending_count >= 2
                    and pending_displacement >= max(0.80 * noise_scale, turn_margin)
                )

                if immediate_turn or confirmed_turn:
                    output = candidate
                    output_direction = move_direction
                    pending_direction = 0
                    pending_count = 0
                    pending_displacement = 0.0
                else:
                    # Hold most, rather than all, of a rejected move. This avoids
                    # excessive lag if the next sample confirms the reversal.
                    output = previous + 0.12 * step

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