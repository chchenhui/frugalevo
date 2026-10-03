# EVOLVE-BLOCK-START
"""
Bounded adaptive level-and-slope Kalman filtering for volatile,
non-stationary causal time series.

The enhanced filter uses a two-state local-linear-trend model:
    state = [signal level, per-sample slope]

Its acceleration process noise is continuously adapted from the innovation,
with conservative lower and upper bounds. This provides prompt tracking of
real turns while preventing impulsive noise from causing covariance inflation
or repeated false reversals.
"""
import numpy as np


def _validate_signal(x, window_size):
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

    Returns an array with length len(x) - window_size + 1.
    """
    signal, window_size = _validate_signal(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    return (
        cumulative[window_size:] - cumulative[:-window_size]
    ) / float(window_size)


def _initial_noise_variance(signal, window_size):
    """
    Robust observation-noise variance from first differences.

    The difference-domain MAD avoids interpreting a slowly changing level as
    measurement noise. A standard-deviation fallback handles nearly constant
    signals and very short windows.
    """
    initial = signal[:window_size]
    differences = np.diff(initial)

    if differences.size == 0:
        return max((abs(initial[0]) * 1e-4) ** 2, 1e-12)

    center = np.median(differences)
    mad = np.median(np.abs(differences - center))
    mad_std = mad / (0.6745 * np.sqrt(2.0))
    std_fallback = np.std(differences) / np.sqrt(2.0)

    noise_std = max(mad_std, 0.40 * std_fallback, 1e-6)
    return float(noise_std * noise_std)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal bounded-adaptive level/slope Kalman trend filter.

    Key adaptive parameters:
        process baseline:       0.002 * measurement_variance
        innovation coefficient: 0.010 * innovation^2
        process-noise floor:    0.001 * measurement_variance
        process-noise cap:      0.250 * measurement_variance
        innovation clip:        4.0 predicted standard deviations
        noise learning rate:    0.02
        output deadband:        0.08 local noise standard deviations

    Output index j corresponds to input index j + window_size - 1.
    """
    signal, window_size = _validate_signal(x, window_size)
    n_samples = signal.size

    measurement_var = _initial_noise_variance(signal, window_size)

    # Local linear trend state: level and one-sample velocity.
    level = float(signal[0])
    slope = 0.0

    # Symmetric covariance entries for the 2x2 covariance matrix.
    p00 = measurement_var
    p01 = 0.0
    p11 = max(0.08 * measurement_var, 1e-12)

    filtered = np.empty(n_samples, dtype=float)
    filtered[0] = level

    # Output-direction state is continuous: it only suppresses insignificant
    # movements and never enforces a fixed turn-confirmation delay.
    output_direction = 0
    previous_output_step = 0.0
    slow_residual_var = measurement_var

    for index in range(1, n_samples):
        # Predict state under x[k] = x[k-1] + slope[k-1].
        predicted_level = level + slope
        predicted_slope = slope

        # P <- F P F' before acceleration process covariance is added.
        pp00 = p00 + 2.0 * p01 + p11
        pp01 = p01 + p11
        pp11 = p11

        raw_innovation = signal[index] - predicted_level
        predicted_measurement_var = max(pp00 + measurement_var, 1e-12)
        innovation_scale = np.sqrt(predicted_measurement_var)

        # Robustly limit isolated observations without discarding a persistent
        # transition: following samples retain the same innovation direction.
        innovation = float(
            np.clip(
                raw_innovation,
                -4.0 * innovation_scale,
                4.0 * innovation_scale,
            )
        )

        # Bounded innovation-energy process-noise scheduler. Unlike a hard
        # regime detector, this raises gain smoothly during actual dynamics.
        process_floor = 0.001 * measurement_var
        process_cap = 0.250 * measurement_var
        acceleration_var = np.clip(
            0.002 * measurement_var + 0.010 * innovation * innovation,
            process_floor,
            max(process_cap, process_floor),
        )

        # Discrete white-acceleration covariance for one sample interval.
        pp00 += 0.25 * acceleration_var
        pp01 += 0.50 * acceleration_var
        pp11 += acceleration_var

        innovation_var = max(pp00 + measurement_var, 1e-12)
        gain_level = pp00 / innovation_var
        gain_slope = pp01 / innovation_var

        level = predicted_level + gain_level * innovation
        slope = predicted_slope + gain_slope * innovation

        # Stable scalar Joseph-equivalent covariance update for H=[1, 0].
        p00 = max(pp00 - gain_level * pp00, 1e-12)
        p01 = pp01 - gain_level * pp01
        p11 = max(pp11 - gain_slope * pp01, 1e-12)

        # Keep the covariance safely positive semidefinite after rounding.
        covariance_limit = np.sqrt(p00 * p11) * (1.0 - 1e-12)
        p01 = float(np.clip(p01, -covariance_limit, covariance_limit))

        # Robust residual-energy adaptation. A clipped update tracks changing
        # volatility without allowing a single large transition to make the
        # filter permanently insensitive.
        residual_energy = min(
            innovation * innovation,
            7.0 * measurement_var,
        )
        slow_residual_var = (
            0.98 * slow_residual_var + 0.02 * residual_energy
        )
        measurement_var = max(
            0.92 * measurement_var + 0.08 * slow_residual_var,
            1e-12,
        )

        candidate = level
        previous = filtered[index - 1]
        raw_step = candidate - previous
        noise_scale = np.sqrt(measurement_var)

        # Small motions are shrunk continuously. The threshold is deliberately
        # below one tenth of the local noise scale to preserve responsiveness.
        deadband = max(0.08 * noise_scale, 1e-10)
        if abs(raw_step) <= deadband:
            output = previous + 0.28 * raw_step
        else:
            direction = 1 if raw_step > 0.0 else -1

            # Slightly stronger shrinkage for a very small opposing step
            # suppresses flicker near extrema, without a delayed reversal gate.
            if (
                output_direction != 0
                and direction != output_direction
                and abs(raw_step) < 0.32 * noise_scale + 0.18 * abs(previous_output_step)
            ):
                output = previous + 0.55 * raw_step
            else:
                output = candidate
                output_direction = direction

        previous_output_step = output - previous
        filtered[index] = output

    return filtered[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the requested filtering algorithm.

    Args:
        input_signal: One-dimensional input signal.
        window_size: Sliding trailing alignment window.
        algorithm_type: "basic" or "enhanced".

    Returns:
        Filtered output with length len(input_signal) - window_size + 1.
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(
            input_signal,
            window_size,
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