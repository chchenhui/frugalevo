# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal robust adaptive Kalman filter with local trend tracking.

    A two-state (level, slope) model predicts the next sample before each
    measurement update.  This provides substantially less trend lag than a
    moving average.  Innovation clipping and adaptive noise estimates suppress
    impulsive noise and noise-induced slope reversals.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Sliding-window length used for initialization and output
            alignment.

    Returns:
        Filtered signal aligned with x[window_size - 1:].
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    # Robust initial noise scale from first differences.  Differencing makes
    # this estimate insensitive to a slowly changing level.
    initial = x[:window_size]
    differences = np.diff(initial)
    diff_mad = np.median(np.abs(differences - np.median(differences)))
    measurement_var = max((diff_mad / 0.6745) ** 2 * 0.5, 1e-8)
    process_var = max(measurement_var * 0.08, 1e-10)

    # State is [level, slope]; initialize slope from a robust window trend.
    time_index = np.arange(window_size, dtype=float)
    slope = np.polyfit(time_index, initial, 1)[0] if window_size > 1 else 0.0
    state = np.array([initial[0], slope], dtype=float)
    covariance = np.array(
        [[measurement_var * 4.0, 0.0], [0.0, measurement_var]],
        dtype=float,
    )
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    output = np.empty(len(x) - window_size + 1, dtype=float)

    innovation_var = measurement_var
    for index, sample in enumerate(x):
        # Constant-velocity prediction.  The process covariance permits real
        # slope changes without requiring noisy measurements to define slope.
        state = transition @ state
        covariance = transition @ covariance @ transition.T
        covariance += process_var * np.array([[0.25, 0.5], [0.5, 1.0]])

        innovation = sample - state[0]
        predicted_var = covariance[0, 0] + measurement_var
        innovation_scale = np.sqrt(max(predicted_var, 1e-12))

        # Huber-style clipping limits the effect of isolated outliers.
        clipped_innovation = np.clip(innovation, -3.0 * innovation_scale, 3.0 * innovation_scale)
        gain = covariance[:, 0] / predicted_var
        state += gain * clipped_innovation
        covariance -= np.outer(gain, covariance[0, :])
        covariance = 0.5 * (covariance + covariance.T)

        # Adapt slowly: innovations beyond the clipping threshold do not cause
        # a transient spike to make the filter permanently less smooth.
        innovation_var = 0.96 * innovation_var + 0.04 * clipped_innovation ** 2
        measurement_var = max(0.98 * measurement_var + 0.02 * innovation_var, 1e-8)
        process_var = max(0.995 * process_var + 0.005 * innovation_var * 0.08, 1e-10)

        if index >= window_size - 1:
            output[index - window_size + 1] = state[0]

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