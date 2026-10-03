# EVOLVE-BLOCK-START
"""
Causal multiscale robust local-polynomial signal filtering.

The enhanced path uses two endpoint-aligned robust LOESS-style polynomial fits:
a short horizon for responsiveness and a longer horizon for trend stability.
A continuous slope soft-shrinkage output stage suppresses small noise-induced
direction reversals without imposing a discrete reversal confirmation delay.
"""
import numpy as np


def _validate_signal(x, window_size):
    signal = np.asarray(x, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array of real-valued samples")

    if int(window_size) != window_size or window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    window_size = int(window_size)
    if signal.size < window_size:
        raise ValueError(
            f"Input signal length ({signal.size}) must be >= "
            f"window_size ({window_size})"
        )

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
    return (
        cumulative[window_size:] - cumulative[:-window_size]
    ) / float(window_size)


def _robust_scale(values, fallback=1e-6):
    """MAD scale estimate with a stable nonzero floor."""
    if values.size == 0:
        return fallback

    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return max(1.4826 * mad, fallback)


def _endpoint_polynomial_fit(samples, decay):
    """
    Robust weighted quadratic endpoint regression.

    The final sample is located at time zero, so coefficient zero is the
    current causal estimate.  The linear coefficient is its local slope.
    """
    count = samples.size

    if count == 1:
        return float(samples[0]), 0.0, 1e-6

    # Normalize time for stable fitting.  Newest observation is always t = 0.
    t = np.linspace(-1.0, 0.0, count)
    if count == 2:
        design = np.column_stack((np.ones(count), t))
    else:
        design = np.column_stack((np.ones(count), t, t * t))

    base_weights = np.exp(decay * t)
    base_weights /= np.max(base_weights)

    # Initial weighted least-squares estimate.
    weighted_design = design * base_weights[:, None]
    normal_matrix = design.T @ weighted_design
    normal_matrix += np.eye(normal_matrix.shape[0]) * 1e-10
    coefficients = np.linalg.solve(normal_matrix, weighted_design.T @ samples)

    residuals = samples - design @ coefficients
    difference_scale = _robust_scale(np.diff(samples)) / np.sqrt(2.0)
    scale = max(_robust_scale(residuals), difference_scale * 0.35, 1e-6)

    # One Huber IRLS refinement rejects spikes while retaining sustained motion.
    normalized = np.abs(residuals) / scale
    huber_weights = np.minimum(1.0, 2.5 / np.maximum(normalized, 1e-12))
    weights = base_weights * huber_weights

    weighted_design = design * weights[:, None]
    normal_matrix = design.T @ weighted_design
    normal_matrix += np.eye(normal_matrix.shape[0]) * 1e-10
    coefficients = np.linalg.solve(normal_matrix, weighted_design.T @ samples)

    residuals = samples - design @ coefficients
    scale = max(_robust_scale(residuals), difference_scale * 0.30, 1e-6)

    # t spans approximately one window duration; convert derivative to samples.
    slope = coefficients[1] / max(count - 1, 1)
    return float(coefficients[0]), float(slope), float(scale)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal multiscale robust local-polynomial filter.

    A short robust endpoint fit tracks genuine rapid changes.  A longer fit
    rejects noise and anchors the trajectory.  Their adaptive blend is then
    softly slope-thresholded using the local residual scale, reducing false
    reversals while preserving large directional changes.

    Returns:
        Filtered output aligned to input samples window_size - 1 onward.
    """
    signal, window_size = _validate_signal(x, window_size)

    # Preserve streaming behavior for occasional invalid observations.
    stream = np.empty_like(signal)
    last_valid = 0.0
    for i, value in enumerate(signal):
        if np.isfinite(value):
            last_valid = value
        stream[i] = last_valid

    length = stream.size
    estimates = np.empty(length, dtype=float)

    # The short fit resolves local oscillations; the long fit provides a stable
    # low-frequency reference.  These horizons scale naturally with the API.
    short_window = max(5, int(round(window_size * 0.55)))
    short_window = min(short_window, window_size)

    estimates[0] = stream[0]
    previous_scale = max(abs(stream[0]) * 0.01, 1e-5)

    for index in range(1, length):
        short_start = max(0, index - short_window + 1)
        long_start = max(0, index - window_size + 1)

        short_value, short_slope, short_scale = _endpoint_polynomial_fit(
            stream[short_start:index + 1], decay=2.0
        )
        long_value, long_slope, long_scale = _endpoint_polynomial_fit(
            stream[long_start:index + 1], decay=1.25
        )

        local_scale = max(0.65 * short_scale + 0.35 * long_scale, 1e-6)

        # If the models agree, the long model can contribute more smoothing.
        # If they diverge due to local curvature or a regime change, prioritize
        # the responsive short-window estimate.
        disagreement = abs(short_value - long_value)
        agreement = disagreement / max(local_scale, 1e-6)
        short_weight = 0.62 + 0.23 * (agreement / (agreement + 2.0))

        candidate = (
            short_weight * short_value
            + (1.0 - short_weight) * long_value
        )

        # A modest polynomial-slope lead offsets regression attenuation but is
        # bounded by residual uncertainty to avoid overshoot around turns.
        blended_slope = (
            short_weight * short_slope
            + (1.0 - short_weight) * long_slope
        )
        lead_limit = 0.55 * local_scale
        candidate += np.clip(0.20 * blended_slope, -lead_limit, lead_limit)

        # Continuous soft shrinkage only suppresses motion smaller than the
        # measured local uncertainty. Unlike a reversal gate it never waits for
        # a fixed number of samples and therefore maintains low causal latency.
        raw_step = candidate - estimates[index - 1]
        threshold = 0.13 * max(local_scale, previous_scale * 0.65, 1e-6)
        shrunk_step = np.sign(raw_step) * max(abs(raw_step) - threshold, 0.0)

        estimates[index] = estimates[index - 1] + shrunk_step
        previous_scale = 0.85 * previous_scale + 0.15 * local_scale

    return estimates[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply either the basic moving-average or enhanced robust trend filter.

    Args:
        input_signal: Input one-dimensional time series
        window_size: Trailing processing horizon
        algorithm_type: "basic" or "enhanced"

    Returns:
        Filtered signal with length len(input_signal) - window_size + 1.
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