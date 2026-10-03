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
    Causal robust alpha-beta tracker with endpoint-aligned local trend estimation.

    The output remains aligned to the newest sample in each window.  A weighted
    local linear fit reduces average-filter lag, while adaptive gains and a
    reversal gate suppress noise-driven changes in direction.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Time zero is the latest sample, so the fitted intercept is a causal
    # estimate of the current level rather than a delayed window average.
    time = np.arange(window_size, dtype=float) - (window_size - 1)
    # A moderately broader recency horizon stabilizes endpoint slope estimates
    # while retaining an endpoint-aligned, low-phase-delay measurement.
    weights = np.exp(np.linspace(-2.85, 0.0, window_size))
    weights /= np.sum(weights)
    weighted_time = np.sum(weights * time)
    time_variance = np.sum(weights * (time - weighted_time) ** 2)

    # A short trailing regression provides independent, fast trend evidence.
    # It is used only to qualify velocity adaptation, not as a noisy direct
    # endpoint measurement.
    recent_length = min(window_size, 9)
    recent_time = time[-recent_length:]
    recent_weights = np.exp(np.linspace(-1.35, 0.0, recent_length))
    recent_weights /= np.sum(recent_weights)
    recent_time_mean = np.sum(recent_weights * recent_time)
    recent_time_variance = np.sum(
        recent_weights * (recent_time - recent_time_mean) ** 2
    )

    position = 0.0
    velocity = 0.0
    previous_direction = 0.0
    pending_direction = 0.0
    pending_count = 0

    for i in range(output_length):
        window = x[i : i + window_size]

        # Weighted least-squares local line, evaluated at the current endpoint.
        weighted_mean = np.sum(weights * window)
        local_slope = np.sum(weights * (time - weighted_time) * (window - weighted_mean)) / time_variance
        measurement = weighted_mean - local_slope * weighted_time

        recent_window = window[-recent_length:]
        recent_mean = np.sum(recent_weights * recent_window)
        recent_slope = np.sum(
            recent_weights
            * (recent_time - recent_time_mean)
            * (recent_window - recent_mean)
        ) / max(recent_time_variance, 1e-12)

        # Estimate local noise from residuals without allowing spikes to inflate
        # the scale estimate and make the tracker insensitive.
        residuals = window - (measurement + local_slope * time)
        residual_center = np.median(residuals)
        noise_scale = max(
            1.4826 * np.median(np.abs(residuals - residual_center)),
            1e-8,
        )

        if i == 0:
            position = measurement
            velocity = local_slope
            y[i] = position
            previous_direction = np.sign(velocity)
            continue

        prediction = position + velocity
        innovation = measurement - prediction

        # Significant innovations receive a rapid update; ordinary innovations
        # use conservative gains to improve smoothness.
        activity = min(abs(innovation) / (2.5 * noise_scale), 1.0)
        # Limit the state impact of a single corrupted endpoint fit.  Activity
        # still responds to the unbounded innovation, so sustained real turns
        # receive the stronger adaptive gain.
        bounded_innovation = np.clip(
            innovation,
            -3.25 * noise_scale,
            3.25 * noise_scale,
        )
        alpha = 0.18 + 0.50 * activity
        beta = 0.012 + 0.085 * activity

        # Only let the measured slope materially alter velocity when a short
        # and long horizon agree. Disagreement is characteristic of transient
        # noise or an unconfirmed turn, so persistence is preferred then.
        slope_ratio = min(
            abs(local_slope),
            abs(recent_slope),
        ) / (max(abs(local_slope), abs(recent_slope)) + 1e-12)
        slope_agreement = (
            np.sign(local_slope) == np.sign(recent_slope)
            and abs(local_slope) > 0.045 * noise_scale
            and abs(recent_slope) > 0.045 * noise_scale
        )
        slope_confidence = slope_ratio if slope_agreement else 0.0

        # Multi-scale agreement is used solely as gain control. This retains
        # the stable endpoint measurement while making coherent trends faster
        # and preventing a disagreeing short-scale fluctuation from rotating
        # the velocity state.
        if slope_agreement:
            alpha = min(alpha * (1.0 + 0.10 * slope_confidence), 0.72)
            beta *= 1.0 + 0.20 * slope_confidence
        elif abs(recent_slope) > 0.045 * noise_scale:
            beta *= 0.55

        slope_gain = 0.035 + 0.145 * slope_confidence
        candidate_velocity = (
            velocity
            + beta * bounded_innovation
            + slope_gain * (local_slope - velocity)
        )
        candidate_direction = np.sign(candidate_velocity)

        # A real turn normally appears at both horizons. Otherwise require two
        # consecutive proposed turns before changing the persistent direction;
        # this blocks alternating endpoint-noise reversals without delaying a
        # strong, multi-scale-confirmed reversal.
        if (
            previous_direction != 0.0
            and candidate_direction != 0.0
            and candidate_direction != previous_direction
        ):
            strong_confirmed_turn = (
                slope_agreement
                and np.sign(local_slope) == candidate_direction
                and abs(candidate_velocity)
                > 0.30 * noise_scale + 0.40 * abs(velocity)
            )

            if strong_confirmed_turn:
                pending_direction = 0.0
                pending_count = 0
            elif pending_direction == candidate_direction:
                pending_count += 1
            else:
                pending_direction = candidate_direction
                pending_count = 1

            if not strong_confirmed_turn and pending_count < 2:
                candidate_velocity = 0.60 * velocity
            else:
                pending_direction = 0.0
                pending_count = 0
        else:
            pending_direction = 0.0
            pending_count = 0

        position = prediction + alpha * bounded_innovation
        velocity = candidate_velocity
        if abs(velocity) > noise_scale / max(window_size, 1):
            previous_direction = np.sign(velocity)
        y[i] = position

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