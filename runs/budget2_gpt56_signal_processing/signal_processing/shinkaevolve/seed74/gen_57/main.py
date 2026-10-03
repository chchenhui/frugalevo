# EVOLVE-BLOCK-START
"""
Robust adaptive Kalman filtering for volatile non-stationary time series.

Pipeline:
1. Causal robust endpoint observation using a median-assisted local regression.
2. Innovation-adaptive constant-velocity Kalman filter.
3. CUSUM/hysteretic reversal confirmation applied only to reported output.

The output is aligned with the newest sample of each sliding window and retains
the original API: output length is len(x) - window_size + 1.
"""
import numpy as np


def _validate_signal(x, window_size):
    """Convert and validate a one-dimensional finite signal."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite values")
    return x


def _mad_scale(values, fallback=1e-6):
    """Robust standard-deviation estimate."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return fallback
    center = np.median(values)
    return max(1.4826 * np.median(np.abs(values - center)), fallback)


def _endpoint_observation(x, index, span):
    """
    Produce a causal endpoint estimate.

    A small median filter handles isolated spikes. A recency weighted line is
    then evaluated at the current endpoint, avoiding moving-average lag.
    """
    start = max(0, index - span + 1)
    segment = x[start:index + 1]
    n = len(segment)

    if n < 3:
        return float(segment[-1])

    # Causal three-point median is robust but still reacts at the current time.
    if n >= 3:
        endpoint_median = float(np.median(segment[-3:]))
    else:
        endpoint_median = float(segment[-1])

    t = np.arange(-(n - 1), 1, dtype=float)
    weights = np.exp(t / max(1.5, n / 2.5))
    weights /= np.sum(weights)

    mean_t = np.sum(weights * t)
    mean_x = np.sum(weights * segment)
    centered_t = t - mean_t
    denominator = np.sum(weights * centered_t * centered_t)

    if denominator < 1e-12:
        regression_endpoint = float(segment[-1])
    else:
        slope = np.sum(weights * centered_t * (segment - mean_x)) / denominator
        regression_endpoint = float(mean_x - slope * mean_t)

    # Median dominates during impulse contamination; regression reduces ordinary
    # measurement variance and gives a nearly zero-delay endpoint estimate.
    return 0.58 * endpoint_median + 0.42 * regression_endpoint


class _AdaptiveKalmanTracker:
    """
    Robust constant-velocity Kalman tracker with temporary reversal hysteresis.

    The state itself is never frozen: it continues to assimilate measurements
    immediately. Only the exposed signal is held for a short confirmation
    interval when a weak counter-trend movement occurs.
    """

    def __init__(self, window_size, initial_value):
        self.window_size = window_size

        self.state = np.array([float(initial_value), 0.0], dtype=float)
        self.covariance = np.array([[1.0, 0.0], [0.0, 0.25]], dtype=float)

        self.reported = float(initial_value)
        self.reported_direction = 0.0
        self.pending_direction = 0.0
        self.pending_score = 0.0
        self.pending_count = 0

        self.innovations = []

    def _noise_scale(self, x, index):
        start = max(0, index - self.window_size)
        differences = np.diff(x[start:index + 1])
        diff_scale = _mad_scale(differences, 1e-5) / np.sqrt(2.0)

        if self.innovations:
            innovation_scale = _mad_scale(
                self.innovations[-self.window_size:], 1e-5
            )
        else:
            innovation_scale = 1e-5

        # Difference scale reacts quickly to non-stationary measurement noise;
        # innovation scale protects the filter when local slopes are tiny.
        return max(0.70 * diff_scale, 0.45 * innovation_scale, 1e-5)

    def step(self, observation, x, index):
        """Consume one endpoint observation and return a stable causal estimate."""
        noise = self._noise_scale(x, index)
        measurement_variance = max((0.82 * noise) ** 2, 1e-9)

        transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        predicted_state = transition @ self.state
        predicted_covariance = transition @ self.covariance @ transition.T

        raw_innovation = observation - predicted_state[0]
        clipped_innovation = np.clip(raw_innovation, -4.0 * noise, 4.0 * noise)

        normalized = abs(raw_innovation) / max(noise, 1e-8)
        maneuver = np.clip((normalized - 0.8) / 2.5, 0.0, 1.0)

        # Low process noise smooths ordinary samples. A significant innovation
        # expands the model uncertainty so real accelerations are followed with
        # much lower delay than a fixed-gain tracker.
        acceleration_variance = (
            (0.012 * noise) ** 2
            + maneuver * (0.19 * noise) ** 2
        )
        process_covariance = acceleration_variance * np.array(
            [[0.25, 0.5], [0.5, 1.0]], dtype=float
        )
        predicted_covariance += process_covariance

        innovation_variance = predicted_covariance[0, 0] + measurement_variance
        gain = predicted_covariance[:, 0] / max(innovation_variance, 1e-12)

        self.state = predicted_state + gain * clipped_innovation
        identity = np.eye(2)
        observation_matrix = np.array([[1.0, 0.0]])
        correction = identity - gain[:, None] @ observation_matrix

        # Joseph form is numerically stable for long real-time streams.
        self.covariance = (
            correction @ predicted_covariance @ correction.T
            + np.outer(gain, gain) * measurement_variance
        )

        candidate = float(self.state[0])
        delta = candidate - self.reported
        slope_floor = max(0.055 * noise, 1e-7)
        candidate_direction = np.sign(delta) if abs(delta) > slope_floor else 0.0
        innovation_direction = np.sign(clipped_innovation)

        if self.reported_direction == 0.0 and candidate_direction != 0.0:
            self.reported_direction = candidate_direction

        counter_trend = (
            self.reported_direction != 0.0
            and candidate_direction != 0.0
            and candidate_direction != self.reported_direction
        )

        if not counter_trend:
            self.pending_direction = 0.0
            self.pending_score = 0.0
            self.pending_count = 0
            self.reported = candidate
            if candidate_direction != 0.0:
                self.reported_direction = candidate_direction
        else:
            # A valid turn must be supported by the measurement innovation in
            # the same direction as the proposed reported displacement. This
            # prevents prediction drift and alternating endpoint noise from
            # accumulating reversal credit.
            coherent_evidence = innovation_direction == candidate_direction
            evidence = (
                max(abs(clipped_innovation) / max(noise, 1e-8) - 0.65, 0.0)
                if coherent_evidence
                else 0.0
            )

            if candidate_direction != self.pending_direction:
                self.pending_direction = candidate_direction
                self.pending_score = 0.0
                self.pending_count = 0

            if coherent_evidence:
                self.pending_score = min(self.pending_score + evidence, 8.0)
                self.pending_count += 1
            else:
                # Preserve only a small fraction of prior evidence when the
                # measurement stops supporting the pending direction.
                self.pending_score *= 0.35
                self.pending_count = 0

            immediate_turn = (
                coherent_evidence
                and abs(delta) > 3.0 * noise
                and abs(clipped_innovation) > 2.0 * noise
            )
            confirmed = immediate_turn or (
                self.pending_count >= 3
                and self.pending_score >= 2.8
            )

            if confirmed:
                self.reported = candidate
                self.reported_direction = candidate_direction
                self.pending_direction = 0.0
                self.pending_score = 0.0
                self.pending_count = 0
            else:
                # Brief hold removes weak counter-trend jitter while allowing
                # the internal Kalman state to develop reversal evidence.
                self.state[0] = 0.75 * self.state[0] + 0.25 * self.reported

        self.innovations.append(float(raw_innovation))
        if len(self.innovations) > self.window_size:
            self.innovations.pop(0)

        return self.reported


def adaptive_filter(x, window_size=20):
    """Baseline trailing moving-average filter."""
    x = _validate_signal(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Apply robust causal adaptive Kalman filtering.

    Returns filtered samples aligned with input samples from window_size - 1.
    """
    x = _validate_signal(x, window_size)

    observation_span = max(4, min(9, window_size))
    tracker = _AdaptiveKalmanTracker(window_size, x[0])
    estimates = np.empty(len(x), dtype=float)
    estimates[0] = x[0]

    for index in range(1, len(x)):
        observation = _endpoint_observation(x, index, observation_span)
        estimates[index] = tracker.step(observation, x, index)

    return estimates[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected algorithm.

    Args:
        input_signal: Input time series data.
        window_size: Sliding-window size.
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