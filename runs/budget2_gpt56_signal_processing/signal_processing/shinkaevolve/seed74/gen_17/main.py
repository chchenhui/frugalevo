# EVOLVE-BLOCK-START
"""
Robust real-time adaptive filtering for volatile non-stationary time series.

The enhanced filter is organized as a streaming estimation pipeline:
    1. Causal median measurement conditioning
    2. Robust innovation/noise-scale estimation
    3. Adaptive constant-velocity Kalman prediction/correction
    4. Hysteretic velocity stabilization

The output remains aligned with input samples window_size - 1 onward, preserving
the public API and output length of the original implementation.
"""
from collections import deque
import numpy as np


class _RobustScaleEstimator:
    """Rolling robust scale estimator for innovations and sample differences."""

    def __init__(self, window_size, floor=1e-6):
        self.values = deque(maxlen=max(3, window_size))
        self.floor = float(floor)

    def update(self, value):
        self.values.append(float(value))

    def scale(self):
        if len(self.values) < 3:
            return self.floor

        values = np.asarray(self.values, dtype=float)
        center = np.median(values)
        mad = np.median(np.abs(values - center))
        return max(1.4826 * mad, self.floor)


class _CausalMedianConditioner:
    """Small causal median stage that removes isolated single-sample spikes."""

    def __init__(self):
        self.samples = deque(maxlen=3)

    def update(self, sample):
        self.samples.append(float(sample))
        if len(self.samples) < 3:
            return float(sample)
        return float(np.median(np.asarray(self.samples, dtype=float)))


class _AdaptiveTrendTracker:
    """
    Adaptive two-state level/velocity estimator.

    Measurement uncertainty is estimated from recent first differences. This
    allows the tracker to become smoother in noisy intervals while retaining
    responsiveness when the underlying trend has meaningful momentum.
    """

    def __init__(self, window_size, initial_value):
        self.level = float(initial_value)
        self.velocity = 0.0

        # Covariance for [level, velocity].
        self.p00 = 1.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 0.25

        self.previous_measurement = float(initial_value)
        self.diff_scale = _RobustScaleEstimator(window_size)
        self.innovation_scale = _RobustScaleEstimator(window_size)
        self.trend_measurements = deque(maxlen=max(4, window_size))

    def _trend_is_coherent(self, measurement, noise_scale):
        """
        Compare full-window and recent-window regression slopes.

        A genuine maneuver normally has compatible slopes at both scales.
        A disagreement is a useful causal indication that a large innovation is
        an excursion rather than a trend that should receive high Kalman gain.
        """
        self.trend_measurements.append(float(measurement))
        values = np.asarray(self.trend_measurements, dtype=float)
        count = len(values)

        if count < 6:
            return True

        recent_count = min(count, max(6, min(10, self.trend_measurements.maxlen // 2)))
        full_time = np.arange(count, dtype=float)
        recent_time = np.arange(recent_count, dtype=float)

        def fitted_slope(samples, time):
            centered_time = time - np.mean(time)
            denominator = np.dot(centered_time, centered_time)
            return np.dot(centered_time, samples - np.mean(samples)) / max(denominator, 1e-12)

        full_slope = fitted_slope(values, full_time)
        recent_slope = fitted_slope(values[-recent_count:], recent_time)
        slope_floor = 0.06 * max(noise_scale, 1e-8)

        if abs(full_slope) < slope_floor:
            # In a flat broad trend, only permit high gain after the recent
            # movement itself rises clearly above the local noise floor.
            return abs(recent_slope) < 2.5 * slope_floor

        same_direction = full_slope * recent_slope > 0.0
        magnitude_ratio = abs(recent_slope) / max(abs(full_slope), slope_floor)
        return same_direction and 0.30 <= magnitude_ratio <= 3.25

    def _measurement_noise(self):
        # Difference MAD estimates noise in a nearly causal, robust manner.
        # sqrt(2) converts difference scale back to per-sample scale.
        noise = self.diff_scale.scale() / np.sqrt(2.0)
        return max(noise * noise, 1e-7)

    def update(self, measurement):
        measurement = float(measurement)

        difference = measurement - self.previous_measurement
        self.diff_scale.update(difference)
        self.previous_measurement = measurement

        measurement_variance = self._measurement_noise()
        noise_scale = np.sqrt(measurement_variance)
        trend_coherent = self._trend_is_coherent(measurement, noise_scale)

        # ---- Predict: constant velocity state model ----
        predicted_level = self.level + self.velocity
        predicted_velocity = self.velocity

        # Process noise grows gently with estimated movement. This makes the
        # filter responsive during genuine transitions without making it chase
        # stationary measurement noise.
        motion = abs(self.velocity)
        motion_ratio = min(1.0, motion / (noise_scale + 1e-8))
        coherence_factor = 1.0 if trend_coherent else 0.42
        q_level = measurement_variance * (
            0.020 + coherence_factor * 0.095 * motion_ratio
        )
        q_velocity = measurement_variance * (0.003 + 0.005 * coherence_factor)

        pp00 = self.p00 + self.p01 + self.p10 + self.p11 + q_level
        pp01 = self.p01 + self.p11
        pp10 = self.p10 + self.p11
        pp11 = self.p11 + q_velocity

        # ---- Robust innovation gate ----
        innovation = measurement - predicted_level
        innovation_noise = max(self.innovation_scale.scale(), np.sqrt(measurement_variance))
        gate = 4.0 * innovation_noise

        # Huber clipping prevents isolated impulses from causing false turns.
        clipped_innovation = float(np.clip(innovation, -gate, gate))
        self.innovation_scale.update(innovation)

        # Large innovations retain some responsiveness, but receive extra
        # measurement variance when they appear statistically impulsive.
        outlier_ratio = abs(innovation) / (gate + 1e-12)
        effective_r = measurement_variance * (
            1.0 + max(0.0, outlier_ratio - 1.0) * 5.0
        )
        if not trend_coherent:
            # Conflicting multi-scale slopes cap reaction to a single local
            # excursion without suppressing coherent sustained movement.
            effective_r *= 2.4

        # ---- Correct ----
        innovation_variance = max(pp00 + effective_r, 1e-10)
        k_level = pp00 / innovation_variance
        k_velocity = pp10 / innovation_variance

        if not trend_coherent:
            k_level = min(k_level, 0.46)
            k_velocity = min(k_velocity, 0.18) if k_velocity >= 0.0 else max(k_velocity, -0.18)

        self.level = predicted_level + k_level * clipped_innovation
        raw_velocity = predicted_velocity + k_velocity * clipped_innovation

        # Joseph-style simplified covariance update for scalar measurement.
        self.p00 = max((1.0 - k_level) * pp00, 1e-10)
        self.p01 = (1.0 - k_level) * pp01
        self.p10 = pp10 - k_velocity * pp00
        self.p11 = max(pp11 - k_velocity * pp01, 1e-10)

        # ---- Direction hysteresis ----
        # A reversal must exceed a noise-relative threshold. This suppresses
        # flickering slopes while allowing sustained movement to change trend.
        reversal_threshold = 0.18 * innovation_noise
        if (
            self.velocity != 0.0
            and raw_velocity * self.velocity < 0.0
            and abs(raw_velocity) < reversal_threshold
        ):
            raw_velocity = 0.0

        # Light velocity damping reduces slope-change count without creating
        # the substantial phase delay of a trailing moving average.
        self.velocity = 0.78 * self.velocity + 0.22 * raw_velocity

        return self.level


class _StreamingAdaptiveFilter:
    """Composable real-time enhanced filtering pipeline."""

    def __init__(self, window_size, initial_value):
        self.conditioner = _CausalMedianConditioner()
        self.tracker = _AdaptiveTrendTracker(window_size, initial_value)

    def update(self, sample):
        conditioned = self.conditioner.update(sample)
        return self.tracker.update(conditioned)


def _validate_signal(x, window_size):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return x


def adaptive_filter(x, window_size=20):
    """
    Efficient trailing moving-average baseline.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        Filtered output signal with length len(x) - window_size + 1.
    """
    x = _validate_signal(x, window_size)
    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal robust adaptive level-and-trend filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Rolling robust-statistics window size.

    Returns:
        Filtered samples aligned with input samples window_size - 1 onward.
    """
    x = _validate_signal(x, window_size)

    pipeline = _StreamingAdaptiveFilter(window_size, x[0])
    estimates = np.empty(len(x), dtype=float)

    for index, sample in enumerate(x):
        estimates[index] = pipeline.update(sample)

    return estimates[window_size - 1:]


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected signal-processing algorithm.

    Args:
        input_signal: Input time series data.
        window_size: Sliding statistics window size.
        algorithm_type: "basic" selects moving average; all other values select
            the enhanced adaptive tracker for backward compatibility.

    Returns:
        Filtered signal.
    """
    if algorithm_type == "basic":
        return adaptive_filter(input_signal, window_size)
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


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