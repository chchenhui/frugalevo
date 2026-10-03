# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series.

This version uses a causal adaptive trend pipeline:
1. A constant-velocity Kalman tracker estimates level and slope continuously.
2. A robust local regression estimates the newest sample in each sliding window.
3. Adaptive fusion combines both estimates to reduce noise while preserving motion.

The returned values retain the original sliding-window contract: output[i]
corresponds to the input endpoint x[i + window_size - 1].
"""
import numpy as np


class _AdaptiveTrendTracker:
    """Low-lag constant-velocity tracker with innovation-adaptive noise tuning."""

    def __init__(self, initial_value, measurement_variance, process_variance):
        self.state = np.array([float(initial_value), 0.0], dtype=float)
        self.covariance = np.array(
            [[max(measurement_variance, 1e-6), 0.0],
             [0.0, max(process_variance, 1e-6)]],
            dtype=float,
        )
        self.measurement_variance = max(float(measurement_variance), 1e-6)
        self.process_variance = max(float(process_variance), 1e-7)
        self.innovation_scale = np.sqrt(self.measurement_variance)

        self.transition = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        self.measurement = np.array([1.0, 0.0], dtype=float)

    def update(self, sample):
        """Update tracker and return causal level and velocity estimates."""
        predicted_state = self.transition @ self.state

        base_q = self.process_variance
        process_noise = np.array(
            [[0.25 * base_q, 0.5 * base_q],
             [0.5 * base_q, base_q]],
            dtype=float,
        )
        predicted_covariance = (
            self.transition @ self.covariance @ self.transition.T + process_noise
        )

        innovation = float(sample - predicted_state[0])

        # Robustly track innovation scale. Large persistent innovations indicate
        # genuine regime changes, while isolated spikes remain down-weighted.
        clipped_innovation = np.clip(
            innovation,
            -4.0 * self.innovation_scale,
            4.0 * self.innovation_scale,
        )
        self.innovation_scale = np.sqrt(
            0.96 * self.innovation_scale ** 2 + 0.04 * clipped_innovation ** 2 + 1e-10
        )

        adaptive_r = max(
            self.measurement_variance,
            0.55 * self.innovation_scale ** 2,
        )
        innovation_covariance = float(predicted_covariance[0, 0] + adaptive_r)
        gain = predicted_covariance[:, 0] / innovation_covariance

        # Huber-style innovation limiting prevents single-sample spikes from
        # causing false changes in slope direction.
        innovation_limit = 3.0 * max(self.innovation_scale, 1e-6)
        robust_innovation = np.clip(
            innovation,
            -innovation_limit,
            innovation_limit,
        )

        self.state = predicted_state + gain * robust_innovation
        self.covariance = (
            np.eye(2) - np.outer(gain, self.measurement)
        ) @ predicted_covariance

        return float(self.state[0]), float(self.state[1])


class _RobustEndpointRegressor:
    """Robust exponentially weighted local linear endpoint estimator."""

    def __init__(self, window_size):
        self.window_size = int(window_size)
        self.time = np.arange(-(window_size - 1), 1, dtype=float)

        # Recent samples receive more influence, preserving responsiveness.
        decay = max(window_size / 3.5, 1.0)
        self.base_weights = np.exp(self.time / decay)
        self.base_weights /= np.sum(self.base_weights)

    def estimate(self, window):
        """
        Return endpoint level, local slope, and residual-scale estimate.

        The regression is centered at the newest point, so its intercept is a
        causal estimate of the signal at the window endpoint.
        """
        values = np.asarray(window, dtype=float)
        design = np.column_stack((np.ones(self.window_size), self.time))
        weights = self.base_weights.copy()

        coefficients = np.array([values[-1], 0.0], dtype=float)

        # Two inexpensive IRLS passes provide robustness to impulsive samples.
        for _ in range(2):
            weighted_design = design * weights[:, None]
            normal_matrix = design.T @ weighted_design
            normal_matrix[1, 1] += 1e-8
            coefficients = np.linalg.solve(
                normal_matrix,
                weighted_design.T @ values,
            )

            residuals = values - design @ coefficients
            median_residual = np.median(residuals)
            scale = 1.4826 * np.median(np.abs(residuals - median_residual)) + 1e-8
            huber_weights = np.minimum(1.0, 1.8 * scale / (np.abs(residuals) + 1e-12))
            weights = self.base_weights * huber_weights

        final_residuals = values - design @ coefficients
        residual_scale = 1.4826 * np.median(
            np.abs(final_residuals - np.median(final_residuals))
        ) + 1e-8

        return float(coefficients[0]), float(coefficients[1]), float(residual_scale)


class _AdaptiveWindowPipeline:
    """Coordinates streaming state estimation and window-aligned output."""

    def __init__(self, samples, window_size):
        self.samples = np.asarray(samples, dtype=float)
        self.window_size = int(window_size)

        initial_window = self.samples[:self.window_size]
        differences = np.diff(initial_window)
        difference_scale = 1.4826 * np.median(
            np.abs(differences - np.median(differences))
        ) if differences.size else 0.0

        sample_scale = 1.4826 * np.median(
            np.abs(initial_window - np.median(initial_window))
        ) + 1e-6

        measurement_variance = max(
            (0.45 * sample_scale) ** 2,
            (0.7 * difference_scale) ** 2,
            1e-6,
        )
        process_variance = max(0.08 * measurement_variance, 1e-7)

        self.tracker = _AdaptiveTrendTracker(
            self.samples[0],
            measurement_variance,
            process_variance,
        )
        self.regressor = _RobustEndpointRegressor(self.window_size)

    def run(self):
        output_length = len(self.samples) - self.window_size + 1
        output = np.empty(output_length, dtype=float)

        for index, sample in enumerate(self.samples):
            kalman_level, kalman_slope = self.tracker.update(sample)

            if index < self.window_size - 1:
                continue

            window_start = index - self.window_size + 1
            window = self.samples[window_start:index + 1]
            local_level, local_slope, residual_scale = self.regressor.estimate(window)

            # Motion relative to local residual noise determines fusion.
            # In quiet periods, Kalman smoothing suppresses false reversals.
            # During genuine movement, endpoint regression receives more weight.
            trend_strength = abs(local_slope) * self.window_size
            motion_ratio = trend_strength / (residual_scale + 1e-8)
            local_weight = 0.20 + 0.45 * (motion_ratio / (1.0 + motion_ratio))

            # If the independent trend estimates disagree strongly, favor the
            # robust state tracker to avoid a single-window directional artifact.
            slope_disagreement = abs(local_slope - kalman_slope)
            disagreement_ratio = slope_disagreement / (residual_scale + 1e-8)
            local_weight /= 1.0 + 0.12 * disagreement_ratio
            local_weight = float(np.clip(local_weight, 0.12, 0.58))

            output[window_start] = (
                local_weight * local_level
                + (1.0 - local_weight) * kalman_level
            )

        return output


def _validate_signal_and_window(x, window_size):
    """Validate the original API inputs and return a numeric one-dimensional signal."""
    signal = np.asarray(x, dtype=float)

    if signal.ndim != 1:
        raise ValueError("Input signal must be a 1D array of real-valued samples")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if len(signal) < window_size:
        raise ValueError(
            f"Input signal length ({len(signal)}) must be >= window_size ({window_size})"
        )

    return signal


def adaptive_filter(x, window_size=20):
    """
    Baseline trailing moving-average filter.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    signal = _validate_signal_and_window(x, window_size)

    cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Adaptive low-lag filter for volatile non-stationary signals.

    The output preserves the original trailing-window alignment: output[i]
    estimates the signal at x[i + window_size - 1].
    """
    signal = _validate_signal_and_window(x, window_size)

    if window_size == 1:
        return signal.copy()

    return _AdaptiveWindowPipeline(signal, window_size).run()


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