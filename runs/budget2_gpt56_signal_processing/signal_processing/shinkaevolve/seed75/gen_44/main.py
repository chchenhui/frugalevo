# EVOLVE-BLOCK-START
"""
Causal robust multi-scale local-regression signal filter.

Enhanced filtering uses two exponentially weighted trailing local-linear models:
a responsive short-horizon fit and a stable long-horizon fit.  Their endpoint
predictions are blended according to slope activity and robust fit reliability.
Huber reweighting rejects impulses, and directional consensus hysteresis avoids
noise-induced reversals while retaining causal endpoint alignment.
"""
from collections import deque
import numpy as np


class _RobustMultiscaleRegression:
    """Causal robust local-regression smoother with confirmed output turns."""

    def __init__(self, window_size):
        self.window_size = max(3, int(window_size))
        self.fast_span = max(3.0, min(5.5, 0.36 * self.window_size))
        self.slow_span = max(5.0, min(float(self.window_size), 0.72 * self.window_size))

        self.samples = deque(maxlen=self.window_size)
        self.differences = deque(maxlen=self.window_size)

        self.last_sample = None
        self.last_output = None
        self.noise_scale = None

        self.direction = 0.0
        self.pending_direction = 0.0
        self.pending_count = 0
        self.pending_excursion = 0.0

    @staticmethod
    def _mad_scale(values):
        """Robust standard-deviation estimate with a small RMS fallback."""
        values = np.asarray(values, dtype=float)
        if values.size < 3:
            return None

        median = np.median(values)
        residual = np.abs(values - median)
        mad = np.median(residual)
        scale = mad / 0.67448975

        # MAD can vanish for quantized data; trimmed RMS remains robust enough
        # to establish an adaptive nonzero gate in that case.
        cutoff = np.percentile(residual, 80.0)
        trimmed = residual[residual <= cutoff]
        if trimmed.size:
            scale = max(scale, 0.35 * np.sqrt(np.mean(trimmed * trimmed)))
        return max(float(scale), 1e-6)

    def _update_noise(self, difference):
        self.differences.append(float(difference))
        raw = self._mad_scale(self.differences)
        if raw is None:
            return

        # Difference variance contains two independent observation errors.
        target = max(raw / np.sqrt(2.0), 1e-6)
        if self.noise_scale is None:
            self.noise_scale = target
        else:
            # Volatility expansion is intentionally faster than relaxation.
            rate = 0.10 if target > self.noise_scale else 0.025
            self.noise_scale = max(
                1e-6, (1.0 - rate) * self.noise_scale + rate * target
            )

    @staticmethod
    def _weighted_line(values, span, sigma):
        """
        Robust endpoint local-linear fit.

        Coordinates are negative sample ages, so the intercept is directly the
        fitted value at the latest observation and the slope is samples/step.
        """
        y = np.asarray(values, dtype=float)
        count = y.size
        if count == 1:
            return float(y[0]), 0.0, max(sigma, 1e-6)

        age = np.arange(count, dtype=float) - (count - 1.0)
        base_weight = np.exp(age / max(span, 1.0))
        weights = base_weight.copy()
        design = np.column_stack((np.ones(count), age))
        coefficient = np.array([y[-1], 0.0], dtype=float)
        fit_scale = max(sigma, 1e-5)

        # Two IRLS passes are sufficient at these small windows and preserve
        # real ramps better than median-only smoothing.
        for _ in range(2):
            normal = design.T @ (weights[:, None] * design)
            rhs = design.T @ (weights * y)
            ridge = 1e-8 * max(float(normal[0, 0]), 1.0)
            normal[1, 1] += ridge
            try:
                coefficient = np.linalg.solve(normal, rhs)
            except np.linalg.LinAlgError:
                coefficient = np.linalg.lstsq(
                    design * np.sqrt(weights)[:, None],
                    y * np.sqrt(weights),
                    rcond=None,
                )[0]

            residual = y - design @ coefficient
            local_scale = _RobustMultiscaleRegression._mad_scale(residual)
            if local_scale is not None:
                fit_scale = max(local_scale, 0.30 * sigma, 1e-6)

            # Huber influence: impulses lose leverage rather than shifting the
            # endpoint and generating a false trend reversal.
            limit = 2.35 * fit_scale
            influence = np.ones(count, dtype=float)
            mask = np.abs(residual) > limit
            influence[mask] = limit / np.maximum(np.abs(residual[mask]), 1e-12)
            weights = base_weight * influence

        return float(coefficient[0]), float(coefficient[1]), float(fit_scale)

    def update(self, sample):
        """Consume one sample and return a causal filtered endpoint estimate."""
        sample = float(sample)
        if self.last_sample is None:
            self.samples.append(sample)
            self.last_sample = sample
            self.last_output = sample
            self.noise_scale = max(0.03, 0.025 * abs(sample), 1e-6)
            return sample

        self._update_noise(sample - self.last_sample)
        self.last_sample = sample
        self.samples.append(sample)

        sigma = max(self.noise_scale, 1e-6)
        values = np.asarray(self.samples, dtype=float)

        fast_level, fast_slope, fast_fit = self._weighted_line(
            values, self.fast_span, sigma
        )
        slow_level, slow_slope, slow_fit = self._weighted_line(
            values, self.slow_span, sigma
        )

        # The fast endpoint is useful only when its residual quality is at
        # least comparable to the stable fit and the newest residual is not an
        # isolated impulse.
        endpoint_residual = abs(sample - fast_level) / max(fast_fit, sigma, 1e-6)
        reliability = 1.0 / (
            1.0
            + max(0.0, fast_fit / max(slow_fit, 1e-6) - 1.0)
            + 0.30 * max(0.0, endpoint_residual - 2.0)
        )

        slope_gap = abs(fast_slope - slow_slope)
        activity = abs(fast_slope) / (abs(fast_slope) + 0.75 * sigma + 1e-12)
        fast_weight = np.clip(
            (0.18 + 0.57 * activity) * reliability,
            0.08,
            0.64,
        )

        level = fast_weight * fast_level + (1.0 - fast_weight) * slow_level
        slope = fast_weight * fast_slope + (1.0 - fast_weight) * slow_slope

        # A restrained endpoint extrapolation offsets residual causal fitting
        # lag without allowing disagreement between models to create overshoot.
        agreement = 1.0 / (1.0 + slope_gap / (sigma + 1e-12))
        lead = 0.20 * agreement * slope
        lead_limit = max(0.65 * sigma, 0.70 * abs(slope))
        candidate = level + float(np.clip(lead, -lead_limit, lead_limit))

        delta = candidate - self.last_output
        proposed_direction = float(np.sign(delta))
        if proposed_direction == 0.0:
            return self.last_output

        if self.direction == 0.0 or proposed_direction == self.direction:
            self.direction = proposed_direction
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
            self.last_output = candidate
            return self.last_output

        consensus = (
            np.sign(fast_slope) == proposed_direction
            and np.sign(slow_slope) == proposed_direction
        )
        strength = abs(slope) / sigma
        reversal_band = max(0.42 * sigma, 1e-7)
        decisive_band = max(1.15 * sigma, 1e-7)

        if abs(delta) <= reversal_band:
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
            return self.last_output

        if proposed_direction == self.pending_direction:
            self.pending_count += 1
            self.pending_excursion += abs(delta)
        else:
            self.pending_direction = proposed_direction
            self.pending_count = 1
            self.pending_excursion = abs(delta)

        required = 2 if consensus and strength >= 0.32 else 3
        accepted = (
            (consensus and abs(delta) >= decisive_band)
            or (
                self.pending_count >= required
                and self.pending_excursion >= (0.85 if consensus else 1.25) * sigma
            )
        )

        if accepted:
            self.direction = proposed_direction
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
            self.last_output = candidate

        return self.last_output


class _SignalProcessingPipeline:
    """Validation and execution coordinator preserving the public API."""

    def __init__(self, window_size, mode):
        self.window_size = int(window_size)
        self.mode = mode

    @staticmethod
    def _as_signal_array(x):
        signal = np.asarray(x, dtype=float)
        if signal.ndim != 1:
            raise ValueError("Input signal must be a 1D array of real-valued samples")
        return signal

    def validate(self, x):
        signal = self._as_signal_array(x)
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if len(signal) < self.window_size:
            raise ValueError(
                f"Input signal length ({len(signal)}) must be >= "
                f"window_size ({self.window_size})"
            )
        return signal

    def run_basic(self, signal):
        cumulative = np.concatenate(([0.0], np.cumsum(signal, dtype=float)))
        return (
            cumulative[self.window_size:] - cumulative[:-self.window_size]
        ) / self.window_size

    def run_enhanced(self, signal):
        tracker = _RobustMultiscaleRegression(self.window_size)
        estimates = np.empty(signal.size, dtype=float)
        last_valid = 0.0

        for index, value in enumerate(signal):
            if np.isfinite(value):
                last_valid = float(value)
            estimates[index] = tracker.update(last_valid)

        return estimates[self.window_size - 1:]

    def run(self, x):
        signal = self.validate(x)
        if self.mode == "enhanced":
            return self.run_enhanced(signal)
        return self.run_basic(signal)


def adaptive_filter(x, window_size=20):
    """Basic trailing moving-average filter."""
    return _SignalProcessingPipeline(window_size, "basic").run(x)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Robust causal multi-scale local-regression trend filter."""
    return _SignalProcessingPipeline(window_size, "enhanced").run(x)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Apply enhanced adaptive filtering or the basic trailing average."""
    mode = "enhanced" if algorithm_type == "enhanced" else "basic"
    return _SignalProcessingPipeline(window_size, mode).run(input_signal)


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