# EVOLVE-BLOCK-START
"""
Confirmed dual-scale adaptive signal filter for volatile non-stationary series.

Enhanced mode uses robust first-difference noise estimation, causal alpha-beta
tracking, fast/slow slope consensus, bounded innovations, predictive lead, and
model-confirmed directional hysteresis.  Outputs remain causally aligned to the
newest sample of each trailing window.
"""
from collections import deque
import numpy as np


class _ConfirmedDualScaleTracker:
    """Causal adaptive level/slope tracker with directional model confirmation."""

    def __init__(self, window_size):
        self.window_size = max(3, int(window_size))
        self.fast_span = max(4, min(8, self.window_size))

        self.level = None
        self.slope = 0.0
        self.fast_slope = 0.0
        self.slow_slope = 0.0
        self.last_sample = None
        self.last_output = None
        self.noise_scale = None

        self.differences = deque(maxlen=self.window_size)
        self.output_direction = 0.0
        self.pending_direction = 0.0
        self.pending_count = 0
        self.pending_excursion = 0.0

    @staticmethod
    def _difference_scale(values):
        """Robust observation-noise estimate obtained from first differences."""
        if len(values) < 3:
            return None
        d = np.asarray(values, dtype=float)
        center = np.median(d)
        mad = np.median(np.abs(d - center))
        scale = mad / (0.67448975 * np.sqrt(2.0))

        # Retain a useful floor for quantized or nearly constant signals.
        deviation = np.abs(d - center)
        trimmed = deviation[deviation <= np.percentile(deviation, 80)]
        if trimmed.size:
            rms = np.sqrt(np.mean(trimmed * trimmed)) / np.sqrt(2.0)
            scale = max(scale, 0.40 * rms)
        return max(scale, 1e-6)

    def _update_noise(self):
        slow = self._difference_scale(self.differences)
        if slow is None:
            return

        if len(self.differences) >= self.fast_span:
            fast = self._difference_scale(list(self.differences)[-self.fast_span:])
        else:
            fast = slow

        fast = np.clip(fast, 0.50 * slow, 2.40 * slow)
        target = 0.64 * fast + 0.36 * slow

        if self.noise_scale is None:
            self.noise_scale = target
        else:
            # Faster expansion than contraction accommodates volatility bursts
            # without allowing isolated impulses to dominate later thresholds.
            decay = 0.945 if target > self.noise_scale else 0.972
            self.noise_scale = max(
                1e-6, decay * self.noise_scale + (1.0 - decay) * target
            )

    def update(self, sample):
        """Consume one finite sample and return a filtered causal estimate."""
        sample = float(sample)

        if self.level is None:
            self.level = sample
            self.last_sample = sample
            self.last_output = sample
            self.noise_scale = max(abs(sample) * 0.025, 0.03, 1e-6)
            return sample

        raw_difference = sample - self.last_sample
        self.last_sample = sample
        self.differences.append(raw_difference)
        self._update_noise()

        sigma = max(self.noise_scale, 1e-6)
        predicted = self.level + self.slope
        raw_innovation = sample - predicted

        # Huber-like bounded innovation avoids state displacement by impulses.
        clip_limit = max(3.5 * sigma, 1e-6)
        innovation = float(np.clip(raw_innovation, -clip_limit, clip_limit))
        normalized = abs(innovation) / sigma
        response = normalized / (normalized + 2.15)

        alpha = 0.090 + 0.430 * response
        beta = 0.0030 + 0.085 * response * response

        self.level = predicted + alpha * innovation
        self.slope = (1.0 - beta) * self.slope + beta * innovation

        # Dual-rate derivative models provide a directional confidence measure.
        # The fast model recognizes a real turn early; the slow model prevents
        # brief noisy counter-moves from being mistaken for a reversal.
        self.fast_slope = 0.72 * self.fast_slope + 0.28 * raw_difference
        self.slow_slope = 0.925 * self.slow_slope + 0.075 * raw_difference

        consensus_slope = 0.55 * self.fast_slope + 0.45 * self.slow_slope
        model_slope = 0.60 * self.slope + 0.40 * consensus_slope

        lead_limit = max(1.15 * sigma, 0.80 * abs(model_slope), 1e-8)
        lead = float(np.clip(0.30 * model_slope, -lead_limit, lead_limit))
        candidate = self.level + lead

        delta = candidate - self.last_output
        direction = float(np.sign(delta))
        if direction == 0.0:
            return self.last_output

        small_band = max(0.38 * sigma, 1e-7)
        immediate_band = max(1.35 * sigma, 1e-7)

        if self.output_direction == 0.0 or direction == self.output_direction:
            self.output_direction = direction
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
            self.last_output = candidate
            return self.last_output

        fast_sign = np.sign(self.fast_slope)
        slow_sign = np.sign(self.slow_slope)
        consensus_strength = abs(consensus_slope) / sigma

        # A sharp accepted turn needs directional evidence, rather than merely
        # a large noisy endpoint displacement.
        model_confirmed = (
            fast_sign == direction and slow_sign == direction
        ) or (
            np.sign(consensus_slope) == direction and consensus_strength >= 0.80
        )
        strongly_confirmed = model_confirmed and consensus_strength >= 0.55

        if abs(delta) >= immediate_band and strongly_confirmed:
            self.output_direction = direction
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
            self.last_output = candidate
        elif abs(delta) <= small_band:
            self.pending_direction = 0.0
            self.pending_count = 0
            self.pending_excursion = 0.0
        else:
            if direction == self.pending_direction:
                self.pending_count += 1
                self.pending_excursion = max(self.pending_excursion, abs(delta))
            else:
                self.pending_direction = direction
                self.pending_count = 1
                self.pending_excursion = abs(delta)

            required = 2 if model_confirmed else 3
            required_excursion = (0.60 if model_confirmed else 0.90) * sigma
            if (
                self.pending_count >= required
                and self.pending_excursion >= required_excursion
            ):
                self.output_direction = direction
                self.pending_direction = 0.0
                self.pending_count = 0
                self.pending_excursion = 0.0
                self.last_output = candidate

        return self.last_output


class _SignalProcessingPipeline:
    """Validation and execution coordinator preserving the original API."""

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
        tracker = _ConfirmedDualScaleTracker(self.window_size)
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
    """Robust causal low-lag trend-preserving adaptive filter."""
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