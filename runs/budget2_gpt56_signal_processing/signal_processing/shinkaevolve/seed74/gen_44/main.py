# EVOLVE-BLOCK-START
"""
Evidence-locked causal adaptive filter for volatile non-stationary signals.

Processing pipeline:
    1. Robust short/long causal endpoint regressions form a multi-scale
       zero-delay measurement.
    2. An adaptive alpha-beta tracker produces a responsive latent estimate.
    3. A monotonic output controller exposes latent motion in the current
       direction immediately, but requires normalized cumulative evidence
       before allowing a visible reversal.

The returned samples remain aligned to the newest observation of every
complete input window.
"""
import numpy as np
from dataclasses import dataclass


def _robust_scale(values, fallback=1e-6):
    """MAD-based finite scale estimate."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    center = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - center))

    if not np.isfinite(scale) or scale < 1e-10:
        scale = np.std(values)

    if not np.isfinite(scale) or scale < 1e-10:
        scale = max(float(fallback), 1e-6)

    return float(scale)


def _weighted_line_fit(time, values, weights):
    """Return endpoint intercept and slope of a weighted linear fit."""
    sw = float(np.sum(weights))
    st = float(np.sum(weights * time))
    stt = float(np.sum(weights * time * time))
    sz = float(np.sum(weights * values))
    stz = float(np.sum(weights * time * values))

    determinant = sw * stt - st * st
    if not np.isfinite(determinant) or determinant <= 1e-14:
        return float(np.average(values, weights=weights)), 0.0

    level = (stt * sz - st * stz) / determinant
    slope = (sw * stz - st * sz) / determinant
    return float(level), float(slope)


def _endpoint_model(window, decay, fallback_scale):
    """
    Robust exponentially weighted endpoint line model.

    Time is centered at the latest observation, making the intercept a causal
    estimate with no additional phase shift.
    """
    window = np.asarray(window, dtype=float)
    valid = np.isfinite(window)

    count = int(np.count_nonzero(valid))
    if count == 0:
        return np.nan, 0.0, float(fallback_scale)
    if count == 1:
        return float(window[valid][0]), 0.0, float(fallback_scale)

    length = len(window)
    time = np.arange(length, dtype=float) - float(length - 1)
    time = time[valid]
    values = window[valid]

    weights = np.exp(time / max(float(decay), 1.0))
    weights /= max(float(np.sum(weights)), 1e-12)

    level, slope = _weighted_line_fit(time, values, weights)

    # A single Huber reweighting pass robustly suppresses impulse noise while
    # retaining the low latency of endpoint regression.
    residual = values - (level + slope * time)
    scale = _robust_scale(residual, fallback=fallback_scale)
    limit = max(1.7 * scale, 1e-10)
    huber = np.minimum(1.0, limit / np.maximum(np.abs(residual), 1e-12))

    level, slope = _weighted_line_fit(time, values, weights * huber)
    residual = values - (level + slope * time)
    scale = _robust_scale(residual, fallback=fallback_scale)

    return level, slope, scale


def _recent_slope(window, fallback=0.0):
    """Fast unweighted slope used as independent turning evidence."""
    window = np.asarray(window, dtype=float)
    valid = np.isfinite(window)

    if np.count_nonzero(valid) < 2:
        return float(fallback)

    time = np.arange(len(window), dtype=float)[valid]
    values = window[valid]
    time -= np.mean(time)
    denominator = float(np.dot(time, time))

    if denominator <= 1e-12:
        return float(fallback)

    return float(np.dot(time, values - np.mean(values)) / denominator)


def _global_noise_level(x):
    """Estimate sample noise scale from robust first differences."""
    values = x[np.isfinite(x)]

    if values.size > 2:
        noise = _robust_scale(np.diff(values), fallback=1e-5) / np.sqrt(2.0)
    else:
        noise = _robust_scale(values, fallback=1e-5)

    return max(float(noise), 1e-6)


def _consensus_measurement(short_fit, long_fit, noise):
    """
    Combine short and long endpoint models.

    Short-scale response dominates when model slopes agree. During disagreement,
    the longer model receives more influence to reject transient noise.
    """
    short_level, short_slope, short_scale = short_fit
    long_level, long_slope, long_scale = long_fit

    if not np.isfinite(short_level):
        return long_level, long_slope, long_scale, 0.0
    if not np.isfinite(long_level):
        return short_level, short_slope, short_scale, 0.0

    reference = max(
        0.30 * (abs(short_slope) + abs(long_slope)),
        0.13 * noise,
        1e-9,
    )
    agreement = float(
        np.tanh((short_slope * long_slope) / (reference * reference))
    )

    short_weight = float(np.clip(0.51 + 0.18 * agreement, 0.33, 0.70))
    level = short_weight * short_level + (1.0 - short_weight) * long_level
    slope = short_weight * short_slope + (1.0 - short_weight) * long_slope
    scale = max(0.5 * (short_scale + long_scale), 0.24 * noise)

    # Modest causal extrapolation offsets endpoint-regression attenuation.
    level += 0.10 * slope
    return float(level), float(slope), float(scale), agreement


@dataclass
class _FilterState:
    """Latent tracker and externally visible direction-lock state."""
    latent_level: float = np.nan
    velocity: float = 0.0
    output_level: float = np.nan
    output_direction: float = 0.0
    pending_direction: float = 0.0
    pending_evidence: float = 0.0
    pending_streak: int = 0


class _EvidenceLockedTracker:
    """
    Responsive latent alpha-beta tracker with visible reversal confirmation.

    Latent state always incorporates measurements. The visible output has a
    hard monotonic lock while a reversal is uncertain, which prevents internal
    estimator jitter from becoming a false output reversal.
    """

    def __init__(self, noise):
        self.noise = max(float(noise), 1e-6)
        self.state = _FilterState()

    def _update_pending(self, direction, contribution, qualifies):
        state = self.state

        if state.pending_direction != direction:
            state.pending_direction = direction
            state.pending_evidence = 0.0
            state.pending_streak = 0

        if qualifies:
            state.pending_evidence += contribution
            state.pending_streak += 1
        else:
            # Preserve some evidence for a broad real turn, but alternating
            # jitter loses confidence rapidly.
            state.pending_evidence *= 0.42
            state.pending_streak = 0

    def update(self, measurement, slope, recent_slope, local_scale, agreement):
        """Advance tracker by one causal multi-scale measurement."""
        state = self.state
        local_scale = max(float(local_scale), 0.22 * self.noise, 1e-8)

        if not np.isfinite(measurement):
            if np.isfinite(state.output_level):
                return state.output_level
            return np.nan

        if not np.isfinite(state.latent_level):
            state.latent_level = float(measurement)
            state.velocity = float(slope) if np.isfinite(slope) else 0.0
            state.output_level = state.latent_level
            return state.output_level

        predicted = state.latent_level + state.velocity
        innovation = float(measurement - predicted)
        innovation_limit = max(3.6 * local_scale, 2.4 * self.noise)
        clipped = float(np.clip(innovation, -innovation_limit, innovation_limit))

        slope_reference = max(0.20 * local_scale, 0.10 * self.noise, 1e-8)
        motion = min(abs(slope) / slope_reference, 3.0)

        slope_agrees = (
            np.sign(slope) != 0.0
            and np.sign(recent_slope) != 0.0
            and np.sign(slope) == np.sign(recent_slope)
        )

        # Gain scheduling retains fast response on coherent movement while
        # damping the latent velocity during cross-scale disagreement.
        if slope_agrees and agreement > 0.0:
            alpha = 0.47 + 0.05 * min(motion / 2.0, 1.0)
            beta = 0.090 + 0.025 * min(motion / 2.0, 1.0)
        elif agreement > -0.20:
            alpha = 0.40
            beta = 0.064
        else:
            alpha = 0.31
            beta = 0.035

        state.latent_level = predicted + alpha * clipped
        state.velocity = (1.0 - beta) * state.velocity + beta * clipped

        desired = state.latent_level
        visible_delta = float(desired - state.output_level)
        delta_direction = float(np.sign(visible_delta))
        deadband = max(0.075 * local_scale, 0.018 * self.noise)

        if delta_direction == 0.0 or abs(visible_delta) < deadband:
            return state.output_level

        if state.output_direction == 0.0:
            state.output_level = desired
            state.output_direction = delta_direction
            return state.output_level

        if delta_direction == state.output_direction:
            state.output_level = desired
            state.pending_direction = 0.0
            state.pending_evidence = 0.0
            state.pending_streak = 0
            return state.output_level

        # Counter-trend evidence must agree across measurement innovation,
        # consensus slope, and the independent very-recent slope.
        slope_support = np.sign(slope) == delta_direction
        recent_support = np.sign(recent_slope) in (0.0, delta_direction)
        innovation_support = np.sign(clipped) in (0.0, delta_direction)
        qualifies = slope_support and recent_support and innovation_support

        normalized_move = abs(visible_delta) / max(local_scale, 0.35 * self.noise)
        contribution = min(1.45, normalized_move)
        if agreement < -0.25:
            contribution *= 0.65

        self._update_pending(delta_direction, contribution, qualifies)

        # Approximately three normalized observations are required. A run of
        # three coherent samples can also confirm a gradual low-noise turn.
        confirmed = (
            self.state.pending_evidence >= 2.85
            or self.state.pending_streak >= 3
        )

        if confirmed:
            state.output_level = desired
            state.output_direction = delta_direction
            state.pending_direction = 0.0
            state.pending_evidence = 0.0
            state.pending_streak = 0

        return state.output_level


def adaptive_filter(x, window_size=20):
    """
    Filter a 1D signal causally using robust consensus tracking.

    Args:
        x: Input one-dimensional real-valued signal.
        window_size: Required history before the first emitted estimate.

    Returns:
        Array of length len(x) - window_size + 1, endpoint-aligned.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    output_length = len(x) - window_size + 1
    if not np.any(np.isfinite(x)):
        return np.full(output_length, np.nan)
    if window_size == 1:
        return x.copy()

    noise = _global_noise_level(x)
    short_size = max(4, min(window_size, int(np.ceil(window_size * 0.55))))
    recent_size = max(3, min(short_size, 6))
    short_decay = max(1.5, 0.36 * short_size)
    long_decay = max(2.0, 0.43 * window_size)

    output = np.empty(output_length, dtype=float)
    tracker = _EvidenceLockedTracker(noise)

    for output_index, endpoint in enumerate(range(window_size - 1, len(x))):
        long_window = x[endpoint - window_size + 1:endpoint + 1]
        short_window = x[endpoint - short_size + 1:endpoint + 1]
        recent_window = x[endpoint - recent_size + 1:endpoint + 1]

        short_fit = _endpoint_model(short_window, short_decay, noise)
        long_fit = _endpoint_model(long_window, long_decay, noise)
        measurement, slope, local_scale, agreement = _consensus_measurement(
            short_fit, long_fit, noise
        )

        output[output_index] = tracker.update(
            measurement,
            slope,
            _recent_slope(recent_window, fallback=slope),
            local_scale,
            agreement,
        )

    return output


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Backward-compatible trend-preserving public entry point."""
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected signal-processing algorithm.

    Args:
        input_signal: Input time series.
        window_size: Causal history length.
        algorithm_type: "basic" or "enhanced".

    Returns:
        Endpoint-aligned filtered series.
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
