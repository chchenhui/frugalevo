# EVOLVE-BLOCK-START
"""
Adaptive state-space filter for volatile non-stationary signals.

The enhanced path uses a robust, causal, innovation-adaptive Kalman tracker
with directional hysteresis.  Output samples remain endpoint aligned with each
trailing window and have length len(x) - window_size + 1.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Sliding-window moving-average baseline.

    Args:
        x: Input one-dimensional signal.
        window_size: Trailing window length.

    Returns:
        Moving-average output of length len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite real-valued samples")

    return np.convolve(x, np.ones(window_size, dtype=float) / window_size, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust adaptive Kalman trend filter.

    A constant-velocity state [level, velocity] is propagated causally.  The
    measurement covariance is estimated from robust trailing first differences,
    while process covariance expands during persistent innovations so genuine
    maneuvers receive a larger gain.  Huber clipping rejects isolated spikes.

    Directional hysteresis only holds weak one-window countertrend moves;
    repeated or large countertrend innovations are released promptly.  This
    specifically reduces slope changes and false reversals without imposing
    moving-average phase lag.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be one-dimensional")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    if window_size < 3:
        raise ValueError("window_size must be at least 3")
    if not np.all(np.isfinite(x)):
        raise ValueError("Input signal must contain only finite real-valued samples")

    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    n_out = windows.shape[0]

    # Estimate observation noise from local first differences.  Dividing by
    # sqrt(2) converts the scale of independent differenced white noise back
    # to an approximate per-observation scale.
    differences = np.diff(windows, axis=1)
    diff_median = np.median(differences, axis=1)
    diff_mad = np.median(
        np.abs(differences - diff_median[:, None]), axis=1
    )
    noise_scale = np.maximum(
        1.4826 * diff_mad / np.sqrt(2.0),
        np.finfo(float).eps,
    )

    # The endpoint is the causal observation.  Winsorize only extreme endpoint
    # excursions relative to its trailing neighborhood, preserving ordinary
    # high-frequency dynamics while limiting impulsive outliers.
    medians = np.median(windows, axis=1)
    endpoint = windows[:, -1]
    robust_range = np.maximum(4.5 * noise_scale, 0.18 * np.ptp(windows, axis=1))
    measurements = np.clip(
        endpoint,
        medians - robust_range,
        medians + robust_range,
    )

    # Robust local displacement is used only to tune maneuver covariance, not
    # as a fitted output.  This makes the method fundamentally state-space
    # based rather than a local regression smoother.
    recent_count = max(3, min(window_size - 1, window_size // 3 + 1))
    recent_diffs = differences[:, -recent_count:]
    drive = np.median(recent_diffs, axis=1)

    y = np.empty(n_out, dtype=float)

    level = measurements[0]
    velocity = drive[0]

    # Covariance of [level, velocity].
    s0 = max(noise_scale[0], 1e-8)
    p00 = 2.0 * s0 * s0
    p01 = 0.0
    p11 = 0.35 * s0 * s0

    y[0] = level
    committed_direction = np.sign(velocity)
    pending_direction = 0.0
    pending_count = 0

    for i in range(1, n_out):
        scale = max(noise_scale[i], 1e-8)
        observation_variance = scale * scale

        # Prediction for constant velocity.
        predicted_level = level + velocity
        predicted_velocity = velocity

        # Innovation-dependent maneuver model.  Quiet periods use very low
        # acceleration noise; repeated residual activity automatically raises
        # gain and prevents excessive lag at real changes.
        raw_innovation = measurements[i] - predicted_level
        normalized = abs(raw_innovation) / (scale + 1e-12)
        maneuver = min(normalized / 2.5, 2.0)
        q = observation_variance * (0.008 + 0.075 * maneuver * maneuver)

        pp00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pp01 = p01 + p11 + 0.50 * q
        pp11 = p11 + q

        # Huber innovation limiter: a single large observation cannot create
        # a large velocity impulse or a noise-induced directional reversal.
        innovation_limit = (2.8 + 0.7 * maneuver) * scale
        innovation = np.clip(raw_innovation, -innovation_limit, innovation_limit)

        innovation_variance = pp00 + observation_variance
        k0 = pp00 / innovation_variance
        k1 = pp01 / innovation_variance

        candidate_level = predicted_level + k0 * innovation
        candidate_velocity = predicted_velocity + k1 * innovation

        # Joseph-equivalent scalar covariance update for H=[1, 0].
        p00_new = (1.0 - k0) * pp00
        p01_new = (1.0 - k0) * pp01
        p11_new = pp11 - k1 * pp01

        proposed_step = candidate_level - level
        proposed_direction = np.sign(proposed_step)

        # A weak countertrend move must persist for two windows before it is
        # emitted.  A material residual bypasses the hold, limiting turn lag.
        direction_band = 0.32 * scale + 0.10 * abs(velocity)
        countertrend = (
            committed_direction != 0.0
            and proposed_direction != 0.0
            and proposed_direction != committed_direction
        )
        strong_turn = (
            abs(proposed_step) > 1.35 * direction_band
            or abs(raw_innovation) > 2.4 * scale
        )

        if countertrend and not strong_turn:
            if pending_direction == proposed_direction:
                pending_count += 1
            else:
                pending_direction = proposed_direction
                pending_count = 1

            if pending_count < 2:
                # Keep output continuous and damp the latent velocity toward
                # zero rather than allowing a held state to overshoot.
                candidate_level = level
                candidate_velocity = 0.55 * velocity
        else:
            pending_direction = 0.0
            pending_count = 0

        actual_step = candidate_level - level
        actual_direction = np.sign(actual_step)

        if actual_direction != 0.0 and abs(actual_step) > 0.08 * scale:
            committed_direction = actual_direction

        level = candidate_level
        velocity = candidate_velocity
        p00 = max(p00_new, 1e-14)
        p01 = p01_new
        p11 = max(p11_new, 1e-14)
        y[i] = level

    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the requested filtering algorithm.

    Args:
        input_signal: Input time series.
        window_size: Sliding window length.
        algorithm_type: "enhanced" for adaptive Kalman filtering, otherwise
            the moving-average baseline.

    Returns:
        Filtered one-dimensional signal.
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