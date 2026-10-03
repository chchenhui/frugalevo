# EVOLVE-BLOCK-START
"""
Causal robust alpha-beta trend tracker for volatile non-stationary signals.

The implementation produces one estimate per complete input window, aligned to
the newest sample in that window.  It uses a lightweight state-space tracker
rather than repeatedly blending endpoint level fits.  Sliding-window statistics
are used to estimate local uncertainty and to adapt the internal tracker gains.
"""
import numpy as np


def _robust_scale(values, fallback=1e-6):
    """Return a finite MAD-based robust scale estimate."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))

    if not np.isfinite(scale) or scale < 1e-10:
        scale = np.std(values)

    if not np.isfinite(scale) or scale < 1e-10:
        scale = max(float(fallback), 1e-6)

    return float(scale)


def _finite_differences(values):
    """Return adjacent finite differences without bridging missing samples."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return np.empty(0, dtype=float)

    valid = np.isfinite(values[:-1]) & np.isfinite(values[1:])
    return values[1:][valid] - values[:-1][valid]


def _endpoint_slope(values, decay=4.0):
    """
    Estimate a causal endpoint slope using exponentially weighted centered data.

    Only slope information is used by the state tracker.  This intentionally
    avoids using a second independently smoothed level estimate, keeping the
    state-space update as the sole output mechanism.
    """
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if np.count_nonzero(valid) < 2:
        return 0.0

    length = values.size
    t_all = np.arange(length, dtype=float) - float(length - 1)
    t = t_all[valid]
    z = values[valid]

    tau = max(float(decay), 1.0)
    weights = np.exp(t / tau)
    weights /= max(np.sum(weights), 1e-12)

    t_mean = np.sum(weights * t)
    z_mean = np.sum(weights * z)
    dt = t - t_mean

    denominator = np.sum(weights * dt * dt)
    if denominator < 1e-12:
        return 0.0

    slope = np.sum(weights * dt * (z - z_mean)) / denominator
    return float(slope) if np.isfinite(slope) else 0.0


def adaptive_filter(x, window_size=20):
    """
    Adaptive causal robust alpha-beta filtering.

    Args:
        x: Input one-dimensional real-valued signal.
        window_size: Required history before emitting the first output.

    Returns:
        Filtered endpoint estimates of length len(x) - window_size + 1.
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

    n = len(x)
    output_length = n - window_size + 1

    if not np.any(np.isfinite(x)):
        return np.full(output_length, np.nan)

    if window_size == 1:
        return x.copy()

    finite_x = x[np.isfinite(x)]
    global_diff = _finite_differences(finite_x)
    if global_diff.size:
        global_noise = _robust_scale(global_diff, fallback=1e-5) / np.sqrt(2.0)
    else:
        global_noise = _robust_scale(finite_x, fallback=1e-5)

    global_noise = max(float(global_noise), 1e-6)

    # Short and medium derivative horizons provide agreement evidence.  They
    # modulate state gains only; the reported estimate remains the tracker state.
    short_span = max(4, min(7, window_size))
    long_span = max(short_span + 1, min(window_size, 12))

    level = np.nan
    velocity = 0.0
    previous_short_sign = 0.0
    opposing_count = 0
    y = np.empty(output_length, dtype=float)

    for index in range(n):
        start = max(0, index - window_size + 1)
        history = x[start:index + 1]

        short_history = x[max(0, index - short_span + 1):index + 1]
        long_history = x[max(0, index - long_span + 1):index + 1]

        local_diff = _finite_differences(history)
        if local_diff.size >= 3:
            local_noise = _robust_scale(local_diff, fallback=global_noise) / np.sqrt(2.0)
        else:
            local_noise = global_noise

        local_noise = max(
            float(local_noise),
            0.30 * global_noise,
            1e-8,
        )

        short_slope = _endpoint_slope(short_history, decay=max(1.5, 0.45 * short_span))
        long_slope = _endpoint_slope(long_history, decay=max(2.0, 0.58 * long_span))

        slope_floor = max(0.06 * local_noise, 1e-10)
        short_sign = np.sign(short_slope) if abs(short_slope) > slope_floor else 0.0
        long_sign = np.sign(long_slope) if abs(long_slope) > slope_floor else 0.0

        slope_strength = (
            abs(short_slope) + abs(long_slope)
        ) / max(2.0 * local_noise, 1e-10)

        if short_sign != 0.0 and long_sign != 0.0:
            agreement = 1.0 if short_sign == long_sign else -1.0
        elif short_sign != 0.0 or long_sign != 0.0:
            agreement = 0.20
        else:
            agreement = 0.0

        observation = x[index]

        if not np.isfinite(level):
            if np.isfinite(observation):
                level = float(observation)
                velocity = 0.0
            if index >= window_size - 1:
                y[index - window_size + 1] = level
            continue

        predicted_level = level + velocity
        predicted_velocity = velocity

        if np.isfinite(observation):
            residual = float(observation - predicted_level)
            normalized_residual = abs(residual) / max(local_noise, 1e-10)

            # Soft clipping rejects isolated spikes while retaining genuine
            # large changes once they persist across subsequent samples.
            huber_limit = 3.25 * local_noise
            innovation = float(np.clip(residual, -huber_limit, huber_limit))

            # Gain scheduling: dynamic agreement raises response, disagreement
            # protects velocity persistence and suppresses false reversals.
            motion = normalized_residual / (1.0 + normalized_residual)
            alpha = 0.20 + 0.16 * motion
            beta = 0.020 + 0.050 * motion

            if agreement > 0.0:
                alpha += 0.10 * min(1.0, slope_strength / 1.5)
                beta += 0.075 * min(1.0, slope_strength / 1.2)
            elif agreement < 0.0:
                alpha *= 0.76
                beta *= 0.28

            # A direction change needs repeated recent evidence unless its
            # innovation is clearly stronger than local noise.
            velocity_sign = np.sign(predicted_velocity)
            opposing_short = (
                velocity_sign != 0.0
                and short_sign != 0.0
                and short_sign != velocity_sign
            )

            if opposing_short:
                opposing_count += 1
            else:
                opposing_count = max(0, opposing_count - 1)

            weak_opposition = (
                opposing_short
                and opposing_count < 2
                and abs(innovation) < 1.45 * local_noise
            )

            if weak_opposition:
                alpha *= 0.42
                beta *= 0.16

            alpha = float(np.clip(alpha, 0.08, 0.56))
            beta = float(np.clip(beta, 0.006, 0.18))

            updated_level = predicted_level + alpha * innovation
            updated_velocity = predicted_velocity + beta * innovation

            # Velocity hysteresis avoids a one-sample sign flip.  This acts on
            # the latent trend state, so it does not hard-quantize the output.
            if (
                velocity_sign != 0.0
                and np.sign(updated_velocity) != 0.0
                and np.sign(updated_velocity) != velocity_sign
                and opposing_count < 2
                and abs(updated_velocity) < max(
                    0.12 * local_noise,
                    0.55 * abs(predicted_velocity),
                )
            ):
                updated_velocity = 0.55 * predicted_velocity

            level = float(updated_level)
            velocity = float(updated_velocity)
        else:
            # Missing data is propagated causally by prediction only, with
            # mild velocity decay to avoid indefinitely extrapolating a trend.
            level = float(predicted_level)
            velocity = float(0.985 * predicted_velocity)
            opposing_count = 0

        previous_short_sign = short_sign

        if index >= window_size - 1:
            y[index - window_size + 1] = level

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Backward-compatible trend-preserving public entry point."""
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected signal-processing algorithm.

    Args:
        input_signal: Input time series data.
        window_size: Sliding history size.
        algorithm_type: "basic" or "enhanced".

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
