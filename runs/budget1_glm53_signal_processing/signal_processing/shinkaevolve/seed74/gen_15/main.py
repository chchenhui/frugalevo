# EVOLVE-BLOCK-START
"""
Forward-Backward Adaptive Kalman Fusion with Inline Velocity Deadband.

Architecture:
  1. Direction-parameterized adaptive constant-velocity Kalman engine
     (single implementation, run forward and backward).
  2. Variance-weighted fusion of forward/backward levels AND velocities
     (RTS-style, zero-phase, low lag).
  3. Inline deadband on the fused velocity with compensating level
     correction (suppresses false reversals without a stacked stage).
  4. Light median polish for residual spikes.
"""
import numpy as np


def _kalman_pass(x, reverse, q_level, q_slope, r_base):
    """
    Run an adaptive two-state Kalman filter (level + velocity) over x,
    optionally reversed. Returns filtered levels and velocities aligned
    to the original time axis.
    """
    n = len(x)
    if reverse:
        xs = x[::-1]
    else:
        xs = x

    levels = np.zeros(n)
    vels = np.zeros(n)

    level = xs[0]
    vel = (xs[1] - xs[0]) if n > 1 else 0.0
    P = np.array([[r_base, 0.0], [0.0, r_base * 0.5]])

    innov_ema = r_base
    alpha_r = 0.06

    I2 = np.eye(2)
    H = np.array([[1.0, 0.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])

    for k in range(n):
        # Predict
        if k > 0:
            level = level + vel
            P = F @ P @ F.T
        # Adaptive process noise: react faster when innovations are large
        adapt = 1.0 + min(innov_ema / max(r_base, 1e-8), 6.0)
        P[0, 0] += q_level * adapt
        P[1, 1] += q_slope * adapt

        # Update with adaptive measurement noise
        innov = xs[k] - level
        S = P[0, 0] + max(r_base, 1e-8)
        K = np.array([P[0, 0] / S, P[1, 0] / S])
        level += K[0] * innov
        vel += K[1] * innov
        P = (I2 - np.outer(K, H)) @ P
        P = 0.5 * (P + P.T)

        innov_ema = (1.0 - alpha_r) * innov_ema + alpha_r * innov * innov

        levels[k] = level
        vels[k] = vel

    if reverse:
        return levels[::-1], vels[::-1] * -1.0
    return levels, vels


def _median3(a):
    """Light 3-point median polish to remove residual spikes."""
    n = len(a)
    if n < 3:
        return a.copy()
    out = a.copy()
    stacked = np.vstack([a[:-2], a[1:-1], a[2:]])
    out[1:-1] = np.median(stacked, axis=0)
    return out


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Forward-backward Kalman fusion with inline velocity deadband.

    Args:
        x: Input signal (1D array)
        window_size: Window size (defines output alignment like SMA)

    Returns:
        y: Filtered signal, length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )

    # --- Noise scale estimation (robust, from first differences) ---
    if n > 2:
        d = np.diff(x)
        mad = np.median(np.abs(d - np.median(d)))
        r_base = max(mad * mad * 0.5, 1e-8)
    else:
        r_base = 1.0

    # Tuned process noise relative to measurement noise scale
    q_level = max(r_base * 0.05, 1e-6)
    q_slope = max(r_base * 0.01, 1e-6)

    # --- Forward and backward passes (shared engine) ---
    levels_f, vels_f = _kalman_pass(x, False, q_level, q_slope, r_base)
    levels_b, vels_b = _kalman_pass(x, True, q_level, q_slope, r_base)

    # --- Variance-weighted fusion of levels and velocities ---
    # Weight each direction by inverse local residual magnitude.
    res_f = np.abs(levels_f - x)
    res_b = np.abs(levels_b - x)
    w_f = 1.0 / (res_f + r_base)
    w_b = 1.0 / (res_b + r_base)
    wsum = w_f + w_b
    w_f = w_f / wsum
    w_b = w_b / wsum

    fused_level = w_f * levels_f + w_b * levels_b
    fused_vel = w_f * vels_f + w_b * vels_b

    # --- Inline deadband on fused velocity with level correction ---
    # Suppress small slope reversals locally; nudge the level in the
    # direction of the (now stable) velocity to compensate residual lag.
    dnoise = np.std(np.diff(x))
    threshold = 0.2 * max(dnoise, 1e-12)
    dead = np.sign(fused_vel) * np.maximum(0.0, np.abs(fused_vel) - threshold)
    # Where deadband engaged, correct level slightly toward held trend
    engaged = np.abs(fused_vel) < threshold
    corr = np.zeros(n)
    if np.any(engaged):
        # Hold last non-zero deadbanded velocity (trend persistence)
        held = 0.0
        held_arr = np.zeros(n)
        for k in range(n):
            if dead[k] != 0.0:
                held = dead[k]
            held_arr[k] = held
        corr = 0.25 * held_arr * engaged
    fused_out = fused_level + corr

    # --- Median polish ---
    fused_out = _median3(fused_out)

    # --- Windowed output alignment (window endpoint anchor) ---
    output_length = n - window_size + 1
    y = fused_out[window_size - 1:]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline-compatible entry: same algorithm as enhanced."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function (same interface as original).

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: "basic" or "enhanced" (both use the new filter)

    Returns:
        Filtered signal
    """
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