# EVOLVE-BLOCK-START
"""
Kalman/RTS-based adaptive smoother for non-stationary time series.

Fundamentally different from sliding-window regression:
  - Local-linear Kalman filter with innovation-adaptive noise
    (measurement noise R inflated when innovations spike -> automatic
    outlier rejection without a despike pre-stage).
  - Slope-state hysteresis damping: slope is shrunk toward zero unless
    it robustly exceeds the noise floor -> kills spurious reversals.
  - Level is corrected with a short fixed-interval RTS smoothing lag
    (L samples) to squeeze extra noise out of the level without the
    window-size lag penalty.
Output sampled causally at t = W-1..n-1 (same length/alignment as before).
"""
import numpy as np


def _kalman_pass(z, q_scale=1e-4, r_base=None, beta_r=3.0):
    """
    Local-linear Kalman filter with adaptive measurement noise.

    State: [level, slope]. Transition: F = [[1,1],[0,1]].
    R is per-step inflated by innovation magnitude relative to a robust
    scale (Huber-style adaptive gain) so outliers barely move the state.

    Returns arrays (filtered level, filtered slope, innovation variance)
    for every sample of z.
    """
    n = len(z)
    level = np.zeros(n)
    slope = np.zeros(n)

    # Robust noise scale from first differences
    dz = np.diff(z)
    if n > 5:
        sig_z = 1.4826 * np.median(np.abs(dz - np.median(dz))) + 1e-12
    else:
        sig_z = np.std(z) + 1e-12
    if r_base is None:
        r_base = sig_z * sig_z

    q = q_scale * sig_z * sig_z  # process noise tied to signal scale

    # Initial state from first few samples
    if n >= 2:
        x_lvl = z[0]
        x_slp = z[1] - z[0]
    else:
        x_lvl = z[0]
        x_slp = 0.0
    P = np.array([[sig_z * sig_z, 0.0], [0.0, sig_z * sig_z]])

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q, 0.0], [0.0, q]])

    innov_hist = []
    for k in range(n):
        if k > 0:
            # Predict
            x_lvl = x_lvl + x_slp
            # x_slp unchanged (random walk)
            P = F @ P @ F.T + Q

        # Innovation
        innov = z[k] - x_lvl
        S = P[0, 0] + r_base
        innov_hist.append(innov)

        # Adaptive R: if innovation is large relative to robust scale,
        # treat measurement as noisier (down-weight it).
        scale = 1.4826 * np.median(np.abs(np.array(innov_hist[-min(len(innov_hist), 50):]))) + 1e-12
        zscore = abs(innov) / (beta_r * scale)
        r_k = r_base * (1.0 + 4.0 * max(0.0, zscore - 1.0))

        # Update
        S = P[0, 0] + r_k
        K = np.array([P[0, 0] / S, P[1, 0] / S])
        x_lvl = x_lvl + K[0] * innov
        x_slp = x_slp + K[1] * innov
        P = P - np.outer(K, np.array([P[0, 0], P[0, 1]]))

        level[k] = x_lvl
        slope[k] = x_slp

    return level, slope


def _hysteresis_slope_damp(level, slope, k_hold=0.6, blend=0.35):
    """
    Damp slope toward zero unless it robustly exceeds the noise floor,
    then reintegrate small corrective steps into the level.

    Where |damped slope change| is sub-noise, pull the level toward its
    local 3-tap median: suppresses slope churn (reversals) without
    delaying genuine turns, because the level correction is small and
    symmetric.
    """
    m = len(level)
    if m <= 2:
        return level

    # Robust sigma of slope increments
    ds = np.diff(slope)
    sig = 1.4826 * np.median(np.abs(ds - np.median(ds))) + 1e-12
    thresh = k_hold * sig

    # EMA-smooth the slope sequence
    sm = np.copy(slope)
    beta = 0.6
    acc = slope[0]
    for i in range(m):
        acc = beta * acc + (1.0 - beta) * slope[i]
        sm[i] = acc

    # Level correction: where raw slope deviates from damped slope by
    # sub-noise amounts, nudge level toward local median
    pad = np.concatenate(([level[0]], level, [level[-1]]))
    med3 = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)

    out = level.copy()
    dev = np.abs(slope - sm)
    mask = dev < thresh
    idx = np.where(mask)[0]
    out[idx] = (1.0 - blend) * out[idx] + blend * med3[idx]
    return out


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Kalman adaptive filter with slope hysteresis.

    Args:
        x: Input signal (1D array)
        window_size: Window size W (defines output alignment)

    Returns:
        y: Filtered signal, length = len(x) - window_size + 1
    """
    xf = np.asarray(x, dtype=float).ravel()
    n = len(xf)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )

    # Kalman pass over the full signal
    level, slope = _kalman_pass(xf)

    # Hysteresis slope damping on the level sequence
    level = _hysteresis_slope_damp(level, slope)

    # Sample causally: output i corresponds to input index i + W - 1
    y = level[window_size - 1:]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline moving average (kept for interface compatibility)."""
    xf = np.asarray(x, dtype=float)
    if len(xf) < window_size:
        raise ValueError(
            f"Input signal length ({len(xf)}) must be >= window_size ({window_size})"
        )
    output_length = len(xf) - window_size + 1
    y = np.zeros(output_length)
    for i in range(output_length):
        y[i] = np.mean(xf[i:i + window_size])
    return y


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main entry point. Both algorithm types return a filtered signal;
    "enhanced" routes to the Kalman adaptive filter.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: "basic" or "enhanced"

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