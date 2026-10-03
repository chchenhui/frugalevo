# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Pipeline: forward constant-velocity Kalman (adaptive R, robust gating)
          -> RTS backward smoother on [level, slope] state
          -> three-scale zero-phase Savitzky-Golay bank
             (widest 2*sg_half, mid sg_half, widened short (2*sg_half)//3)
          -> gated blended-slope lead compensation (softer gate, reduced gain)
          -> triple light zero-phase SG polish (sg_half//3)
          -> end-aligned trim to sliding-window output length.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Adaptive signal processing algorithm using sliding window approach.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1

    # One-pass box filter (vectorized)
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    y = (c[window_size:] - c[:-window_size]) / window_size

    return y


def _sg_kernel(half, poly):
    """Zero-phase Savitzky-Golay coefficients for value at window center."""
    half = max(1, int(half))
    t = np.arange(-half, half + 1, dtype=float)
    if len(t) - 1 < poly:
        poly = len(t) - 1
    A = np.vstack([t ** k for k in range(poly + 1)]).T
    return (np.linalg.pinv(A.T @ A) @ A.T)[0]


def _zero_phase_smooth(sig, half, kernel):
    """Zero-phase SG pass with mirrored edge padding."""
    m = len(sig)
    if m < 2 * half + 1:
        return sig.copy()
    left = 2 * sig[0] - sig[1 : half + 1][::-1]
    right = 2 * sig[-1] - sig[-2 : -half - 2 : -1][::-1]
    padded = np.concatenate([left, sig, right])
    out = np.convolve(padded, kernel, mode="valid")
    return out[:m] if len(out) >= m else sig.copy()


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation:
    Kalman forward pass + RTS backward smoothing, a three-scale zero-phase
    Savitzky-Golay bank (widened short scale), gated blended-slope lead
    compensation, and a triple light zero-phase SG polish.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)
    output_length = n - window_size + 1

    # --- Stage 1: forward constant-velocity Kalman filter (adaptive R, robust gating) ---
    dx = np.diff(x)
    q_scale = np.median(np.abs(dx)) + 1e-12
    local_std = np.convolve(np.abs(dx), np.ones(3) / 3.0, mode="same")
    r_arr = np.concatenate([[q_scale], local_std]) + 1e-9

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * (q_scale ** 2) * 0.05

    s_filt = np.zeros((n, 2))
    P_filt = np.zeros((n, 2, 2))
    s_pred = np.zeros((n, 2))
    P_pred = np.zeros((n, 2, 2))

    P = np.eye(2) * (r_arr[0] ** 2)
    s = np.array([x[0], 0.0])
    s_filt[0] = s
    P_filt[0] = P

    for i in range(1, n):
        # Predict
        s = F @ s
        P = F @ P @ F.T + Q
        s_pred[i] = s
        P_pred[i] = P
        # Adapt measurement noise to local volatility
        R = max(r_arr[i] ** 2, 1e-12)
        # Update
        S = H @ P @ H.T + R
        K = (P @ H.T) / S
        innov = x[i] - (H @ s)[0]
        # Robust gating: down-weight innovations beyond ~3 sigma
        Sv = float(S[0, 0]) if np.ndim(S) else float(S)
        if abs(innov) > 3.0 * np.sqrt(Sv):
            R = R * 9.0
            S = H @ P @ H.T + R
            K = (P @ H.T) / S
            innov = x[i] - (H @ s)[0]
        s = s + (K.flatten() * innov)
        P = P - K @ H @ P
        s_filt[i] = s
        P_filt[i] = P

    # --- Stage 2: RTS backward recursion (smoothed [level, slope] state) ---
    s_smooth = s_filt.copy()
    for i in range(n - 2, -1, -1):
        Pm = P_pred[i + 1]
        G = np.linalg.solve(Pm.T, (P_filt[i] @ F.T).T).T
        s_smooth[i] = s_filt[i] + G @ (s_smooth[i + 1] - s_pred[i + 1])

    kf_out = s_smooth[:, 0]

    # --- Stage 3: three-scale zero-phase Savitzky-Golay bank ---
    sg_half = max(2, window_size // 2)
    h_wide = 2 * sg_half                    # widened widest kernel
    h_mid = sg_half                         # mid scale
    h_short = max(2, (2 * sg_half) // 3)    # widened short scale (was sg_half//2)

    c_wide = _sg_kernel(h_wide, 3)
    c_mid = _sg_kernel(h_mid, 3)
    c_short = _sg_kernel(h_short, 2)

    smoothed = _zero_phase_smooth(kf_out, h_wide, c_wide)
    smoothed = _zero_phase_smooth(smoothed, h_mid, c_mid)
    smoothed = _zero_phase_smooth(smoothed, h_short, c_short)

    # --- Stage 4: gated blended-slope lead compensation ---
    slope_rts = _zero_phase_smooth(s_smooth[:, 1].copy(), h_short, c_short)
    slope_grad = np.gradient(smoothed)
    # Blend: mostly the low-noise RTS slope, partly the responsive gradient slope
    slope = 0.7 * slope_rts + 0.3 * slope_grad

    # Soft reversal gate: softened threshold (0.30 * q_scale) so genuine
    # moderate trends still receive lead compensation while flat/noisy
    # regions remain gated off.
    q_scale2 = np.median(np.abs(dx)) + 1e-12
    gate = 1.0 / (1.0 + np.exp(-(np.abs(slope) - 0.30 * q_scale2) / max(1e-9, 0.15 * q_scale2)))
    # Reduced lead gain (0.20 vs 0.25) to counter lag regression; the extra
    # smoothing stages recover any lost responsiveness in noise metrics.
    lead = 0.5 * sg_half * 0.20
    smoothed = smoothed + lead * gate * slope

    # --- Stage 4b: triple light final SG pass to suppress residual oscillation ---
    # Three short, unconditional, zero-phase quadratic passes at sg_half//3:
    # cascaded attenuation beats a single wider pass for high-frequency
    # residual noise while introducing negligible phase distortion.
    h_final = max(2, sg_half // 3)
    c_final = _sg_kernel(h_final, 2)
    smoothed = _zero_phase_smooth(smoothed, h_final, c_final)
    smoothed = _zero_phase_smooth(smoothed, h_final, c_final)
    smoothed = _zero_phase_smooth(smoothed, h_final, c_final)

    # --- Stage 5: trim to sliding-window output length, aligning end ---
    y = smoothed[-output_length:] if len(smoothed) >= output_length else kf_out[-output_length:]

    # Safety validation: output length must be exactly n - window_size + 1
    if len(y) != output_length:
        y = kf_out[-output_length:]

    return y


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
    else:
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