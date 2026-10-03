# EVOLVE-BLOCK-START
import numpy as np


def kalman_trend_adaptive_filter(data, window_size=32, q_level=1e-2, q_slope_slow=1e-6,
                                 q_slope_fast=0.2, trend_lookback=5, innov_gain=3.0,
                                 blend_span=None):
    """
    Adaptive two-state Kalman filter (level + slope) with trend-gated process noise
    and forward/backward zero-lag fusion.

    data: 1D numpy array of raw samples (volatile, non-stationary)
    returns: filtered output, same length as data
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    if n == 0:
        return np.copy(data)
    if n == 1:
        return np.copy(data)

    if blend_span is None:
        blend_span = max(3, (window_size // 4) * 2 + 1)  # odd, modest
    blend_span = min(blend_span, n)
    blend_half = blend_span // 2

    # ---------------- forward Kalman pass ----------------
    # State: [level, slope]. Model: level' = level + slope, slope' = slope
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])

    def run_pass(x, d, direction_forward=True):
        """Run adaptive Kalman pass over d (already oriented). Returns levels."""
        m = len(d)
        out = np.empty(m)
        P = np.eye(2) * 1.0
        slope_hist = []          # recent slope sign history for trend gating
        innov_var = 1.0          # running innovation variance estimate
        x = x.copy()

        for k in range(m):
            # --- predict ---
            x = F @ x
            P = F @ P @ F.T

            # --- adaptive process noise (trend gating) ---
            if len(slope_hist) >= trend_lookback:
                s = np.sign(slope_hist[-trend_lookback:])
                if np.all(s == s[0]) and s[0] != 0:
                    qs = q_slope_slow          # consistent trend -> stiff slope
                else:
                    qs = q_slope_fast         # mixed signs -> loose slope
            else:
                qs = q_slope_fast
            Q = np.diag([q_level, qs])
            P = P + Q

            # --- innovation ---
            y = d[k] - (H @ x)[0]
            S = (H @ P @ H.T)[0, 0] + 1.0
            # update innovation variance estimate (EMA)
            innov_var = 0.9 * innov_var + 0.1 * (y * y)
            thresh = innov_gain * np.sqrt(max(innov_var, 1e-9))

            # --- if innovation too large vs. recent scale, loosen slope ---
            if abs(y) > thresh:
                P[1, 1] += q_slope_fast * 4.0
                S = (H @ P @ H.T)[0, 0] + 1.0

            # --- update ---
            K = (P @ H.T / S).ravel()
            x = x + K * y
            P = P - np.outer(K, H @ P)
            P = 0.5 * (P + P.T)  # symmetrize

            out[k] = x[0]
            slope_hist.append(np.sign(x[1]))
            if len(slope_hist) > trend_lookback * 3:
                slope_hist.pop(0)

        if not direction_forward:
            # backward pass output must be re-reversed by caller
            pass
        return out

    x0 = np.array([data[0], 0.0])
    fwd = run_pass(x0, data, True)

    # ---------------- backward pass (for zero-lag fusion) ----------------
    # Run filter on reversed data: prediction then runs "backward in time",
    # so its lag is on the opposite side; fusing cancels lag for linear trends.
    xb0 = np.array([data[-1], 0.0])
    bwd_full = run_pass(xb0, data[::-1], False)
    bwd = bwd_full[::-1]

    # ---------------- zero-lag fusion ----------------
    # For linear trends, forward estimate at k lags by ~dt, backward by ~-dt,
    # so their average is lag-free. Weight by blend over a short span to keep
    # endpoints stable.
    fused = np.empty(n)
    for k in range(n):
        # local slope estimate for extrapolation alignment
        lo = max(0, k - blend_half)
        hi = min(n, k + blend_half + 1)
        seg_f = fwd[lo:hi]
        seg_b = bwd[lo:hi]
        # weighted average centered on k (weights ramp toward center)
        w = np.arange(len(seg_f), dtype=float)
        if len(seg_f) > 1:
            w = np.abs(w - (k - lo))
            w = 1.0 / (w + 1.0)
        else:
            w = np.array([1.0])
        fused[k] = float(np.sum(w * (0.5 * seg_f + 0.5 * seg_b)) / np.sum(w))

    # ---------------- gentle zero-lag post-smoother ----------------
    # One-pole filter applied forward and backward (cancels its own group delay),
    # moderate strength alpha chosen for noise reduction without lag.
    def zerolag_smooth(sig, alpha=0.45):
        if len(sig) < 3:
            return sig.copy()
        a = alpha
        # forward
        f = np.empty_like(sig)
        f[0] = sig[0]
        for i in range(1, len(sig)):
            f[i] = a * sig[i] + (1 - a) * f[i - 1]
        # backward
        b = np.empty_like(sig)
        b[-1] = sig[-1]
        for i in range(len(sig) - 2, -1, -1):
            b[i] = a * sig[i] + (1 - a) * b[i + 1]
        return 0.5 * (f + b)

    fused = zerolag_smooth(fused, alpha=0.45)

    return fused
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