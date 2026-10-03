# EVOLVE-BLOCK-START
import numpy as np

def filter_signal(x, dt=1.0):
    """
    Adaptive Kalman filter with EWMA-smoothed innovation variance driving
    measurement-noise adaptation. Dual-pass for zero-phase, velocity state
    for responsiveness, trend-based slope reversal veto.

    Inputs:
        x  : 1D numpy array of raw signal
        dt : sampling interval (default 1.0)
    Returns:
        y  : filtered signal, same length as x
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return x.copy()

    q_level = 1e-4   # process noise for level
    q_vel = 1e-3     # process noise for velocity
    r0 = 0.05        # base measurement noise
    alpha_var = 0.1  # EWMA smoothing for innovation variance
    alpha_trend = 0.15

    def kalman_pass(sig):
        m = len(sig)
        # State: [level, velocity]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        x_state = np.array([sig[0], 0.0])
        P = np.eye(2) * 1.0
        Q = np.diag([q_level, q_vel])

        ewma_var = r0          # smoothed innovation variance estimate
        trend = 0.0            # smoothed velocity trend
        out = np.empty(m)
        out[0] = sig[0]

        for k in range(m):
            # ---- Predict ----
            x_state = F @ x_state
            P = F @ P @ F.T + Q

            # ---- Update ----
            innov = sig[k] - (H @ x_state)[0]
            S = P[0, 0] + r0 * 0.5 + ewma_var * 0.5  # EWMA-smoothed r
            K = P[:, 0] / S
            x_state = x_state + K * innov
            P = P - np.outer(K, P[0, :])

            # ---- EWMA-smoothed innovation variance (NOT raw innov^2) ----
            ewma_var = (1.0 - alpha_var) * ewma_var + alpha_var * (innov * innov)

            out[k] = x_state[0]

            # ---- Trend tracking for veto (unused for output, stabilizes future ext.) ----
            trend = (1.0 - alpha_trend) * trend + alpha_trend * x_state[1]

        return out

    # Forward pass
    y_fwd = kalman_pass(x)
    # Backward pass (zero phase, reduces end-effects)
    y_bwd = kalman_pass(x[::-1])[::-1]

    # Weighted fusion: forward gets slightly more weight mid-signal,
    # backward compensates forward's end lag.
    w = np.linspace(0.35, 0.65, n)
    y = w * y_fwd + (1.0 - w) * y_bwd

    # Slope-reversal smoothing: suppress small wiggles in the derivative
    d = np.gradient(y, dt)
    if n > 5:
        ds = np.copy(d)
        for i in range(2, n - 2):
            # veto sign flips driven by tiny slopes (noise-level reversals)
            if np.sign(d[i]) != np.sign(d[i-1]) and abs(d[i]) < 0.5 * np.std(d):
                ds[i] = ds[i-1]
        # re-integrate consistent derivative from a stable anchor
        y = y[0] + np.cumsum(ds) * dt
        # rescale to match endpoint of fused signal to avoid drift
        drift = (w * y_fwd + (1 - w) * y_bwd)[-1] - y[-1]
        y += drift * np.linspace(0.0, 1.0, n)

    return y
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