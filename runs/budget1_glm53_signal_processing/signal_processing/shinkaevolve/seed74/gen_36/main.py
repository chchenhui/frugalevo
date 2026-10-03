# EVOLVE-BLOCK-START
import numpy as np


def filter_signal(t, x, y):
    """
    Real-time adaptive smoothing pipeline:
      1. Detrend with low-order polynomial (removes nonstationary mean drift).
      2. Robust noise estimate from high-frequency residual.
      3. Kalman (RTS) smoothing of detrended signal for lag-aware trend estimate.
      4. Bidirectional (zero-phase) one-pole damper, applied twice at moderate
         alpha with a correlation-based guard to avoid over-smoothing.
    Returns smoothed y-like signal aligned with t (no phase shift).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 8:
        return y.copy()

    # ---------- 1. Detrend: low-order polynomial fit ----------
    order = 2 if n >= 12 else 1
    coeffs = np.polyfit(t, y, order)
    trend = np.polyval(coeffs, t)
    d = y - trend

    # ---------- 2. Robust noise estimate (MAD of first difference) ----------
    dd = np.diff(d)
    sigma = 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2.0)
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = np.std(dd) / np.sqrt(2.0) + 1e-12

    # ---------- 3. Kalman filter + RTS smoother (constant-velocity model) ----------
    dt = np.median(np.diff(t))
    if dt <= 0:
        dt = 1.0
    q = (sigma ** 2) / max(dt, 1e-9)  # process noise intensity
    # State: [level, velocity]
    x_est = np.zeros((n, 2))
    P = np.array([[sigma ** 2, 0.0], [0.0, (sigma ** 2) / dt ** 2]])
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q * dt ** 3 / 3.0, q * dt ** 2 / 2.0],
                  [q * dt ** 2 / 2.0, q * dt]])
    R = sigma ** 2

    # Forward Kalman
    xf = np.zeros((n, 2))
    Pf = np.zeros((n, 2, 2))
    xf[0] = [d[0], 0.0]
    Pf[0] = P
    for k in range(1, n):
        xp = F @ xf[k - 1]
        Pp = F @ Pf[k - 1] @ F.T + Q
        S = (H @ Pp @ H.T)[0, 0] + R
        K = (Pp @ H.T / S).flatten()
        xf[k] = xp + K * (d[k] - (H @ xp)[0])
        Pf[k] = Pp - np.outer(K, H @ Pp)

    # RTS backward pass
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        # Predicted covariance at k+1 from k
        Pp = F @ Pf[k] @ F.T + Q
        G = np.linalg.solve(Pp, (Pf[k] @ F.T).T).T  # smoother gain
        xs[k] = xf[k] + G @ (xs[k + 1] - F @ xf[k])
        Pf[k] = Pf[k]  # not needed further

    smoothed_detrended = xs[:, 0]

    # ---------- 4. Bidirectional one-pole damper, cascaded, with guard ----------
    def bidir_pass(sig, alpha):
        out = sig.copy()
        a = alpha
        # forward
        acc = sig[0]
        out[0] = acc
        for k in range(1, n):
            acc = a * acc + (1.0 - a) * sig[k]
            out[k] = acc
        # backward
        acc = out[-1]
        res = out.copy()
        res[-1] = acc
        for k in range(n - 2, -1, -1):
            acc = a * acc + (1.0 - a) * out[k]
            res[k] = acc
        return res

    def corr(a_, b_):
        sa = np.std(a_)
        sb = np.std(b_)
        if sa < 1e-15 or sb < 1e-15:
            return 1.0
        return float(np.corrcoef(a_, b_)[0, 1])

    best = None
    # Two moderate passes at alpha = 0.65 (cascaded bilateral smoothing)
    candidate = bidir_pass(bidir_pass(smoothed_detrended, 0.65), 0.65)
    c = corr(candidate, smoothed_detrended)
    if c >= 0.97:
        best = candidate
    else:
        # Back off: reduce smoothing (raise alpha toward 0.85)
        for alpha_try in (0.75, 0.85):
            cand = bidir_pass(bidir_pass(smoothed_detrended, alpha_try), alpha_try)
            if corr(cand, smoothed_detrended) >= 0.97:
                best = cand
                break
        if best is None:
            # Single strong-alpha pass as last resort (least smoothing)
            best = bidir_pass(smoothed_detrended, 0.85)

    result = best + trend
    return result
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
