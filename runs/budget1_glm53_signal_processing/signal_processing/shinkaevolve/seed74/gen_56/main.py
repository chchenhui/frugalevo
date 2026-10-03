# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Staged pipeline architecture:
  Stage 1: robust noise statistics estimation (MAD-based)
  Stage 2: RTS smoother pass 1 (adaptive R via innovation EWMA)
  Stage 3: inter-pass zero-phase bilateral damper (alpha ~ 0.65)
  Stage 4: RTS smoother pass 2 with locally adaptive block-MAD Q (refinement)
  Stage 5: guarded final zero-phase ripple damper

Output contract: y[i] corresponds to input time i + window_size - 1,
len(y) = len(x) - window_size + 1.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Stage primitives
# ---------------------------------------------------------------------------

def _bilateral_one_pole(x, alpha):
    """Forward then reversed one-pole smoothing — net zero phase delay."""
    y = np.empty_like(x)
    acc = x[0]
    y[0] = acc
    for i in range(1, len(x)):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        y[i] = acc
    acc = y[-1]
    out = y.copy()
    for i in range(len(x) - 2, -1, -1):
        acc = alpha * y[i] + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _estimate_noise_stats(x, window_size):
    """Stage 1: robust measurement/process-noise statistics."""
    n = x.size
    scale = max(float(np.std(x)), 1e-9)
    d1 = np.diff(x)
    sigma_r = np.median(np.abs(d1 - np.median(d1))) / (0.6745 * np.sqrt(2.0))
    if not np.isfinite(sigma_r) or sigma_r <= 0.0:
        sigma_r = scale
    sigma_r = float(np.clip(sigma_r, 1e-4 * scale, scale))

    w_r = int(max(3, min(window_size, n // 3)))
    if w_r % 2 == 0:
        w_r += 1
    rough = np.convolve(x, np.ones(w_r) / w_r, mode="valid")
    var_acc = 0.0
    if rough.size > 2:
        d2 = rough[2:] - 2.0 * rough[1:-1] + rough[:-2]
        mad2 = np.median(np.abs(d2 - np.median(d2))) / 0.6745
        noise_d2 = (sigma_r / np.sqrt(w_r)) * np.sqrt(6.0)
        var_acc = mad2 * mad2 - noise_d2 * noise_d2
    floor_acc = 0.05 * sigma_r
    if var_acc < floor_acc * floor_acc:
        var_acc = floor_acc * floor_acc
    sigma_acc = float(min(np.sqrt(var_acc), 0.6 * sigma_r))
    return sigma_r, sigma_acc, w_r


def _rts_smooth(z, R, qseq, Rseq=None):
    """
    Reusable RTS engine: forward constant-velocity Kalman filter + true
    backward RTS smoother. qseq may be scalar or per-sample array;
    Rseq optional per-sample measurement variance.
    Returns (smoothed_level, smoothed_velocity, smoothed_state).
    """
    n = z.size
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Qbase = np.array([[0.25, 0.5], [0.5, 1.0]])
    qarr = np.broadcast_to(np.asarray(qseq, dtype=float), (n,)) if np.isscalar(qseq) else np.asarray(qseq, dtype=float)
    if Rseq is None:
        Rseq = np.full(n, R)

    xp = np.zeros((n, 2))
    xf = np.zeros((n, 2))
    Pp = np.zeros((n, 2, 2))
    Pf = np.zeros((n, 2, 2))
    I2 = np.eye(2)

    x_est = np.array([z[0], 0.0])
    P = np.eye(2) * max(R, 1e-15)
    for k in range(n):
        if k == 0:
            xp[k] = x_est
            Pp[k] = P
        else:
            Qk = qarr[k] * Qbase
            xp[k] = F @ x_est
            Pp[k] = F @ P @ F.T + Qk
        S = Pp[k][0, 0] + Rseq[k]
        K = Pp[k][:, 0] / S
        x_est = xp[k] + K * (z[k] - xp[k, 0])
        P = (I2 - np.outer(K, H)) @ Pp[k]
        P = 0.5 * (P + P.T)
        xf[k] = x_est
        Pf[k] = P

    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        Ppred = Pp[k + 1]
        try:
            Ppred_inv = np.linalg.inv(Ppred)
        except np.linalg.LinAlgError:
            Ppred_inv = np.linalg.pinv(Ppred)
        G = Pf[k] @ F.T @ Ppred_inv
        xs[k] = xf[k] + G @ (xs[k + 1] - xp[k + 1])
    return xs[:, 0].copy(), xs[:, 1].copy(), xs


def _adaptive_R_sequence(x, n, R, qa):
    """Innovation-EWMA widen-only adaptive measurement noise sequence."""
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = qa * np.array([[0.25, 0.5], [0.5, 1.0]])
    H = np.array([1.0, 0.0])
    I2 = np.eye(2)
    Rseq = np.full(n, R)
    ewma_var = R
    x_est = np.array([x[0], 0.0])
    P = np.eye(2) * max(R, 1e-15)
    for k in range(n):
        if k > 0:
            x_est = F @ x_est
            P = F @ P @ F.T + Q
        innov = x[k] - x_est[0]
        Rk = max(R, 0.7 * R + 0.3 * ewma_var)
        S = P[0, 0] + Rk
        K = P[:, 0] / S
        x_est = x_est + K * innov
        P = (I2 - np.outer(K, H)) @ P
        P = 0.5 * (P + P.T)
        Rseq[k] = Rk
        ewma_var = 0.9 * ewma_var + 0.1 * innov * innov
        ewma_var = float(np.clip(ewma_var, 1e-4 * R, 100.0 * R))
    return Rseq


def _block_mad_qseq(y, n, w_r, sigma_r, qa):
    """Locally adaptive process noise via robust block-MAD second differences."""
    Qs = []
    centers = []
    step = max(1, w_r // 2)
    for start in range(0, max(1, n - w_r + 1), step):
        seg = y[start:start + w_r]
        if seg.size < 3:
            continue
        d2s = seg[2:] - 2.0 * seg[1:-1] + seg[:-2]
        mad_b = np.median(np.abs(d2s - np.median(d2s))) / 0.6745
        noise_d2_b = (sigma_r / np.sqrt(w_r)) * np.sqrt(6.0)
        var_b = mad_b * mad_b - noise_d2_b * noise_d2_b
        s_b = float(np.clip(np.sqrt(max(var_b, 0.0)), 0.05 * sigma_r, 1.5 * sigma_r))
        Qs.append(s_b)
        centers.append(start + w_r // 2)
    if len(Qs) >= 2:
        return np.interp(
            np.arange(n, dtype=float),
            np.asarray(centers, dtype=float),
            np.asarray(Qs, dtype=float),
        ) ** 2
    if len(Qs) == 1:
        return np.full(n, Qs[0] ** 2)
    return np.full(n, qa)


def _guarded_final_damper(y, sigma_r):
    """Stage 5: adaptive-strength bilateral damper with correlation guard."""
    dy = np.diff(y)
    r_std = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0)) if dy.size > 2 else 0.0
    if r_std <= 0:
        return y
    alpha = float(np.clip(0.55 + 0.4 * np.tanh(2.0 * sigma_r / max(r_std, 1e-12) - 2.0), 0.40, 0.95))
    if alpha >= 0.999:
        return y
    if alpha < 0.65:
        y_damped = _bilateral_one_pole(_bilateral_one_pole(y, 0.65), 0.65)
    else:
        y_damped = _bilateral_one_pole(y, alpha)
    # Correlation guard against over-flattening
    v1 = y - y.mean()
    v2 = y_damped - y_damped.mean()
    denom = np.sqrt(float(v1 @ v1) * float(v2 @ v2))
    corr = float(v1 @ v2) / denom if denom > 0 else 1.0
    if not np.isfinite(corr):
        corr = 1.0
    if corr < 0.97 and alpha < 0.85:
        frac = float(np.clip((0.97 - corr) / 0.10, 0.0, 1.0))
        alpha_eff = alpha + frac * (0.85 - alpha)
        y_damped = _bilateral_one_pole(y, alpha_eff)
    return y_damped


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def kalman_rts_filter(x, window_size=20):
    """Full staged pipeline. Output index i corresponds to input time
    i + window_size - 1."""
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n == 1:
        return x.copy()

    # Stage 1: robust statistics
    sigma_r, sigma_acc, w_r = _estimate_noise_stats(x, window_size)
    R = sigma_r * sigma_r
    qa = sigma_acc * sigma_acc

    # Stage 2: first RTS pass with adaptive (widen-only) R
    Rseq = _adaptive_R_sequence(x, n, R, qa)
    y, _, _ = _rts_smooth(x, R, qa, Rseq=Rseq)

    # Stage 3: inter-pass moderate bilateral damper — removes ripple BEFORE
    # the refinement pass; RTS re-sharpens corners the damper rounded while
    # the removed ripple stays gone.
    y = _bilateral_one_pole(y, 0.65)

    # Stage 4: refinement RTS pass with locally adaptive block-MAD Q (0.25x)
    qseq = _block_mad_qseq(y, n, w_r, sigma_r, qa)
    y_ref, _, _ = _rts_smooth(y, R, 0.25 * qseq)

    # Dynamics-aware blend: heavy smoothing when signal is quiet,
    # lighter refinement (less lag) when dynamics are strong.
    if sigma_acc > 1e-12 and sigma_r > 1e-12:
        dyn_ratio = sigma_acc / sigma_r
        w_refine = float(np.clip(1.25 - 1.25 * dyn_ratio, 0.25, 1.0))
    else:
        w_refine = 1.0
    y = w_refine * y_ref + (1.0 - w_refine) * y

    # Stage 5: guarded final damper
    y = _guarded_final_damper(y, sigma_r)

    # Slice to output contract
    y = y[window_size - 1:].copy()
    expected = n - window_size + 1
    if y.shape[0] != expected:
        y = y[:expected]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline entry — routes to the staged pipeline."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Enhanced entry — staged RTS/damper pipeline."""
    return kalman_rts_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic", "enhanced", or "rts")

    Returns:
        Filtered signal
    """
    return kalman_rts_filter(input_signal, window_size)

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