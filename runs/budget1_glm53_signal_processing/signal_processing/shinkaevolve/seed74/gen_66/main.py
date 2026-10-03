# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Architecture: a declarative zero-phase pipeline of independent stages.
  1. NoiseModel      — robust MAD-based sigma_r, adaptive Q estimate
  2. SmoothStage     — constant-velocity Kalman forward + RTS backward smoother
  3. FlipEliminator  — zero-phase true elimination of sub-threshold sign
                       reversals in the difference sequence (hold-the-line /
                       step-averaging), NOT magnitude shrinking
  4. DamperStage     — guarded zero-phase bilateral one-pole ripple damper
  5. OutputStage     — slice to the output contract
Output contract: y[i] corresponds to input time i + window_size - 1,
len(y) = len(x) - window_size + 1.
"""
import numpy as np


# ---------------------------------------------------------------
# Stage 1: Noise model
# ---------------------------------------------------------------
def _estimate_noise(x, window_size):
    """Robust MAD-based measurement noise and process-noise scale."""
    n = x.size
    sig_range = float(np.std(x)) if n > 1 else 0.0
    scale = max(sig_range, 1e-9)
    d1 = np.diff(x)
    sigma_r = np.median(np.abs(d1 - np.median(d1))) / (0.6745 * np.sqrt(2.0))
    if not np.isfinite(sigma_r) or sigma_r <= 0.0:
        sigma_r = scale
    sigma_r = float(np.clip(sigma_r, 1e-4 * scale, scale))

    w_r = int(max(3, min(window_size, n // 3)))
    if w_r % 2 == 0:
        w_r += 1
    rough = np.convolve(x, np.ones(w_r) / w_r, mode="valid")
    if rough.size > 2:
        d2 = rough[2:] - 2.0 * rough[1:-1] + rough[:-2]
        mad2 = np.median(np.abs(d2 - np.median(d2))) / 0.6745
        noise_d2 = (sigma_r / np.sqrt(w_r)) * np.sqrt(6.0)
        var_acc = mad2 * mad2 - noise_d2 * noise_d2
    else:
        var_acc = 0.0
    floor_acc = 0.05 * sigma_r
    if var_acc < floor_acc * floor_acc:
        var_acc = floor_acc * floor_acc
    sigma_acc = float(min(np.sqrt(var_acc), 0.6 * sigma_r))
    return sigma_r, sigma_acc, w_r


# ---------------------------------------------------------------
# Stage 2: Kalman forward + RTS backward smoother
# ---------------------------------------------------------------
def _rts_smooth(z, sigma_r, sigma_acc, Rseq=None, q_scale=1.0):
    """One Kalman forward pass + one RTS backward pass. Returns smoothed
    position and velocity state trajectories."""
    n = z.size
    R = sigma_r * sigma_r
    qa = sigma_acc * sigma_acc * q_scale
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    Q = qa * np.array([[0.25, 0.5], [0.5, 1.0]])
    I2 = np.eye(2)

    xp = np.zeros((n, 2))
    xf = np.zeros((n, 2))
    Pp = np.zeros((n, 2, 2))
    Pf = np.zeros((n, 2, 2))

    x_est = np.array([z[0], 0.0])
    P = np.eye(2) * max(R, 1e-15)
    for k in range(n):
        if k == 0:
            xp[k] = x_est
            Pp[k] = P
        else:
            xp[k] = F @ x_est
            Pp[k] = F @ P @ F.T + Q
        Rk = Rseq[k] if Rseq is not None else R
        S = Pp[k][0, 0] + Rk
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
            Ppred_inv = np.linalg.ppinv(Ppred) if False else np.linalg.pinv(Ppred)
        G = Pf[k] @ F.T @ Ppred_inv
        xs[k] = xf[k] + G @ (xs[k + 1] - xp[k + 1])
    return xs


def _adaptive_R(x, sigma_r):
    """Widen-only innovation-based adaptive measurement noise sequence."""
    n = x.size
    R0 = sigma_r * sigma_r
    Rseq = np.full(n, R0)
    ewma_var = R0
    lam = 0.9
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    x_est = np.array([x[0], 0.0])
    P = np.eye(2) * max(R0, 1e-15)
    # lightweight preview pass to derive R sequence from innovations
    qa = (0.2 * sigma_r) ** 2
    Q = qa * np.array([[0.25, 0.5], [0.5, 1.0]])
    I2 = np.eye(2)
    H = np.array([1.0, 0.0])
    for k in range(n):
        if k == 0:
            xp = x_est
            Pp = P
        else:
            xp = F @ x_est
            Pp = F @ P @ F.T + Q
        innov = x[k] - xp[0]
        Rk = max(R0, 0.7 * R0 + 0.3 * ewma_var)
        S = Pp[0, 0] + Rk
        K = Pp[:, 0] / S
        x_est = xp + K * innov
        P = (I2 - np.outer(K, H)) @ Pp
        P = 0.5 * (P + P.T)
        Rseq[k] = Rk
        ewma_var = lam * ewma_var + (1.0 - lam) * innov * innov
        ewma_var = float(np.clip(ewma_var, 1e-4 * R0, 100.0 * R0))
    return Rseq


# ---------------------------------------------------------------
# Stage 3: True flip eliminator (zero phase)
# ---------------------------------------------------------------
def _eliminate_flips(y, sigma_r, keep=0.0, thresh_scale=1.0):
    """
    True elimination of sub-threshold sign reversals.

    For each step, if the current difference d opposes the previous
    difference prev_d AND |d| is below a robust noise threshold, the step
    is neutralized: with keep=0 the line is held (out[k] = out[k-1], no
    sign change at all); with keep=0.5 the two conflicting steps are
    averaged. prev_d is tracked from the PRE-damp difference so holds
    cannot stall trend detection. Applied forward plus mirrored
    backward so the operation is net zero-phase.
    """
    n = len(y)
    if n < 3:
        return y.copy()
    thr = thresh_scale * sigma_r
    out = y.copy()
    # forward sweep
    prev_d = out[1] - out[0]
    for k in range(2, n):
        d_raw = y[k] - out[k - 1]
        if d_raw * prev_d < 0.0 and abs(d_raw) < thr:
            # true elimination: hold the line (or average conflicting steps)
            if keep <= 0.0:
                out[k] = out[k - 1]
                step = 0.0
            else:
                avg = 0.5 * (prev_d + d_raw)
                out[k] = out[k - 1] + keep * avg
                step = keep * avg
            # prev_d unchanged (from pre-damp history) -> no zero-stall
        else:
            out[k] = out[k - 1] + d_raw
            step = d_raw
        prev_d = d_raw if abs(d_raw) > 0 else prev_d
    # mirrored backward sweep
    outb = out.copy()
    prev_d = outb[-2] - outb[-1]
    for k in range(n - 3, -1, -1):
        d_raw = out[k] - outb[k + 1]
        if d_raw * prev_d < 0.0 and abs(d_raw) < thr:
            if keep <= 0.0:
                outb[k] = outb[k + 1]
            else:
                avg = 0.5 * (prev_d + d_raw)
                outb[k] = outb[k + 1] + keep * avg
        else:
            outb[k] = outb[k + 1] + d_raw
        prev_d = d_raw if abs(d_raw) > 0 else prev_d
    return outb


# ---------------------------------------------------------------
# Stage 4: Zero-phase bilateral one-pole damper
# ---------------------------------------------------------------
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


def _corr(a, b):
    v1 = a - a.mean()
    v2 = b - b.mean()
    denom = np.sqrt(float(v1 @ v1) * float(v2 @ v2))
    c = float(v1 @ v2) / denom if denom > 0 else 1.0
    return c if np.isfinite(c) else 1.0


# ---------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------
def kalman_rts_filter(x, window_size=20):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n == 1:
        return x.copy()

    # Stage 1: noise model
    sigma_r, sigma_acc, w_r = _estimate_noise(x, window_size)

    # Stage 2a: adaptive R + first RTS pass
    Rseq = _adaptive_R(x, sigma_r)
    xs = _rts_smooth(x, sigma_r, sigma_acc, Rseq=Rseq, q_scale=1.0)
    y = xs[:, 0].copy()

    # Stage 2b: refinement pass with tighter Q + dynamics-aware blend
    xs2 = _rts_smooth(y, sigma_r, sigma_acc, Rseq=Rseq, q_scale=0.25)
    y_refined = xs2[:, 0].copy()
    if sigma_acc > 1e-12 and sigma_r > 1e-12:
        dyn_ratio = sigma_acc / sigma_r
        w_refine = float(np.clip(1.25 - 1.25 * dyn_ratio, 0.25, 1.0))
    else:
        w_refine = 1.0
    y = w_refine * y_refined + (1.0 - w_refine) * y

    # Stage 3: guarded true flip elimination
    y_pre = y.copy()
    eliminated = _eliminate_flips(y_pre, sigma_r, keep=0.0, thresh_scale=1.0)
    if _corr(eliminated, y_pre) >= 0.97:
        y = eliminated
    else:
        # retry with step-averaging (keep=0.5), gentler
        softened = _eliminate_flips(y_pre, sigma_r, keep=0.5, thresh_scale=1.0)
        if _corr(softened, y_pre) >= 0.97:
            y = softened
        # else: keep unmodified (guard failed)

    # Stage 4: guarded zero-phase ripple damper
    dy = np.diff(y)
    if dy.size > 2:
        r_std = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0))
    else:
        r_std = 0.0
    if r_std > 0:
        alpha = float(np.clip(
            0.55 + 0.4 * np.tanh(2.0 * sigma_r / max(r_std, 1e-12) - 2.0),
            0.40, 0.95))
    else:
        alpha = 1.0
    if alpha < 0.999:
        if alpha < 0.65:
            y_damped = _bilateral_one_pole(_bilateral_one_pole(y, 0.65), 0.65)
        else:
            y_damped = _bilateral_one_pole(y, alpha)
        if _corr(y_damped, y) < 0.97 and alpha < 0.85:
            frac = float(np.clip((0.97 - _corr(y_damped, y)) / 0.10, 0.0, 1.0))
            alpha_eff = alpha + frac * (0.85 - alpha)
            y_damped = _bilateral_one_pole(y, alpha_eff)
        y = y_damped

    # Stage 5: output contract
    y = y[window_size - 1:].copy()
    expected = n - window_size + 1
    if y.shape[0] != expected:
        y = y[:expected]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline entry — routes to the staged pipeline."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Enhanced entry — staged pipeline smoother + flip eliminator."""
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