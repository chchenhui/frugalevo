# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Approach (crossover of best-performing variants): a causal
constant-acceleration Kalman filter with noise-adaptive measurement
covariance and a trend-reversal gate that temporarily boosts process
noise at statistically significant turning points, followed by an RTS
backward smoother and a final localized reversal-suppression hysteresis
stage that only smooths neighborhoods exhibiting weak, rapidly-flipping
gradients (residual micro-oscillations).
"""
import numpy as np


def _robust_noise_estimate(x, k=7):
    """Estimate measurement noise using a running median residual (MAD)."""
    n = len(x)
    half = k // 2
    noise = np.zeros(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        noise[i] = x[i] - np.median(seg)
    mad = np.median(np.abs(noise - np.median(noise)))
    sigma = max(1e-8, 1.4826 * mad)
    return noise, sigma


def _local_hysteresis(y, flip_span=3, k=0.5, sg_half=2):
    """Suppress weak micro-oscillations with a percentile-blended
    3-scale Savitzky-Golay kernel bank (half-widths
    [sg_half//2, sg_half, 2*sg_half]) with Gaussian weights centered
    at the 20/50/80th percentiles of local volatility. Only applied
    in neighborhoods exhibiting weak, rapidly-flipping gradients."""
    n = len(y)
    if n < 7:
        return y
    g = np.gradient(y)
    med_g = np.median(np.abs(g)) + 1e-12
    weak = np.abs(g) < k * med_g
    s = np.sign(g)
    flip = np.zeros(n, dtype=bool)
    for lag in range(1, flip_span + 1):
        if lag < n:
            flip[lag:] |= (s[lag:] * s[:-lag]) < 0
    flag = weak & flip
    if not np.any(flag):
        return y
    # dilate flags by a couple samples on each side
    dil = np.zeros(n, dtype=bool)
    r = 2
    idx = np.where(flag)[0]
    for i in idx:
        dil[max(0, i - r):min(n, i + r + 1)] = True
    out = y.copy()
    # Build the 3-scale SG coefficient bank (order 2, zero-phase)
    halfs = [max(1, sg_half // 2), max(1, sg_half), 2 * max(1, sg_half)]
    coeffs = []
    for h in halfs:
        win = 2 * h + 1
        if win < 3 or win > n:
            coeffs.append(None)
            continue
        # least-squares fit of order-2 polynomial: smoothing coefficients
        idx = np.arange(win, dtype=float) - h
        A = np.vstack([np.ones(win), idx, idx * idx]).T
        c_h, *_ = np.linalg.lstsq(A, A, rcond=None)
        coeffs.append(c_h[0])
    # local volatility for percentile weighting
    vol = np.abs(np.gradient(g)) + 1e-12
    vol_flag = vol[np.where(flag)[0]] if np.any(flag) else vol
    p20, p50, p80 = np.percentile(vol_flag, [20, 50, 80])

    def _blend_weights(v):
        # Gaussian weights centered at p20, p50, p80 (normalized)
        w = np.exp(-0.5 * ((v - p20) / (p50 - p20 + 1e-9)) ** 2)
        w += np.exp(-0.5 * ((v - p50) / (p50 - p20 + 1e-9)) ** 2)
        w += np.exp(-0.5 * ((v - p80) / (p80 - p50 + 1e-9)) ** 2)
        # low volatility -> narrow kernel; high volatility -> wide kernel
        return w / (w.sum() + 1e-12)

    # process each flagged neighborhood
    i = 0
    while i < n:
        if dil[i]:
            j = i
            while j < n and dil[j]:
                j += 1
            lo = max(0, i - 2)
            hi = min(n, j + 2)
            seg = y[lo:hi]
            if len(seg) >= 5:
                # blended multi-scale SG smoothing of this neighborhood
                v = np.median(vol[i:j])
                wts = _blend_weights(v)
                sm_acc = np.zeros(len(seg))
                w_tot = 0.0
                for c_h, w in zip(coeffs, wts):
                    if c_h is None:
                        continue
                    hw = (len(c_h) - 1) // 2
                    if len(seg) >= len(c_h):
                        pad = hw
                        padded = np.concatenate((
                            np.full(pad, seg[0]), seg, np.full(pad, seg[-1]),
                        ))
                        sm_h = np.convolve(padded, c_h, mode='valid')
                        sm_acc += w * sm_h
                        w_tot += w
                if w_tot > 0:
                    sm = sm_acc / w_tot
                    core_start = i - lo
                    core_len = j - i
                    avail = min(core_len, len(sm) - core_start)
                    if avail > 0:
                        # blend: keep 50% of original to preserve dynamics
                        out[i:i + avail] = 0.5 * out[i:i + avail] + \
                            0.5 * sm[core_start:core_start + avail]
            i = j
        else:
            i += 1
    return out


def adaptive_filter(x, window_size=20):
    """Kalman trend filter + RTS smoother + localized hysteresis."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})")

    _, base_sigma = _robust_noise_estimate(x)

    dt = 1.0
    F = np.array([[1.0, dt, 0.5 * dt * dt],
                  [0.0, 1.0, dt],
                  [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0]])

    q_base = np.diag([1e-4, 1e-4, 1e-5]) * (base_sigma ** 2)
    q_boost = np.diag([5e-2, 2e-1, 1e-2]) * (base_sigma ** 2)
    R_base = max(1e-6, (base_sigma * 0.8) ** 2)

    m = min(n, 5)
    x0 = np.mean(x[:m])
    s0 = (x[m - 1] - x[0]) / max(1, m - 1) if m > 1 else 0.0
    state = np.array([x0, s0, 0.0])
    P = np.eye(3) * (base_sigma ** 2 * 10 + 1e-3)

    I3 = np.eye(3)
    s_f = np.zeros((n, 3))
    P_f = np.zeros((n, 3, 3))
    P_pred = np.zeros((n, 3, 3))

    slope_history = np.zeros(n)
    reversal_window = max(3, window_size // 4)
    boost_counter = 0

    for i in range(n):
        # --- Predict ---
        state = F @ state
        P = F @ P @ F.T + (q_boost if boost_counter > 0 else q_base)
        if boost_counter > 0:
            boost_counter -= 1
        P_pred[i] = P

        # --- Trend reversal gate ---
        slope_hat = state[1]
        slope_var = max(P[1, 1], 1e-12)
        if i >= reversal_window:
            prev_slope = slope_history[i - reversal_window]
            if prev_slope * slope_hat < 0:
                denom = np.sqrt(slope_var) + 1e-9
                sig = abs(slope_hat - prev_slope) / denom
                innovation = x[i] - state[0]
                inno_sig = abs(innovation) / (np.sqrt(P[0, 0] + R_base) + 1e-9)
                if sig > 1.5 and inno_sig > 1.0:
                    boost_counter = max(2, window_size // 8)

        slope_history[i] = slope_hat

        # --- Update ---
        R = R_base
        S = P[0, 0] + R
        K = P[:, 0] / S
        innovation = x[i] - state[0]
        # Huber-like guard against gross outliers
        if abs(innovation) > 4.0 * np.sqrt(S):
            R *= 25.0
            S = P[0, 0] + R
            K = P[:, 0] / S
            innovation = x[i] - state[0]
        state = state + K * innovation
        P = (I3 - np.outer(K, H)) @ P

        s_f[i] = state
        P_f[i] = P

    # --- RTS backward smoother ---
    smoothed = s_f.copy()
    for i in range(n - 2, -1, -1):
        Pp = P_pred[i + 1]
        G = np.linalg.solve(Pp.T, (P_f[i] @ F.T).T).T
        smoothed[i] = s_f[i] + G @ (smoothed[i + 1] - F @ s_f[i])

    estimates = smoothed[:, 0]

    # --- Localized reversal-suppression hysteresis (SG kernel bank) ---
    estimates = _local_hysteresis(estimates, sg_half=max(2, window_size // 10))

    output_length = n - window_size + 1
    y = estimates[n - output_length:]
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Alias for the adaptive Kalman trend filter."""
    return adaptive_filter(x, window_size)


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
    x = np.asarray(input_signal, dtype=float)
    return adaptive_filter(x, window_size)


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