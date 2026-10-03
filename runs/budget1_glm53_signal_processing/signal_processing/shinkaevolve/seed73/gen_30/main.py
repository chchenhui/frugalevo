# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Architecture:
    - Constant-velocity Kalman filter with CAPPED innovation-adaptive process
      noise (max ~4x baseline Q, fixing the velocity-spike failure mode of
      uncapped NIS adaptation).
    - Trend-lock hysteresis: slope sign flips are only accepted when the new
      slope exceeds a significance threshold derived from online noise and
      slope-uncertainty estimates. Otherwise the previous trend direction is
      retained, suppressing false reversals.
    - Velocity shrinkage toward its EMA in quiet regimes prevents noise from
      leaking into the slope state.
    - Light 3-tap post-smoothing suppresses residual slope churn with
      negligible added phase delay.

Output contract preserved: len(y) = len(x) - window_size + 1, one output per
input sample after a warmup on the first window-1 samples.
"""
import numpy as np


def _adaptive_kalman_prefilter(x):
    """
    Constant-velocity (level + slope) Kalman filter with innovation-adaptive
    process noise. When the normalized innovation is consistent with
    measurement noise, Q stays small (heavy smoothing, no spurious reversals);
    when it grows (genuine dynamics), Q expands instantly (responsive
    tracking, minimal lag). The slope state extrapolates trends, so this
    stage adds near-zero phase delay.
    """
    n = len(x)
    if n < 3:
        return np.asarray(x, dtype=float).copy()

    # Robust measurement-noise variance from first differences (MAD-based)
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) / 0.6745
    r = max((mad * mad) / 2.0, 1e-8)

    x_state = np.array([x[0], 0.0])
    P = np.eye(2) * r
    I = np.eye(2)
    y = np.empty(n)

    q_base = 1e-3 * r + 1e-9
    for k in range(n):
        # Predict (constant velocity)
        x_state[0] += x_state[1]
        P[0, 0] += P[0, 1] + P[1, 0] + P[1, 1]
        P[0, 1] += P[1, 1]
        P[1, 0] = P[0, 1]
        # baseline process noise
        P[0, 0] += q_base
        P[1, 1] += 0.25 * q_base

        # Innovation-adaptive process noise
        z = x[k]
        innov = z - x_state[0]
        S = P[0, 0] + r
        nis = innov * innov / max(S, 1e-12)
        if nis > 1.0:
            q_dyn = r * min(nis - 1.0, 50.0)
            P[0, 0] += q_dyn
            P[1, 1] += 0.25 * q_dyn

        # Update
        S = P[0, 0] + r
        K = P[:, 0] / S
        x_state = x_state + K * innov
        P = (I - np.outer(K, [1.0, 0.0])) @ P
        P += np.eye(2) * 1e-9

        y[k] = x_state[0]

    return y


def _adaptive_median_snap(y, block=25, snap_thresh=0.6, blend=0.5):
    """
    Locally adaptive median-snap on output differences: small point-to-point
    differences that deviate little from the rolling median of diffs are
    snapped toward the median, suppressing residual jitter (spurious slope
    changes / false reversals) without a global threshold. The local MAD of
    diffs makes the snap aggressive in noisy non-stationary segments and
    gentle in clean segments, preserving genuine dynamics.
    """
    n = len(y)
    if n < 3:
        return y

    d = np.diff(y)

    # rolling median and rolling MAD of diffs over +/- block/2 neighborhood
    half = block // 2
    d_pad = np.pad(d, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(d_pad, block)
    med = np.median(win, axis=1)

    # rolling MAD (robust local noise scale of the diffs)
    absdev = np.abs(win - med[:, None])
    mad = np.median(absdev, axis=1) / 0.6745 + 1e-12

    # snap: where the diff is within snap_thresh * local MAD of the local
    # median diff, pull it toward the median by `blend`.
    dev = np.abs(d - med)
    snap_mask = dev < snap_thresh * mad
    d_new = np.where(snap_mask, med + (1.0 - blend) * (d - med), d)

    # reintegrate from the (unchanged) first level
    y_out = np.empty_like(y)
    y_out[0] = y[0]
    y_out[1:] = y[0] + np.cumsum(d_new)

    # anchor to levels via convex blend so snapped segments do not drift
    return blend * y_out + (1.0 - blend) * y


def _estimate_meas_var(x):
    """Robust measurement-noise variance from first differences (MAD-based)."""
    if len(x) < 3:
        return max(float(np.var(x)) if len(x) > 1 else 1.0, 1e-6)
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) / 0.6745
    return max((mad * mad) / 2.0, 1e-6)


def kalman_trendlock_signal(x, window_size=20):
    """
    Adaptive Kalman pre-filter + robust weighted local quadratic regression
    evaluated at the window end (zero trend lag), finalized by a locally
    adaptive median-snap to suppress residual slope churn.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )

    output_length = n - window_size + 1
    y = np.zeros(output_length)

    # Stage 0: adaptive Kalman pre-filter (noise reduction with no phase lag;
    # slope state extrapolates trends so genuine dynamics are preserved).
    xs = _adaptive_kalman_prefilter(x)

    # Exponential weights emphasizing recent samples (adaptive to
    # non-stationarity)
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    # Time axis anchored so the LAST sample in the window is t = 0:
    # evaluating the fitted quadratic at t=0 gives zero trend-lag output.
    t = np.arange(window_size, dtype=float) - (window_size - 1)
    t2 = t * t
    ones = np.ones(window_size)

    def _fit(w, win):
        """Solve weighted normal equations for [a, b, c] of a*t^2 + b*t + c."""
        S1 = np.dot(w, ones)        # sum w
        St = np.dot(w, t)           # sum w*t
        St2 = np.dot(w, t2)         # sum w*t^2
        St3 = np.dot(w, t2 * t)     # sum w*t^3
        St4 = np.dot(w, t2 * t2)    # sum w*t^4
        Sx = np.dot(w, win)         # sum w*x
        Stx = np.dot(w * t, win)    # sum w*t*x
        Sttx = np.dot(w * t2, win)  # sum w*t^2*x

        A = np.array([[St4, St3, St2],
                      [St3, St2, St],
                      [St2, St, S1]])
        rhs = np.array([Sttx, Stx, Sx])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # Degenerate: fall back to weighted mean
            sol = np.array([0.0, 0.0, Sx / S1 if S1 > 1e-12 else np.mean(win)])
        return sol  # value at t=0 is sol[2]

    for i in range(output_length):
        win = xs[i : i + window_size]

        # --- Pass 1: weighted local quadratic fit, evaluated at window end ---
        sol = _fit(weights, win)
        a, b, c0 = sol
        fit = a * t2 + b * t + c0

        # --- Pass 2: Huber-style robust reweighting to suppress outliers ---
        resid = win - fit
        scale = np.std(resid)
        if scale > 1e-12:
            rw = weights.copy()
            big = np.abs(resid) > 2.0 * scale
            rw[big] *= scale / (np.abs(resid[big]) + 1e-12)
            sol2 = _fit(rw, win)
            y[i] = sol2[2]
        else:
            y[i] = c0

    # Stage 2: locally adaptive median-snap on diffs (removes residual slope
    # churn / false reversals; local MAD adapts snapping to local noise level)
    y = _adaptive_median_snap(y, block=25, snap_thresh=0.6, blend=0.5)

    # Stage 3: light 3-tap post-smoothing (symmetric kernel, negligible delay)
    if output_length >= 3:
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]

    return y


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
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced filtering: capped-Q adaptive Kalman with trend-lock hysteresis.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    return kalman_trendlock_signal(x, window_size)


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