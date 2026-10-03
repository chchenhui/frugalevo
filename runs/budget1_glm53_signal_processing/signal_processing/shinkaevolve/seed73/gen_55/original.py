# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Crossover architecture (best-of-breed pipeline):
    1. Soft innovation-adaptive constant-velocity Kalman pre-filter
       (NIS capped) -> high tracking accuracy / correlation, near-zero lag.
    2. Savitzky-Golay trend-preserving smoothing -> strong suppression of
       spurious slope churn while preserving genuine polynomial trends.
    3. Recency-weighted sliding-window aggregation (vectorized) -> output
       contract len(y) = len(x) - window_size + 1 with minimal group delay.
    4. Edge-preserving 5-tap median snap -> removes isolated wiggles.
    5. Trend-lock hysteresis on output diffs -> a slope reversal is only
       accepted when it is significant w.r.t. the local robust (MAD) noise
       scale; insignificant reversals snap to the local median trend.

Output contract preserved: len(y) = len(x) - window_size + 1.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter as _scipy_savgol
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


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

    # Vectorized simple moving average (baseline)
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def _adaptive_kalman_prefilter(x):
    """
    Constant-velocity (level + slope) Kalman filter with innovation-adaptive
    process noise. When the normalized innovation is consistent with
    measurement noise, Q stays small (heavy smoothing, stable slope
    estimate); when it grows (genuine dynamics), Q expands so the filter
    tracks trend changes quickly (minimal lag). The soft cap on the NIS
    term prevents velocity spikes from noise, keeping accuracy high.

    Returns the filtered level estimate, same length as x.
    """
    n = len(x)
    if n < 3:
        return np.asarray(x, dtype=float).copy()

    # Robust MAD-based measurement noise variance from first differences
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) / 0.6745
    r = max((mad * mad) / 2.0, 1e-8)

    # State: [level, slope]
    xs = np.array([x[0], 0.0])
    P = np.eye(2) * max(r, 1e-6)
    I = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    q_base = 1e-3 * r + 1e-9

    y = np.empty(n)
    for k in range(n):
        # Predict
        xs = F @ xs
        P = F @ P @ F.T

        innov = x[k] - xs[0]
        S = P[0, 0] + r
        nis = innov * innov / max(S, 1e-12)

        # Adapt process noise: small NIS -> smooth; large NIS -> responsive
        # (softened: less aggressive tracking keeps accuracy/correlation high)
        q_dyn = q_base * (1.0 + 1.0 * min(nis, 12.0))
        P[0, 0] += q_dyn
        P[1, 1] += 0.25 * q_dyn

        # Update
        S = P[0, 0] + r
        K = P[:, 0] / S
        xs = xs + K * innov
        P = (I - np.outer(K, H[0])) @ P
        P += np.eye(2) * 1e-12

        y[k] = xs[0]
    return y


def _savgol_smooth(x, win, order=3):
    """
    Trend-preserving Savitzky-Golay smoothing. Uses SciPy when available,
    otherwise a pure-NumPy equivalent via the least-squares hat matrix.

    Args:
        x: signal (1D array)
        win: window length (forced odd, clipped to signal length)
        order: polynomial order

    Returns:
        Smoothed signal, same length as x.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if win < 3 or n < 3:
        return x.copy()

    # force odd window, clipped to signal support
    if win % 2 == 0:
        win -= 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 3:
        return x.copy()
    order = min(order, win - 1)

    if _HAVE_SCIPY:
        return _scipy_savgol(x, win, order)

    # NumPy fallback: smoothing coefficients are the center row of the
    # least-squares hat matrix on the Vandermonde design.
    half = win // 2
    tt = np.arange(-half, half + 1, dtype=float)
    A = np.vander(tt, order + 1, increasing=True)
    hat = A @ np.linalg.pinv(A)
    coeffs = hat[half]
    pad = np.pad(x, half, mode="edge")
    return np.correlate(pad, coeffs, mode="valid")


def _median5_snap(y):
    """Edge-preserving 5-tap median: removes isolated wiggles (spurious
    slope changes) while passing monotone trends untouched (zero added lag,
    no ramp attenuation). Vectorized."""
    m = len(y)
    if m < 5:
        return y
    pad = np.concatenate(([y[0], y[0]], y, [y[-1], y[-1]]))
    return np.median(
        np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1
    )


def _trend_lock_snap(y, block=15, k=1.5, blend=0.6):
    """
    Trend-lock hysteresis on output differences: a slope sign reversal is
    only accepted if the opposing diff is significant relative to the local
    robust (MAD) noise scale of the diffs. Insignificant reversals are
    snapped to the local median trend direction, suppressing false
    reversals with no added phase delay (large genuine reversals pass
    through untouched).
    """
    n = len(y)
    if n < 4:
        return y

    d = np.diff(y)

    half = block // 2
    dpad = np.pad(d, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(dpad, block)
    med = np.median(win, axis=1)
    mad = (
        np.median(np.abs(win - med[:, None]), axis=1) / 0.6745 + 1e-12
    )

    # spurious reversal: diff opposes the local median trend direction AND
    # is not statistically significant vs. the local noise scale of diffs
    oppose = np.sign(d) != np.sign(med)
    insignificant = np.abs(d) < k * mad
    to_snap = oppose & insignificant

    d_new = np.where(to_snap, med, d)

    # re-integrate from the unchanged first level, then anchor to the
    # original levels via a convex blend so snapped segments cannot drift
    y2 = np.empty_like(y)
    y2[0] = y[0]
    y2[1:] = y[0] + np.cumsum(d_new)
    return blend * y2 + (1.0 - blend) * y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced crossover pipeline:
      (1) soft innovation-adaptive Kalman pre-filter (accuracy, zero lag),
      (2) Savitzky-Golay trend-preserving smoothing (slope churn removal),
      (3) vectorized recency-weighted sliding-window aggregation
          (output contract + minimal group delay),
      (4) edge-preserving 5-tap median snap,
      (5) trend-lock hysteresis on diffs (false reversal suppression).

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    x = np.asarray(x, dtype=float)
    n = len(x)
    output_length = n - window_size + 1

    # --- Stage 0: adaptive Kalman pre-filter ------------------------------
    # Soft NIS-capped adaptation: heavy smoothing in quiet regimes (no
    # spurious reversals), instant responsiveness on genuine dynamics.
    xk = _adaptive_kalman_prefilter(x)

    # --- Stage 1: Savitzky-Golay trend-preserving smoothing ----------------
    # Local low-order polynomial fits preserve trends and slopes while
    # suppressing residual noise, without moving-average phase delay.
    sg_win = window_size if window_size % 2 == 1 else window_size - 1
    sg_win = max(sg_win, 5)
    xs = _savgol_smooth(xk, sg_win, 3)

    # --- Stage 2: recency-weighted sliding-window aggregation -------------
    # Exponential weights emphasizing recent samples (steep decay keeps the
    # effective group delay small); fully vectorized via a matrix product.
    w = np.exp(np.linspace(-2.0, 0.0, window_size))
    w = w / np.sum(w)
    wins = np.lib.stride_tricks.sliding_window_view(xs, window_size)
    y = wins @ w  # shape (output_length,)

    # --- Stage 3: edge-preserving 5-tap median snap ------------------------
    y = _median5_snap(y)

    # --- Stage 4: trend-lock hysteresis on diffs ---------------------------
    y = _trend_lock_snap(y, block=15, k=1.5, blend=0.6)

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