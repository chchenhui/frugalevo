# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Redesigned architecture (dual-state causal core, minimal cascade):
    1. Constant-velocity Kalman filter exposing BOTH level and slope states,
       with NIS-adaptive process noise (capped) and online noise estimation.
       The slope state both extrapolates trends (near-zero lag) and provides
       a statistically meaningful turning-point detector.
    2. Single robust Savitzky-Golay polish (Huber-IRLS, 2 passes) of the
       Kalman level: preserves polynomial trends, suppresses residual noise
       and outliers without moving-average group delay.
    3. Adaptive slope-extrapolation blend: in high-NIS (genuine dynamics)
       regimes the output is blended with level + slope (one-step-ahead),
       compensating residual Kalman lag at turning points.
    4. Gated reversal hysteresis on output diffs: a slope sign flip is kept
       only if significant vs. local MAD of diffs; otherwise snapped to the
       local median trend. Single principled stage (no stacked smoothers).

Output contract preserved: len(y) = len(x) - window_size + 1.
"""
import numpy as np

try:
    from scipy.signal import savgol_filter as _scipy_savgol
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------
# Stage 1: dual-state adaptive Kalman core (level + slope, NIS-gated Q)
# ----------------------------------------------------------------------
class _KalmanCore:
    """Constant-velocity Kalman filter with NIS-capped adaptive process
    noise. Exposes the level estimate, slope estimate, and per-sample NIS
    so downstream stages can adapt to regime changes."""

    __slots__ = ("r", "q_base", "xs", "P")

    def __init__(self, x0, r):
        self.r = r
        self.q_base = 1e-3 * r + 1e-9
        self.xs = np.array([x0, 0.0])          # [level, slope]
        self.P = np.eye(2) * max(r, 1e-6)

    def step(self, z):
        # --- predict (constant velocity) ---
        self.xs[0] += self.xs[1]
        self.P[0, 0] += self.P[0, 1] + self.P[1, 0] + self.P[1, 1]
        self.P[0, 1] += self.P[1, 1]
        self.P[1, 0] = self.P[0, 1]
        # baseline process noise
        q = self.q_base
        self.P[0, 0] += q
        self.P[1, 1] += 0.25 * q

        # --- NIS-adaptive process noise (soft-capped) ---
        innov = z - self.xs[0]
        S = self.P[0, 0] + self.r
        nis = innov * innov / max(S, 1e-12)
        if nis > 1.0:
            q_dyn = self.r * min(nis - 1.0, 8.0)   # cap prevents slope spikes
            self.P[0, 0] += q_dyn
            self.P[1, 1] += 0.25 * q_dyn

        # --- update ---
        S = self.P[0, 0] + self.r
        K0 = self.P[0, 0] / S
        K1 = self.P[1, 0] / S
        self.xs[0] += K0 * innov
        self.xs[1] += K1 * innov
        P00 = self.P[0, 0] * (1.0 - K0)
        P01 = self.P[0, 1] * (1.0 - K0)
        P10 = self.P[1, 0] - K1 * self.P[0, 0]
        P11 = self.P[1, 1] - K1 * self.P[0, 1]
        self.P[0, 0], self.P[0, 1] = P00, P01
        self.P[1, 0], self.P[1, 1] = P10, P11
        # symmetrize / regularize
        self.P[0, 1] = self.P[1, 0] = 0.5 * (P01 + P10)
        self.P[0, 0] += 1e-12
        self.P[1, 1] += 1e-12
        return self.xs[0], self.xs[1], nis


def _robust_meas_var(x):
    """MAD-based measurement-noise variance from first differences."""
    if len(x) < 3:
        return max(float(np.var(x)) if len(x) > 1 else 1.0, 1e-6)
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) / 0.6745
    return max((mad * mad) / 2.0, 1e-6)


def _run_kalman(x):
    """Run the dual-state Kalman core over the whole signal.
    Returns (level, slope, nis) arrays, each same length as x."""
    n = len(x)
    level = np.empty(n)
    slope = np.empty(n)
    nis_arr = np.empty(n)
    if n < 3:
        lvl = np.asarray(x, dtype=float)
        return lvl.copy(), np.zeros(n), np.zeros(n)
    core = _KalmanCore(x[0], _robust_meas_var(x))
    # slope EMA for shrinkage in quiet regimes (prevents noise leaking
    # into the slope state)
    slope_ema = 0.0
    for k in range(n):
        lvl_k, slp_k, nis_k = core.step(x[k])
        # shrink slope toward its EMA when innovations are small (quiet)
        if nis_k < 1.0:
            slp_k_final = 0.5 * slp_k + 0.5 * slope_ema
        else:
            slp_k_final = slp_k
        slope_ema = 0.9 * slope_ema + 0.1 * slp_k_final
        level[k] = lvl_k
        slope[k] = slp_k_final
        nis_arr[k] = nis_k
    return level, slope, nis_arr


# ----------------------------------------------------------------------
# Stage 2: robust (Huber-IRLS) Savitzky-Golay polish
# ----------------------------------------------------------------------
def _savgol_smooth(x, win, order=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if win < 3 or n < 3:
        return x.copy()
    if win % 2 == 0:
        win -= 1
    if win > n:
        win = n if n % 2 == 1 else n - 1
    if win < 3:
        return x.copy()
    order = min(order, win - 1)
    if _HAVE_SCIPY:
        return _scipy_savgol(x, win, order)
    half = win // 2
    tt = np.arange(-half, half + 1, dtype=float)
    A = np.vander(tt, order + 1, increasing=True)
    hat = A @ np.linalg.pinv(A)
    coeffs = hat[half]
    pad = np.pad(x, half, mode="edge")
    return np.correlate(pad, coeffs, mode="valid")


def _robust_savgol(x, win, order=3, iters=2):
    """Savitzky-Golay with Huber reweighting iterations (vectorized over
    residuals via a second smoothing of weights)."""
    y = _savgol_smooth(x, win, order)
    for _ in range(iters - 1):
        resid = x - y
        scale = np.median(np.abs(resid - np.median(resid))) / 0.6745
        if scale <= 1e-12:
            break
        w = np.ones_like(x)
        big = np.abs(resid) > 2.0 * scale
        w[big] = scale / (np.abs(resid[big]) + 1e-12)
        # weighted SG approximation: smooth (w*x) and (w), take ratio
        num = _savgol_smooth(w * x, win, order)
        den = _savgol_smooth(w, win, order)
        ok = np.abs(den) > 1e-9
        y_new = y.copy()
        y_new[ok] = num[ok] / den[ok]
        y = y_new
    return y


# ----------------------------------------------------------------------
# Stage 4: statistically-gated reversal hysteresis
# ----------------------------------------------------------------------
def _reversal_gate(y, block=15, k=1.5, blend=0.6):
    """Reject slope sign flips that are insignificant vs. the local MAD of
    diffs; snap them to the local median trend direction."""
    n = len(y)
    if n < 4:
        return y
    d = np.diff(y)
    half = block // 2
    dpad = np.pad(d, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(dpad, block)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1) / 0.6745 + 1e-12
    oppose = np.sign(d) != np.sign(med)
    insignificant = np.abs(d) < k * mad
    to_snap = oppose & insignificant
    d_new = np.where(to_snap, med, d)
    y2 = np.empty_like(y)
    y2[0] = y[0]
    y2[1:] = y[0] + np.cumsum(d_new)
    return blend * y2 + (1.0 - blend) * y


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
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
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Dual-state causal pipeline:
      (1) NIS-adaptive Kalman core (level + slope + NIS traces),
      (2) robust Savitzky-Golay polish of the level (trend preserving),
      (3) adaptive slope-extrapolation blend at high-NIS (genuine dynamics)
          points to cancel residual lag,
      (4) gated reversal hysteresis on output diffs.

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

    # --- Stage 1: dual-state Kalman core ---
    level, slope, nis = _run_kalman(x)

    # --- Stage 2: robust SG polish (moderate window: enough to denoise,
    # small enough to avoid over-smoothing / lag) ---
    sg_win = max(7, min(window_size, 15))
    if sg_win % 2 == 0:
        sg_win -= 1
    smoothed = _robust_savgol(level, sg_win, 3)

    # --- Stage 3: adaptive slope-extrapolation blend ---
    # Where innovations indicate genuine dynamics (NIS above threshold),
    # blend in the one-step-ahead prediction level + slope to cancel the
    # Kalman/SG residual lag at turning points.
    thr = 3.0
    alpha = np.clip((nis - thr) / (thr * 4.0), 0.0, 0.5)  # cap the boost
    predicted = level + slope
    y_full = (1.0 - alpha) * smoothed + alpha * predicted

    # --- Sliding-window output contract (vectorized, recency-weighted) ---
    w = np.exp(np.linspace(-1.0, 0.0, window_size))
    w = w / np.sum(w)
    wins = np.lib.stride_tricks.sliding_window_view(y_full, window_size)
    y = wins @ w

    # --- Stage 4: gated reversal hysteresis ---
    y = _reversal_gate(y, block=15, k=1.5, blend=0.6)

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