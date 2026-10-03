# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Structural redesign: an explicit modular pipeline with linear data flow.

    x --> [Stage 0: RobustNoiseModel]
          --> [Stage 1: CappedKalmanStage]        (causal, responsive, capped adaptive Q)
          --> [Stage 2: EdgeQuadraticStage]       (vectorized zero-trend-lag local poly fit)
          --> [Stage 3: TrendGovernorStage]       (deadband snap + sign-flip hysteresis)
          --> [Stage 4: FinishStage]              (symmetric 3-tap smoother)
          --> y

Design goals mapped to objectives:
    (1) slope change minimization  -> TrendGovernor deadband + hysteresis
    (2) lag error minimization     -> Kalman slope extrapolation + edge evaluation
    (3) tracking accuracy          -> capped adaptive Q (opens fast on genuine dynamics)
    (4) false reversal penalty     -> sign-flip significance gate vs local slope MAD

Output contract preserved: len(y) = len(x) - window_size + 1.
"""
import numpy as np


# --------------------------------------------------------------------------
# Stage 0: RobustNoiseModel
# --------------------------------------------------------------------------
def _robust_meas_var(x):
    """Robust measurement-noise variance from first differences (MAD-based)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return max(float(np.var(x)) if len(x) > 1 else 1.0, 1e-6)
    d = np.diff(x)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) / 0.6745
    return max((mad * mad) / 2.0, 1e-6)


# --------------------------------------------------------------------------
# Stage 1: CappedKalmanStage
# --------------------------------------------------------------------------
class _CappedKalmanStage:
    """
    Constant-velocity (level + slope) Kalman filter with innovation-adaptive
    process noise, CAPPED at `q_cap` x baseline Q. The cap prevents
    measurement noise from flooding the velocity state (spurious slope
    churn / false reversals) while still allowing Q to expand quickly when
    the normalized innovation squared indicates genuine dynamics.
    """

    def __init__(self, r, q_level, q_slope, q_cap=4.0, vel_shrink=0.35):
        self.r = max(r, 1e-8)
        self.ql = q_level
        self.qs = q_slope
        self.cap = q_cap
        self.vs = vel_shrink
        self.x = None                      # [level, slope]
        self.P = np.eye(2) * max(self.r, 1.0)
        self.slope_ema = 0.0

    def run(self, x):
        n = len(x)
        lvl = np.empty(n)
        for k in range(n):
            z = x[k]
            if self.x is None:
                self.x = np.array([z, 0.0])
                lvl[k] = z
                continue

            # ---- predict (F = [[1,1],[0,1]]) ----
            self.x[0] += self.x[1]
            p00, p01, p11 = self.P[0, 0], self.P[0, 1], self.P[1, 1]
            self.P[0, 0] = p00 + 2.0 * p01 + p11
            self.P[0, 1] = self.P[1, 0] = p01 + p11

            # ---- capped innovation-adaptive Q ----
            innov = z - self.x[0]
            S = self.P[0, 0] + self.r
            nis = innov * innov / max(S, 1e-12)
            mult = 1.0 + (self.cap - 1.0) * min(nis, 6.0) / 6.0   # in [1, cap]
            self.P[0, 0] += self.ql * mult
            self.P[1, 1] += self.qs * mult

            # ---- update ----
            S = self.P[0, 0] + self.r
            k0 = self.P[0, 0] / S
            k1 = self.P[1, 0] / S
            self.x[0] += k0 * innov
            self.x[1] += k1 * innov
            p00, p01, p11 = self.P[0, 0], self.P[0, 1], self.P[1, 1]
            self.P[0, 0] = max((1.0 - k0) * p00, 1e-12)
            self.P[0, 1] = (1.0 - k0) * p01
            self.P[1, 0] = p01 - k1 * p00
            self.P[1, 1] = max(p11 - k1 * p01, 1e-12)

            # ---- quiet-regime velocity shrinkage toward slope EMA ----
            if nis < 2.0:
                self.x[1] = (1.0 - self.vs) * self.x[1] + self.vs * self.slope_ema

            self.slope_ema = 0.95 * self.slope_ema + 0.05 * self.x[1]
            lvl[k] = self.x[0]
        return lvl


# --------------------------------------------------------------------------
# Stage 2: EdgeQuadraticStage (vectorized)
# --------------------------------------------------------------------------
def _edge_quadratic_stage(xk, W, decay=2.0):
    """
    Recency-weighted local polynomial regression evaluated at the window's
    leading edge (newest sample at t=0 -> zero trend lag). Fully vectorized:
    the weighted least-squares projection onto the coefficient of the
    constant term is precomputed once as a length-W kernel, then applied to
    all sliding windows via a single matrix product.
    """
    xk = np.asarray(xk, dtype=float)
    n = len(xk)
    if W < 2 or n < W:
        return xk[W - 1:].copy() if n >= W else xk.copy()

    order = 2 if W >= 5 else 1
    t = np.arange(W, dtype=float) - (W - 1)          # newest sample at t=0
    w = np.exp(np.linspace(-decay, 0.0, W))          # recency weighting

    cols = [np.ones(W)] + [t ** (p + 1) for p in range(order)]
    A = np.vstack(cols).T                            # W x (order+1)
    Aw = A * w[:, None]
    M = Aw.T @ A + 1e-9 * np.eye(order + 1)
    # projection operator: window -> polynomial coefficients
    G = Aw @ np.linalg.solve(M, np.eye(order + 1))   # W x (order+1)

    wins = np.lib.stride_tricks.sliding_window_view(xk, W)   # (n-W+1, W)
    coef = wins @ G                                          # (n-W+1, order+1)
    return coef[:, 0].copy()                                 # value at t = 0


# --------------------------------------------------------------------------
# Stage 3: TrendGovernorStage
# --------------------------------------------------------------------------
def _trend_governor_stage(y, W,
                          snap_thresh=0.7, snap_blend=0.55,
                          hyst_k=1.0, anchor=0.6):
    """
    Multi-scale slope stabilization:
      (a) rolling median/MAD deadband snapping of point-to-point diffs —
          diffs close to the local median diff are pulled toward it,
          aggressive in noisy segments, gentle in clean ones;
      (b) sign-flip hysteresis — a directional reversal is accepted only
          when the new slope magnitude exceeds hyst_k * local slope MAD;
          otherwise the previous trend direction is retained with damped
          magnitude (kills noise-induced false reversals);
      (c) anchored reintegration — snapped diffs are reintegrated and
          convex-blended with the original levels so no drift accumulates.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return y.copy()

    d = np.diff(y)
    m = len(d)

    # block size adapts to window length; force odd for symmetric padding
    block = int(np.clip(2 * W - 9, 5, 41))
    if block % 2 == 0:
        block += 1
    block = min(block, m if m % 2 == 1 else m - 1)
    if block < 3:
        return y.copy()

    half = block // 2
    dp = np.pad(d, half, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(dp, block)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1) / 0.6745 + 1e-12

    # (a) deadband snap
    dev = np.abs(d - med)
    mask = dev < snap_thresh * mad
    d_s = np.where(mask, med + (1.0 - snap_blend) * (d - med), d)

    # (b) sign-flip hysteresis
    out = d_s.copy()
    cur = 0
    for i in range(m):
        s = out[i]
        thr = hyst_k * mad[i]
        if cur == 0:
            if abs(s) > thr:
                cur = 1 if s > 0 else -1
        else:
            if (s > 0 and cur > 0) or (s < 0 and cur < 0):
                pass                                  # direction consistent
            elif abs(s) > thr:
                cur = -cur                            # significant flip: accept
            else:
                # insignificant flip: retain previous direction, damped
                out[i] = cur * 0.3 * abs(s)

    # (c) anchored reintegration
    yr = np.empty(n)
    yr[0] = y[0]
    yr[1:] = y[0] + np.cumsum(out)
    return anchor * yr + (1.0 - anchor) * y


# --------------------------------------------------------------------------
# Stage 4: FinishStage
# --------------------------------------------------------------------------
def _finish_stage(y):
    """Symmetric 3-tap binomial smoother (negligible phase delay)."""
    y = y.copy()
    m = len(y)
    if m >= 3:
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]
    return y


# --------------------------------------------------------------------------
# Pipeline assembly
# --------------------------------------------------------------------------
def pipeline_filter(x, window_size=20):
    """
    Full modular pipeline: noise model -> capped Kalman -> edge quadratic ->
    trend governor -> finish.

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

    # Stage 0: robust noise model
    r = _robust_meas_var(x)

    # Stage 1: capped adaptive Kalman (q_base sized to noise; cap at 4x)
    q_level = 0.01 * r + 1e-9
    q_slope = 2e-4 * r + 1e-12
    lvl = _CappedKalmanStage(r, q_level, q_slope,
                             q_cap=4.0, vel_shrink=0.35).run(x)

    # Stage 2: vectorized edge-evaluated local polynomial fit
    yq = _edge_quadratic_stage(lvl, window_size)

    # Stage 3: multi-scale trend governor (deadband + hysteresis)
    yg = _trend_governor_stage(yq, window_size)

    # Stage 4: light symmetric post-smoothing
    return _finish_stage(yg)


# --------------------------------------------------------------------------
# Public API (unchanged contract)
# --------------------------------------------------------------------------
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
    Enhanced filtering: modular staged pipeline (capped-Q Kalman +
    vectorized edge quadratic + trend governor + finish).

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    return pipeline_filter(x, window_size)


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