# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing for Non-Stationary Time Series

Pipeline architecture:
  1. RobustStats   -> global sigma + per-sample local volatility map
  2. AdaptiveKF    -> 2-state constant-velocity Kalman filter with
                      volatility-scaled R_i (measurement) and
                      Q_i = 0.05*q_scale^2*(1 + v_i/median(v)) (process)
  3. RTSSmoother   -> backward smoothing over stored forward pass
  4. Output stage  -> end-aligned window contract (unchanged)
"""
import numpy as np


# ------------------------------------------------------------------
# Stage 1: robust statistics module
# ------------------------------------------------------------------
class RobustStats:
    """Computes global noise scale and a local volatility map."""

    def __init__(self, k=7):
        self.k = k

    def running_median_residual(self, x):
        n = len(x)
        half = self.k // 2
        noise = np.zeros(n)
        xm = x.copy()
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            noise[i] = x[i] - np.median(x[lo:hi])
        return noise

    def compute(self, x):
        resid = self.running_median_residual(x)
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma = max(1e-8, 1.4826 * mad)
        # local volatility: absolute robust residual, lightly smoothed
        v = np.abs(resid)
        k = 3
        if len(v) > k:
            cs = np.cumsum(v)
            vs = np.empty_like(v)
            vs[:k] = cs[:k] / np.arange(1, k + 1)
            vs[k:] = (cs[k:] - cs[:-k]) / k
            v = vs
        v_med = np.median(v) + 1e-12
        return sigma, v, v_med


# ------------------------------------------------------------------
# Stage 2: volatility-adaptive 2-state constant-velocity Kalman filter
# ------------------------------------------------------------------
class AdaptiveCVKalman:
    """Scalar-optimized 2-state [level, slope] filter.

    F = [[1,1],[0,1]], H = [1,0]
    R_i = (sigma_scale * (0.8 + 0.9 * v_i/v_med))^2   (vol-adaptive)
    Q_i = 0.05 * q_scale^2 * (1 + v_i/v_med)          (vol-adaptive)
    """

    def __init__(self, sigma, v, v_med, q_scale=1.0, r_scale=0.8):
        self.v = v
        self.v_med = v_med
        self.sigma = sigma
        self.q_scale = q_scale
        self.r_scale = r_scale

    def _R(self, i):
        rel = self.v[i] / self.v_med
        return max(1e-8, (self.sigma * (self.r_scale + 0.5 * rel)) ** 2)

    def _Q(self, i):
        rel = self.v[i] / self.v_med
        q = 0.05 * (self.q_scale ** 2) * (1.0 + rel)
        return q * (self.sigma ** 2)

    def run(self, x):
        n = len(x)
        # forward filtered state (2-vector) per step
        s_f = np.zeros((n, 2))
        # prior-prediction covariance per step (needed by RTS)
        P_pred = np.zeros((n, 2, 2))
        # filtered covariance per step
        P_f = np.zeros((n, 2, 2))

        # init from first few samples
        m = min(n, 5)
        x0 = np.mean(x[:m])
        v0 = (x[m - 1] - x[0]) / max(1, m - 1) if m > 1 else 0.0
        state = np.array([x0, v0])
        P = np.eye(2) * (self.sigma ** 2 * 10 + 1e-3)

        for i in range(n):
            # ---- predict (F known in closed form for CV model) ----
            # x = [level + slope, slope]
            state_pred = np.array([state[0] + state[1], state[1]])
            # P = F P F'
            p00, p01, p11 = P[0, 0], P[0, 1], P[1, 1]
            Pp = np.array([
                [p00 + 2 * p01 + p11, p01 + p11],
                [p01 + p11, p11],
            ])
            q = self._Q(i)
            # process noise on both level and slope
            Pp[0, 0] += q
            Pp[1, 1] += q
            P_pred[i] = Pp
            state = state_pred

            # ---- update with volatility-adaptive R ----
            R = self._R(i)
            S = Pp[0, 0] + R
            innovation = x[i] - state[0]
            # Huber-like outlier guard
            if abs(innovation) > 4.0 * np.sqrt(S):
                R *= 25.0
                S = Pp[0, 0] + R
                innovation = x[i] - state[0]
            K0 = Pp[0, 0] / S
            K1 = Pp[1, 0] / S
            state = state + np.array([K0 * innovation, K1 * innovation])
            # P = (I - K H) Pp  ( Joseph-free simplified form )
            Pu = np.array([
                [Pp[0, 0] - K0 * Pp[0, 0], Pp[0, 1] - K0 * Pp[0, 1]],
                [Pp[1, 0] - K1 * Pp[0, 0], Pp[1, 1] - K1 * Pp[0, 1]],
            ])
            P = Pu
            s_f[i] = state
            P_f[i] = P

        return s_f, P_f, P_pred


# ------------------------------------------------------------------
# Stage 3: RTS smoother (closed-form for the CV model)
# ------------------------------------------------------------------
def rts_smooth(s_f, P_f, P_pred):
    """Standard RTS recursion; G = P_f[i] F' inv(P_pred[i+1])."""
    n = len(s_f)
    sm = s_f.copy()
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    for i in range(n - 2, -1, -1):
        Pp = P_pred[i + 1]
        PFt = P_f[i] @ F.T  # 2x2
        G = np.linalg.solve(Pp.T, PFt.T).T
        sm[i] = s_f[i] + G @ (sm[i + 1] - F @ s_f[i])
    return sm


# ------------------------------------------------------------------
# Pipeline entry point
# ------------------------------------------------------------------
def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Stage 1: robust statistics
    stats = RobustStats()
    sigma, v, v_med = stats.compute(x)

    # Stage 2: volatility-adaptive 2-state Kalman
    kf = AdaptiveCVKalman(sigma, v, v_med, q_scale=1.0, r_scale=0.8)
    s_f, P_f, P_pred = kf.run(x)

    # Stage 3: RTS smoothing
    smoothed = rts_smooth(s_f, P_f, P_pred)
    estimates = smoothed[:, 0]

    # Stage 4: output contract (end-aligned window)
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