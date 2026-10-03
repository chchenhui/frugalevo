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


class CappedKalmanTrendLock:
    """Constant-velocity Kalman filter with capped Q adaptation and trend lock."""

    def __init__(self, meas_var, q_base, q_slope, vel_shrink=0.35,
                 trend_lock_factor=1.0):
        self.x = np.zeros(2)              # [level, slope]
        self.P = np.eye(2) * max(meas_var, 1.0)
        self.R = max(meas_var, 1e-8)
        self.q_level = q_level = q_base
        self.q_slope = q_slope
        # online slope estimate stats
        self.slope_ema = 0.0
        self.slope_var = max(q_slope, 1e-12)   # running var of slope estimates
        self.forget = 0.95
        self.vel_shrink = vel_shrink
        self.lock_factor = trend_lock_factor
        self.initialized = False

    def _ensure_init(self, z):
        if not self.initialized:
            self.x[0] = z
            self.x[1] = 0.0
            self.P = np.eye(2) * max(self.R, 1.0)
            self.initialized = True

    def step(self, z):
        self._ensure_init(z)

        # ---- predict ----
        self.x[0] += self.x[1]
        # P = F P F^T for F = [[1,1],[0,1]]
        p00, p01, p11 = self.P[0, 0], self.P[0, 1], self.P[1, 1]
        p00_new = p00 + 2.0 * p01 + p11
        p01_new = p01 + p11
        self.P[0, 0] = p00_new
        self.P[0, 1] = p01_new
        self.P[1, 0] = p01_new

        # ---- innovation-based adaptive Q (CAPPED) ----
        innov = z - self.x[0]
        S = self.P[0, 0] + self.R
        nis = innov * innov / max(S, 1e-12)
        # cap the multiplier at 4x so noise cannot flood the velocity state
        mult = 1.0 + 3.0 * min(nis, 6.0) / 6.0          # in [1, 4]
        self.P[0, 0] += self.q_level * mult
        self.P[1, 1] += self.q_slope * mult
        self.P[0, 1] += 0.5 * self.q_slope * mult
        self.P[1, 0] = self.P[0, 1]

        # ---- update ----
        S = self.P[0, 0] + self.R
        k0 = self.P[0, 0] / S
        k1 = self.P[1, 0] / S
        self.x[0] += k0 * innov
        self.x[1] += k1 * innov

        # covariance update (I - K H) P
        p00, p01, p11 = self.P[0, 0], self.P[0, 1], self.P[1, 1]
        new_p00 = (1.0 - k0) * p00
        new_p01 = (1.0 - k0) * p01
        new_p10 = p01 - k1 * p00
        new_p11 = p11 - k1 * p01
        self.P[0, 0] = max(new_p00, 1e-12)
        self.P[0, 1] = new_p01
        self.P[1, 0] = new_p10
        self.P[1, 1] = max(new_p11, 1e-12)

        # ---- quiet-regime velocity shrinkage (denoise slope) ----
        if nis < 2.0:
            self.x[1] = (1.0 - self.vel_shrink) * self.x[1] + \
                        self.vel_shrink * self.slope_ema

        # ---- trend-lock hysteresis on slope sign ----
        d = self.x[1] - self.slope_ema
        self.slope_var = (self.forget * self.slope_var
                          + (1.0 - self.forget) * d * d)
        slope_sigma = np.sqrt(max(self.slope_var, 1e-12))
        threshold = self.lock_factor * slope_sigma
        if self.slope_ema * self.x[1] < 0.0:  # sign flip proposed
            if abs(self.x[1]) < threshold:
                # not significant: keep previous trend direction, scaled
                self.x[1] = np.sign(self.slope_ema) * \
                            max(abs(self.x[1]), threshold * 0.5)
        self.slope_ema = (self.forget * self.slope_ema
                          + (1.0 - self.forget) * self.x[1])

        return self.x[0]


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
    Capped-Q adaptive Kalman filter with trend-lock hysteresis.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window (W samples)

    Returns:
        y: Filtered output signal with length = len(x) - window_size + 1
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    r = _estimate_meas_var(x)
    # slightly raised q_base so genuine dynamics still track under the cap
    q_level = 4e-3 * r + 1e-7
    q_slope = 8e-5 * r + 1e-9
    kf = CappedKalmanTrendLock(r, q_level, q_slope,
                               vel_shrink=0.35, trend_lock_factor=1.2)

    warmup = window_size - 1
    for i in range(warmup):
        kf.step(x[i])

    n_out = len(x) - window_size + 1
    y = np.empty(n_out)
    for j in range(n_out):
        y[j] = kf.step(x[warmup + j])

    # light 3-tap post-smoothing: suppresses residual spurious reversals
    # with negligible phase delay (symmetric kernel)
    if n_out >= 3:
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