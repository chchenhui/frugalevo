# EVOLVE-BLOCK-START
import numpy as np


class AdaptiveKalmanHybridFilter:
    """
    Hybrid real-time filter combining:
      - Savitzky-Golay style polynomial smoothing (low lag, multi-scale)
      - Adaptive Kalman filtering (process noise adapts to residual volatility)
      - Trend detection with hysteresis (false-reversal suppression)
      - Slope-change damping (spurious directional flip minimization)

    Streaming-friendly: processes samples one at a time with O(window) work.
    """

    def __init__(self, window=11, poly_order=2, base_q=1e-4, r=1.0,
                 reversal_threshold=0.15, damping=0.6, warmup=None):
        self.window = max(window, poly_order + 1)
        self.poly_order = poly_order
        self.base_q = base_q
        self.r = r
        self.reversal_threshold = reversal_threshold
        self.damping = damping
        self.warmup = warmup if warmup is not None else self.window

        # Kalman state (constant-velocity model)
        self.x = np.zeros(2)          # [level, slope]
        self.P = np.eye(2)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.initialized = False

        # Buffers
        self.buf = []                 # raw samples
        self.filt_history = []        # filtered output history
        self.slope_history = []       # estimated slope history
        self.trend = 0                # -1, 0, +1 with hysteresis

    # ---------- Savitzky-Golay style local polynomial fit (causal) ----------
    def _poly_smooth(self, samples):
        n = len(samples)
        w = min(len(samples), self.window)
        seg = np.asarray(samples[-w:], dtype=float)
        if n < self.poly_order + 1:
            return float(np.mean(seg))
        # Fit polynomial over normalized index (most recent sample at t=0)
        t = np.arange(-(w - 1), 1.0)
        coeffs = np.polyfit(t, seg, self.poly_order)
        # Evaluate at t=0 -> zero-lag endpoint estimate of the local fit
        return float(np.polyval(coeffs, 0.0))

    # ---------- Adaptive process noise from residual volatility ----------
    def _adaptive_q(self, residuals):
        if len(residuals) < 2:
            return self.base_q
        vol = np.var(residuals[-self.window:])
        # Scale process noise with observed volatility: faster adaptation in
        # volatile regimes, tighter tracking in quiet regimes.
        return self.base_q * (1.0 + 10.0 * vol / (self.r + 1e-12))

    # ---------- Trend detection with hysteresis (false reversal guard) ----------
    def _update_trend(self, slope):
        if abs(slope) < self.reversal_threshold:
            # dead zone: hold previous trend to avoid noise-induced flips
            return self.trend
        new_trend = 1 if slope > 0 else -1
        if new_trend != self.trend and self.trend != 0:
            # require slope to clearly exceed threshold before flipping
            if abs(slope) < 2.0 * self.reversal_threshold:
                return self.trend
        return new_trend

    # ---------- Main streaming step ----------
    def update(self, sample):
        self.buf.append(float(sample))

        # 1) Polynomial pre-smoothing (multi-scale, low phase delay)
        z = self._poly_smooth(self.buf)

        # 2) Adaptive Kalman update
        if not self.initialized:
            self.x = np.array([z, 0.0])
            self.initialized = True
        else:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T

            # Adaptive process noise from recent innovation residuals
            innovations = [s - f for s, f in
                           zip(self.buf[-self.window:], self.filt_history[-self.window:])
                           if True]
            q = self._adaptive_q(innovations if innovations else [0.0])
            self.P[0, 0] += q
            self.P[1, 1] += q * 0.5

            # Update with Kalman gain
            y = z - (self.H @ self.x)[0]
            S = (self.H @ self.P @ self.H.T)[0, 0] + self.r
            K = (self.P @ self.H.T).flatten() / S
            self.x = self.x + K * y
            I_KH = np.eye(2) - np.outer(K, self.H)
            self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * self.r

        # 3) Slope-change damping: blend Kalman slope with prior slope
        raw_slope = self.x[1]
        if self.slope_history:
            prev = self.slope_history[-1]
            if np.sign(raw_slope) != np.sign(prev) and prev != 0:
                raw_slope = prev * self.damping  # damp spurious flip
        self.slope_history.append(raw_slope)
        self.x[1] = raw_slope

        # 4) Trend with hysteresis
        self.trend = self._update_trend(raw_slope)

        # Output: corrected level (level minus lag bias from slope)
        out = self.x[0] + 0.5 * raw_slope  # lag compensation
        self.filt_history.append(out)
        return out

    # ---------- Batch convenience ----------
    def filter(self, data):
        return np.array([self.update(s) for s in np.asarray(data, dtype=float)])


def filter_signal(data, window=11, poly_order=2, base_q=1e-4, r=1.0,
                  reversal_threshold=0.15, damping=0.6):
    """
    Filter a 1D signal. Returns filtered array with same length as input.
    """
    f = AdaptiveKalmanHybridFilter(window=window, poly_order=poly_order,
                                   base_q=base_q, r=r,
                                   reversal_threshold=reversal_threshold,
                                   damping=damping)
    return f.filter(data)


if __name__ == "__main__":
    # Demo / self-test
    rng = np.random.default_rng(0)
    n = 300
    t = np.arange(n)
    true = np.sin(t / 25.0) + 0.02 * t / n  # non-stationary trend + dynamics
    noisy = true + rng.normal(0, 0.3, n)
    out = filter_signal(noisy)

    print(f"Input length:  {len(noisy)}")
    print(f"Output length: {len(out)}")
    rmse_raw = float(np.sqrt(np.mean((noisy - true) ** 2)))
    rmse_flt = float(np.sqrt(np.mean((out - true) ** 2)))
    print(f"Raw RMSE:    {rmse_raw:.4f}")
    print(f"Filtered RMSE: {rmse_flt:.4f}")
    print(f"Improvement: {(1 - rmse_flt / rmse_raw) * 100:.1f}%")
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