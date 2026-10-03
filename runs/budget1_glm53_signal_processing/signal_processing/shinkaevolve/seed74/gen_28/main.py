# EVOLVE-BLOCK-START
import numpy as np


class AdaptiveHybridKalmanZLAFilter:
    """
    Hybrid crossover filter:
      - Kalman-like level+slope state estimator (zero-lag adaptive smoothing)
      - SNR-adaptive measurement noise (gain scheduling)
      - Polynomial lookahead gating for reversal decisions
      - Hysteresis trend detector with false-reversal penalty
    Inputs:  1D numpy array (or list) of samples
    Output:  dict with smoothed signal, trend (+1/0/-1), reversal flags
    """

    def __init__(self, q_level=0.01, q_slope=1e-4, r_base=1.0,
                 snr_window=16, hysteresis=0.15, lookahead=4,
                 poly_order=2, reversal_penalty=0.5):
        self.q_level = q_level
        self.q_slope = q_slope
        self.r_base = r_base
        self.snr_window = snr_window
        self.hysteresis = hysteresis
        self.lookahead = lookahead
        self.poly_order = poly_order
        self.reversal_penalty = reversal_penalty

    # ---------- SNR-adaptive measurement noise ----------
    def _adaptive_r(self, x):
        n = len(x)
        r = np.empty(n)
        win = self.snr_window
        for i in range(n):
            lo = max(0, i - win + 1)
            seg = x[lo:i + 1]
            if len(seg) < 3:
                r[i] = self.r_base
                continue
            # detrend locally (first-order) to estimate noise
            t = np.arange(len(seg))
            slope = np.polyfit(t, seg, 1)[0]
            resid = seg - (slope * t + seg[0])
            noise = np.std(resid) + 1e-9
            signal = np.std(seg) + 1e-9
            snr = signal / noise
            # high SNR -> trust measurements (small R); low SNR -> smooth more
            r[i] = self.r_base / (1.0 + snr)
        return r

    # ---------- Polynomial lookahead slope (gate only, no lag added) ----------
    def _lookahead_slope(self, smoothed, i):
        lo = max(0, i - self.lookahead)
        seg = smoothed[lo:i + 1]
        if len(seg) < 2:
            return 0.0
        t = np.arange(len(seg))
        coeffs = np.polyfit(t, seg, min(self.poly_order, len(seg) - 1))
        deriv = np.polyder(np.poly1d(coeffs))
        return float(deriv(len(seg) - 1))

    # ---------- Main filter (online-capable, O(1) per step) ----------
    def filter(self, data):
        x = np.asarray(data, dtype=float).ravel()
        n = len(x)
        if n == 0:
            return {"smoothed": np.array([]), "trend": np.array([], int),
                    "reversals": np.array([], bool)}

        r_series = self._adaptive_r(x)

        # --- Kalman-like 2-state (level, slope) filter ---
        level = x[0]
        slope = 0.0
        p = np.eye(2) * 1.0
        smoothed = np.empty(n)
        for i in range(n):
            # predict
            level_pred = level + slope
            slope_pred = slope
            p[0, 0] += self.q_level + 2 * self._p_prev_01 if i > 0 else self.q_level
            p[0, 1] += getattr(self, "_ps_01", 0.0)
            p[1, 0] = p[0, 1]
            p[1, 1] += self.q_slope
            p = 0.5 * (p + p.T)  # symmetrize for stability

            # update with adaptive R
            r = r_series[i]
            y = x[i] - level_pred
            s = p[0, 0] + r + 1e-12
            k0 = p[0, 0] / s
            k1 = p[1, 0] / s
            level = level_pred + k0 * y
            slope = slope_pred + k1 * y
            # Joseph-form covariance update (numerically stable)
            ik = np.eye(2) - np.outer([k0, k1], [1.0, 0.0])
            p = ik @ p @ ik.T
            p[0, 0] += k0 * k0 * r

            self._p_prev_01 = p[0, 1]
            self._ps_01 = p[0, 1]
            smoothed[i] = level

        # --- Trend with hysteresis + lookahead gating + reversal penalty ---
        threshold = self.hysteresis * np.std(np.diff(smoothed) + 1e-12)
        trend = np.zeros(n, dtype=int)
        reversals = np.zeros(n, dtype=bool)
        cur = 0
        for i in range(1, n):
            la_slope = self._lookahead_slope(smoothed, i)
            eff = (1.0 - self.reversal_penalty) * la_slope \
                  + self.reversal_penalty * (smoothed[i] - smoothed[i - 1])
            if cur >= 0 and eff < -threshold:
                if cur != -1:
                    reversals[i] = True
                cur = -1
            elif cur <= 0 and eff > threshold:
                if cur != 1:
                    reversals[i] = True
                cur = 1
            # else keep current trend (hysteresis dead-band)
            trend[i] = cur

        return {"smoothed": smoothed, "trend": trend, "reversals": reversals}
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