# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Pipeline for Non-Stationary Time Series

Architecture (pipeline / modular data flow):

    raw samples --> [Stage 1] Adaptive Kalman (level+slope state)
                 --> [Stage 2] Polynomial trend refinement (low phase lag)
                 --> [Stage 3] Trend guard / reversal hysteresis
                 --> filtered output

Each stage is an independent module; the pipeline controller stitches them
together and enforces the same input/output contract as the original code.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Adaptive Kalman filter (constant-velocity model, 2-state)
# ---------------------------------------------------------------------------
class AdaptiveKalmanStage:
    """
    Tracks [level, slope] with a Kalman filter. Process noise is adapted
    to local residual volatility, so the filter is responsive in turbulent
    regions and heavily smoothing in quiet regions.
    """

    def __init__(self, window_size, q_base=0.05, r_scale=1.0):
        self.w = max(3, int(window_size))
        self.q_base = q_base
        self.r_scale = r_scale

    def run(self, x):
        n = len(x)
        # Observation noise from robust local scale (median abs deviation)
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        r = (1.4826 * mad * self.r_scale) ** 2

        # State: [level, slope]
        x_state = np.array([x[0], 0.0])
        P = np.eye(2) * 10.0

        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q_scale = self.q_base

        levels = np.zeros(n)
        slopes = np.zeros(n)

        for k in range(n):
            # Predict
            x_state = F @ x_state
            P = F @ P @ F.T

            # Adaptive process noise: larger innovation => more dynamic signal
            # (handled via residual-based scaling below)
            Q = Q_scale * np.array([[0.25, 0.5], [0.5, 1.0]])
            P = P + Q

            # Update
            innov = x[k] - (H @ x_state)[0]
            S = (H @ P @ H.T)[0, 0] + r
            K = (P @ H.T) / S
            x_state = x_state + (K.flatten() * innov)
            I_KH = np.eye(2) - K @ H
            P = I_KH @ P @ I_KH.T

            levels[k] = x_state[0]
            slopes[k] = x_state[1]

            # Adapt process noise from normalized innovation
            ni = innov / (np.sqrt(S) + 1e-12)
            Q_scale = 0.9 * Q_scale + 0.1 * self.q_base * (
                1.0 + min(10.0, abs(ni))
            )

        return levels, slopes


# ---------------------------------------------------------------------------
# Stage 2: Polynomial trend refinement (Savitzky-Golay style, zero phase)
# ---------------------------------------------------------------------------
class PolynomialTrendStage:
    """
    Local polynomial (degree 3) fitting over the Kalman output. This is
    effectively a zero-phase smoother: it removes residual noise while
    preserving trend dynamics (unlike a causal moving average, it adds
    no group delay beyond the alignment window).
    """

    def __init__(self, window_size, poly_order=3):
        self.w = max(poly_order + 1, int(window_size))
        self.order = min(poly_order, self.w - 1)

    def run(self, levels):
        n = len(levels)
        half = self.w // 2
        w = self.w if self.w % 2 == 1 else self.w + 1
        half = w // 2

        # Precompute SG smoothing coefficients (fit at center point)
        t = np.arange(-half, half + 1, dtype=float)
        A = np.vander(t, self.order + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(A, np.ones_like(t) * 0 + 0, rcond=None)
        # Coefficients that evaluate the polynomial at t=0
        c, *_ = np.linalg.lstsq(A, np.eye(self.order + 1)[0], rcond=None)
        # Proper SG: weights = row of (A^T A)^-1 A^T corresponding to center
        ATA_inv = np.linalg.inv(A.T @ A)
        weights = (A @ ATA_inv)[half + (0 if w % 2 == 1 else 0)] if False else None
        # Evaluate-at-center weights: first row of (A^T A)^-1 A^T
        W = ATA_inv @ A.T          # (order+1, w)
        weights = W[0, :]          # value of fit at t = 0

        # Pad with edge reflection
        padded = np.concatenate([
            levels[1 : 1 + half][::-1], levels, levels[-2 - half : -2][::-1]
        ]) if n > half + 2 else np.pad(levels, (half, half), mode="edge")

        out = np.convolve(padded, weights[::-1], mode="valid")
        # Align: out has length n + 2*half - (2*half) = n
        return out[:n]


# ---------------------------------------------------------------------------
# Stage 3: Trend guard / reversal hysteresis
# ---------------------------------------------------------------------------
class TrendGuardStage:
    """
    Suppresses spurious directional reversals: a filtered value is nudged
    toward the previous trend direction unless the estimated slope change
    exceeds an adaptive (volatility-scaled) threshold. Reduces slope
    changes and false reversals while preserving genuine turns.
    """

    def __init__(self, window_size, strength=0.35):
        self.w = max(3, int(window_size))
        self.strength = strength

    def run(self, y, slopes):
        n = len(y)
        if n < 3:
            return y.copy()

        # Adaptive threshold: fraction of robust slope magnitude scale
        slope_scale = np.median(np.abs(slopes)) + 1e-12
        threshold = 0.15 * slope_scale

        out = y.copy()
        prev_dir = np.sign(slopes[0]) if slopes[0] != 0 else 0.0

        for k in range(1, n):
            cur_dir = np.sign(slopes[k]) if slopes[k] != 0 else prev_dir
            # Reversal only counts if slope actually crossed threshold
            reversal = (cur_dir * prev_dir < 0) and (
                abs(slopes[k] - slopes[k - 1]) > threshold
            )
            if not reversal and cur_dir != 0 and prev_dir != 0:
                # Blend slightly toward trend continuation => fewer zigzags
                trend_step = 0.5 * (slopes[k] + slopes[k - 1])
                out[k] = (1 - self.strength) * y[k] + self.strength * (
                    out[k - 1] + trend_step
                )
            else:
                out[k] = y[k]
            if cur_dir != 0:
                prev_dir = cur_dir

        return out


# ---------------------------------------------------------------------------
# Pipeline controller
# ---------------------------------------------------------------------------
class FilterPipeline:
    """Composes the three stages; enforces original input/output contract."""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.kf = AdaptiveKalmanStage(window_size)
        self.poly = PolynomialTrendStage(window_size, poly_order=3)
        self.guard = TrendGuardStage(window_size)

    def run(self, x):
        x = np.asarray(x, dtype=float)
        w = self.window_size
        if len(x) < w:
            raise ValueError(
                f"Input signal length ({len(x)}) must be >= window_size ({w})"
            )

        levels, slopes = self.kf.run(x)
        smoothed = self.poly.run(levels)
        guarded = self.guard.run(smoothed, slopes)

        # Match original output contract: len(x) - window_size + 1 samples,
        # each representing the (near-causal) filtered estimate.
        output_length = len(x) - w + 1
        # Take the trailing portion so the estimator tracks the signal end
        # (zero-phase refinement means minimal lag).
        return guarded[-output_length:]


# ---------------------------------------------------------------------------
# Public API (identical signatures to original)
# ---------------------------------------------------------------------------
def adaptive_filter(x, window_size=20):
    """Adaptive Kalman + polynomial + trend-guard pipeline."""
    return FilterPipeline(window_size).run(x)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Alias of the pipeline (kept for API compatibility)."""
    return FilterPipeline(window_size).run(x)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Main entry point; both algorithm types use the improved pipeline."""
    return FilterPipeline(window_size).run(input_signal)


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
