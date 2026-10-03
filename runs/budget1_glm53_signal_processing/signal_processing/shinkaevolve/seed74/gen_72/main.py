# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Three-stage pipeline architecture:
  Stage 1: RobustStatistics  - MAD-based noise & dynamics estimation
  Stage 2: WhittakerStage     - banded penalized-LS smoothing, O(n)
  Stage 3: FlipEliminator     - zero-phase sub-threshold trend-reversal
           removal (true flip elimination, not attenuation)

Zero-phase throughout. Output contract: y[i] corresponds to input time
i + window_size - 1, len(y) = len(x) - window_size + 1.
"""
import numpy as np


# ---------------- Stage 1: robust statistics ----------------

class RobustStats:
    """Estimates measurement noise and signal acceleration levels."""

    def __init__(self, x, window_size):
        self.x = x
        self.n = x.size
        self.scale = max(float(np.std(x)), 1e-12)
        d1 = np.diff(x)
        s = np.median(np.abs(d1 - np.median(d1))) / (0.6745 * np.sqrt(2.0))
        if not np.isfinite(s) or s <= 0.0:
            s = self.scale
        self.sigma_r = float(np.clip(s, 1e-4 * self.scale, self.scale))
        self.R = self.sigma_r ** 2
        self.sigma_acc, self.noise_d2 = self._dynamics(window_size)

    def _dynamics(self, window_size):
        n = self.n
        w_r = int(max(3, min(window_size, n // 3)))
        if w_r % 2 == 0:
            w_r += 1
        rough = np.convolve(self.x, np.ones(w_r) / w_r, mode="valid")
        noise_d2 = (self.sigma_r / np.sqrt(w_r)) * np.sqrt(6.0)
        if rough.size > 2:
            d2 = rough[2:] - 2.0 * rough[1:-1] + rough[:-2]
            mad2 = np.median(np.abs(d2 - np.median(d2))) / 0.6745
            var_acc = mad2 * mad2 - noise_d2 * noise_d2
        else:
            var_acc = 0.0
        floor_acc = 0.08 * self.sigma_r
        if var_acc < floor_acc * floor_acc:
            var_acc = floor_acc * floor_acc
        return float(min(np.sqrt(var_acc), 0.5 * self.sigma_r)), noise_d2


# ---------------- Stage 2: banded Whittaker smoother ----------------

def _solve_penta_banded(main, a1, a2, b):
    """Thomas-like pentadiagonal symmetric solver, O(n).

    main: diagonal; a1: first off-diagonal; a2: second off-diagonal.
    """
    n = len(b)
    if n < 3:
        return b.copy()
    d = main.astype(float).copy()
    e = a1.astype(float).copy()
    f = a2.astype(float).copy()
    rhs = b.astype(float).copy()
    # Forward elimination
    for i in range(1, n):
        if i == 1:
            m_ = e[0] / d[0]
            d[1] -= m_ * e[0]
            rhs[1] -= m_ * rhs[0]
        else:
            m1 = e[i - 1] / d[i - 2]
            d[i - 1] -= m1 * f[i - 2]
            e[i - 1] -= m1 * e[i - 2]
            rhs[i - 1] -= m1 * rhs[i - 2]
            m2 = e[i] / d[i - 1]
            d[i] -= m2 * e[i]
            rhs[i] -= m2 * rhs[i - 1]
    # Back substitution
    x = np.zeros(n)
    x[n - 1] = rhs[n - 1] / d[n - 1]
    if n >= 2:
        x[n - 2] = (rhs[n - 2] - e[n - 2] * x[n - 1]) / d[n - 2]
    for i in range(n - 3, -1, -1):
        x[i] = (rhs[i] - e[i] * x[i + 1] - f[i] * x[i + 2]) / d[i]
    return x


class WhittakerStage:
    """Second-difference penalized least squares with uniform lambda.

    A = I + lam * D2^T D2 (pentadiagonal). Constant/linear trends pass
    through exactly -> no lag, no attenuation.
    """

    @staticmethod
    def run(y, lam):
        n = len(y)
        if n < 4 or lam <= 0:
            return y.copy()
        main = np.full(n, 6.0 * lam) + 1.0
        main[0] = 1.0 + lam
        main[1] = 1.0 + 5.0 * lam
        main[-1] = 1.0 + lam
        main[-2] = 1.0 + 5.0 * lam
        a1 = np.full(max(n - 1, 1), -4.0 * lam)
        a2 = np.full(max(n - 2, 1), lam)
        try:
            out = _solve_penta_banded(main, a1, a2, y)
            if np.all(np.isfinite(out)):
                return out
        except Exception:
            pass
        # Dense fallback
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            D[i, i] = 1.0
            D[i, i + 1] = -2.0
            D[i, i + 2] = 1.0
        A = np.eye(n) + lam * (D.T @ D)
        try:
            return np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, y, rcond=None)[0]


# ---------------- Stage 3: flip eliminator ----------------

def _eliminate_flips_pass(sig, thresh, keep=0.0):
    """One causal sweep of true flip elimination.

    Walks the difference sequence; when a step is sub-threshold AND its
    sign opposes the previous *pre-damp* step, the conflicting step is
    eliminated (hold the line) or averaged (keep=0.5), so the output
    difference sequence genuinely does not reverse.
    """
    d = np.diff(sig)
    out = np.empty_like(sig)
    out[0] = sig[0]
    prev_d = d[0] if d.size else 0.0
    for k in range(d.size):
        dk = d[k]
        if abs(dk) < thresh and prev_d != 0.0 and np.sign(dk) == -np.sign(prev_d):
            dk = keep * dk  # 0 => true elimination; 0.5 => averaging retry
        out[k + 1] = out[k] + dk
        prev_d = d[k]  # track pre-damp difference to avoid zero-stall
    return out


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / den) if den > 0 else 1.0


class FlipEliminator:
    """Zero-phase removal of sub-threshold trend reversals.

    Forward sweep + mirrored backward sweep, averaged => zero phase.
    Guarded by corr >= 0.97 with keep=0.5 retry to preserve genuine
    dynamics.
    """

    @staticmethod
    def run(y, sigma_r, c=1.0, max_retry=3):
        n = y.size
        if n < 4:
            return y.copy()
        dy = np.diff(y)
        s = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0))
        if not np.isfinite(s) or s <= 0:
            return y.copy()
        thresh = c * s
        y0 = y.copy()
        keep = 0.0
        for _ in range(max_retry):
            fwd = _eliminate_flips_pass(y0, thresh, keep)
            bwd = _eliminate_flips_pass(y0[::-1], thresh, keep)[::-1]
            out = 0.5 * (fwd + bwd)
            if _corr(out, y0) >= 0.97:
                return out
            keep = 0.5 if keep == 0.0 else min(keep + 0.25, 0.9)
        return y0


# ---------------- Pipeline orchestration ----------------

def whittaker_robust_filter(x, window_size=20):
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})"
        )
    if n <= 3:
        return x[window_size - 1:].copy() if n >= window_size else x.copy()

    # Stage 1: robust statistics
    st = RobustStats(x, window_size)

    # Global smoothing parameter from noise/dynamics ratio.
    lam = float(np.clip(st.R / max(st.sigma_acc ** 2, 1e-18), 1.0, 5e3))

    # Stage 2: Whittaker smoothing (uniform lambda, zero-phase).
    y = WhittakerStage.run(x, lam)

    # Stage 3a: flip elimination on the smoothed signal.
    y = FlipEliminator.run(y, st.sigma_r)

    # Stage 3b: residual-scale-aware second polish: re-estimate residual
    # noise after smoothing; if still noisy, apply a stronger flip pass.
    dy = np.diff(y)
    if dy.size > 2:
        r_std = np.median(np.abs(dy - np.median(dy))) / (0.6745 * np.sqrt(2.0))
    else:
        r_std = 0.0
    if r_std > 0 and st.sigma_r > 0 and st.sigma_r / r_std > 2.0:
        y = FlipEliminator.run(y, st.sigma_r, c=1.5)

    # Slice to output contract.
    y = y[window_size - 1:].copy()
    expected = n - window_size + 1
    if y.shape[0] != expected:
        y = y[:expected]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline entry — routes to the staged pipeline."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Enhanced entry — staged Whittaker + flip-elimination pipeline."""
    return whittaker_robust_filter(x, window_size)


def kalman_rts_filter(x, window_size=20):
    """Compatibility alias — same pipeline."""
    return whittaker_robust_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main signal processing function that applies the selected algorithm.

    Args:
        input_signal: Input time series data
        window_size: Window size for processing
        algorithm_type: Type of algorithm to use ("basic", "enhanced", or "rts")

    Returns:
        Filtered signal
    """
    return whittaker_robust_filter(input_signal, window_size)

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