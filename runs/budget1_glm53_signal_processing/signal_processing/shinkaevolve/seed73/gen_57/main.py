# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Staged pipeline architecture (crossover of linear-edge pipeline and quadratic
leading-edge regression):
  Stage 1: ImpulseDespike     - pre-filter impulse-like noise-sized outliers
  Stage 2: QuadLeadingEdgeFit - zero-lag weighted QUADRATIC regression at
                                window edge, fully vectorized via convolutions
  Stage 3: TrendLock          - EMA slope damping to suppress reversals
  Stage 4: LocalAdaptiveSnap  - locally adaptive slope-churn suppression
  Stage 5: light 3-tap smoothing (negligible phase delay)
"""
import numpy as np


class ImpulseDespike:
    """Stage 1: remove impulse-like, noise-sized steps from raw input."""

    def __init__(self, strength=0.5, factor=2.0):
        self.strength = strength
        self.factor = factor

    def process(self, xf):
        n = len(xf)
        if n < 5:
            return xf
        dx = np.diff(xf)
        sig_x = 1.4826 * np.median(np.abs(dx - np.median(dx))) + 1e-12
        pad_l = np.concatenate((xf[:2][::-1], xf, xf[-2:][::-1]))
        stacked = np.stack([pad_l[0:n], pad_l[1:n + 1], pad_l[2:n + 2],
                           pad_l[3:n + 3], pad_l[4:n + 4]])
        med5 = np.median(stacked, axis=0)
        dev = np.abs(xf - med5)
        snap_mask = dev < self.factor * sig_x
        out = xf.copy()
        out[snap_mask] = (1 - self.strength) * out[snap_mask] + self.strength * med5[snap_mask]
        return out


class QuadLeadingEdgeFit:
    """
    Stage 2: weighted QUADRATIC regression per window, evaluated at the
    window's leading edge (t=0) -> zero group delay while capturing local
    curvature. All windowed moments computed via convolutions (O(n log n)-ish,
    vectorized); robust IRLS refinement only where the pass-1 weighted
    residual is anomalous.
    """

    def __init__(self, ema_span=2.0, robust_factor=2.0):
        self.ema_span = ema_span
        self.robust_factor = robust_factor

    def _solve_quad(self, w, Sx, Stx, Sttx, pre):
        """Solve weighted normal equations given moment sums."""
        S1, St, St2, St3, St4 = pre
        A = np.array([[St4, St3, St2],
                      [St3, St2, St],
                      [St2, St, S1]])
        rhs = np.array([Sttx, Stx, Sx])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.array([0.0, 0.0, Sx / S1 if S1 > 1e-12 else 0.0])
        return sol  # value at t=0 is sol[2]

    def process(self, xf, window_size, output_length):
        weights = np.exp(np.linspace(-self.ema_span, 0, window_size))
        weights /= np.sum(weights)
        t = np.arange(window_size, dtype=float) - (window_size - 1)
        t2 = t * t

        # Precomputed weight moments
        S1 = np.sum(weights)
        St = np.sum(weights * t)
        St2 = np.sum(weights * t2)
        St3 = np.sum(weights * t2 * t)
        St4 = np.sum(weights * t2 * t2)
        pre = (S1, St, St2, St3, St4)

        # Vectorized windowed moments via convolution
        kern = weights[::-1]
        Sx = np.convolve(xf, kern, mode="valid")[:output_length]
        Stx = np.convolve(xf, (weights * t)[::-1], mode="valid")[:output_length]
        Sttx = np.convolve(xf, (weights * t2)[::-1], mode="valid")[:output_length]

        # Fit quadratic per window (loop over output, but solve is 3x3 and cheap;
        # vectorize the common case via batched solve)
        A = np.array([[St4, St3, St2],
                      [St3, St2, St],
                      [St2, St, S1]])
        try:
            rhs = np.stack([Sttx, Stx, Sx], axis=0)  # (3, m)
            sols = np.linalg.solve(A, rhs)            # (3, m)
            a, b, y = sols[0], sols[1], sols[2]
            slopes = b  # slope at t=0
        except np.linalg.LinAlgError:
            y = Sx / S1
            slopes = np.zeros_like(y)

        # Vectorized weighted residual RMS per window:
        # sum w*(x - fit)^2 with fit = a*t^2 + b*t + c
        win_sq = np.convolve(xf * xf, kern, mode="valid")[:output_length]
        # E[(x - fit)^2] = Ex2 - 2a*Sttx/S1 - 2b*Stx/S1 - 2c*Sx/S1 + (a^2 St4 + 2ab St3 + (b^2+2ac) St2 + 2bc St + c^2 S1)/S1
        quad_fit_var = (a * a * St4 + 2 * a * b * St3 +
                        (b * b + 2 * a * y) * St2 + 2 * b * y * St + y * y * S1) / S1
        cross = (a * Sttx + b * Stx + y * Sx) / S1
        resid_var = np.maximum(win_sq / S1 - 2.0 * cross + quad_fit_var, 0.0)
        resid_scale = np.sqrt(resid_var)

        global_scale = np.median(resid_scale)
        bad = np.where(resid_scale > self.robust_factor * global_scale + 1e-12)[0]

        # Targeted robust reweighting only where pass-1 fits badly
        for i in bad:
            window = xf[i:i + window_size]
            fit = a[i] * t2 + b[i] * t[i] + y[i] if 'b' in dir() else None
            fit = a[i] * t2 + slopes[i] * t + y[i]
            resid = window - fit
            scale = np.std(resid)
            rw = weights.copy()
            if scale > 1e-12:
                big = np.abs(resid) > 2.0 * scale
                rw[big] *= scale / (np.abs(resid[big]) + 1e-12)
            Sx2 = np.sum(rw * window)
            Stx2 = np.sum(rw * t * window)
            Sttx2 = np.sum(rw * t2 * window)
            sol2 = self._solve_quad(rw, Sx2, Stx2, Sttx2,
                                    (np.sum(rw), np.sum(rw * t), np.sum(rw * t2),
                                     np.sum(rw * t2 * t), np.sum(rw * t2 * t2)))
            y[i] = sol2[2]
            slopes[i] = sol2[1]

        return y, slopes


class TrendLock:
    """
    Stage 3.5: trend-lock slope damping.

    EMA-smooth the per-window slope sequence to suppress noise-induced
    directional reversals, then nudge the (already zero-lag) level estimate
    by a fraction of the slope disagreement. Real trend changes survive
    because the level y[i] itself carries no group delay.
    """

    def __init__(self, beta=0.55, gain=0.35):
        self.beta = beta
        self.gain = gain

    def process(self, y, raw_slopes):
        m = len(y)
        if m == 0:
            return y
        locked = np.empty(m)
        ls = raw_slopes[0]
        for i in range(m):
            if i > 0:
                ls = self.beta * ls + (1.0 - self.beta) * raw_slopes[i]
            locked[i] = ls
        return y + self.gain * (locked - raw_slopes)


class LocalAdaptiveSnap:
    """
    Stage 4: locally adaptive slope-churn suppression.

    Rolling robust sigma over +/-block-sample neighborhoods of the smoothed
    output's diffs; sub-noise steps get snapped toward a 3-tap median.
    """

    def __init__(self, block=25, k=0.65, blend=0.55):
        self.block = block
        self.k = k
        self.blend = blend

    def process(self, y):
        m = len(y)
        if m <= 2:
            return y
        d = np.diff(y)
        dm = len(d)

        B = min(self.block, max(3, dm))
        pad_n = B // 2
        dpad = np.concatenate((np.full(pad_n, d[0]), d, np.full(pad_n, d[-1])))
        windows = np.lib.stride_tricks.sliding_window_view(dpad, B)
        local_sigma = 1.4826 * np.median(
            np.abs(windows - np.median(windows, axis=1, keepdims=True)), axis=1
        ) + 1e-12
        local_sigma = local_sigma[:dm]

        pad = np.concatenate(([y[0]], y, [y[-1]]))
        med3 = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)

        thresh = self.k * local_sigma
        small = np.abs(d) < thresh
        idx = np.where(small)[0] + 1
        out = y.copy()
        out[idx] = (1.0 - self.blend) * out[idx] + self.blend * med3[idx]
        return out


class AdaptiveSignalPipeline:
    """Composable staged pipeline with configurable stages."""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.despike = ImpulseDespike(strength=0.5, factor=2.0)
        self.fitter = QuadLeadingEdgeFit(ema_span=2.0, robust_factor=2.0)
        self.trendlock = TrendLock(beta=0.55, gain=0.35)
        self.snapper = LocalAdaptiveSnap(block=25, k=0.65, blend=0.55)

    def run(self, x):
        xf = np.asarray(x, dtype=float)
        n = len(xf)
        if n < self.window_size:
            raise ValueError(
                f"Input signal length ({n}) must be >= window_size ({self.window_size})"
            )
        output_length = n - self.window_size + 1

        xf = self.despike.process(xf)
        y, slopes = self.fitter.process(xf, self.window_size, output_length)
        y = self.trendlock.process(y, slopes)
        y = self.snapper.process(y)

        # Light 3-tap smoothing: negligible phase delay, fewer spurious reversals
        if len(y) >= 3:
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
    xf = np.asarray(x, dtype=float)
    n = len(xf)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")
    c = np.cumsum(np.insert(xf, 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced pipeline: despike -> quadratic leading-edge robust fit ->
    trend lock -> local adaptive snap -> light 3-tap smoothing.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    return AdaptiveSignalPipeline(window_size).run(x)


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