# EVOLVE-BLOCK-START

class FilterStage:
    """Base interface for a pipeline stage."""
    name = "base"

    def apply(self, signal: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ZeroPhaseSavitzkyGolayStage(FilterStage):
    """
    Reflection-padded, zero-phase Savitzky-Golay smoother.
    Kernel is precomputed once: row 0 of pinv(A^T A) A^T (evaluation
    at the window center), applied forward + flipped for zero phase.
    """

    def __init__(self, half_width: int, poly_order: int = 2):
        self.h = max(2, int(half_width))
        self.p = poly_order
        self._kernel = self._build_kernel()

    def _build_kernel(self) -> np.ndarray:
        h, p = self.h, self.p
        idx = np.arange(-h, h + 1, dtype=float)
        A = np.vander(idx, p + 1, increasing=True)
        # row 0 of pinv(A^T A) A^T gives center-evaluation coefficients
        return (np.linalg.pinv(A.T @ A) @ A.T)[0]

    def apply(self, signal: np.ndarray) -> np.ndarray:
        kernel = self._kernel
        h = self.h
        padded = np.concatenate([
            signal[1:h + 1][::-1], signal, signal[-h - 1:-1][::-1]
        ])
        n = signal.shape[0]
        out = np.empty(n, dtype=float)
        # correlation with symmetric kernel == convolution (zero-phase)
        for i in range(n):
            out[i] = np.dot(padded[i:i + 2 * h + 1], kernel)
        return out


class SlopeStabilizationStage(FilterStage):
    """
    Suppress spurious directional reversals: estimate local slope on a
    short window; where the slope sign flips relative to a majority vote
    over a wider neighborhood, blend the sample toward the local trend.
    """
    def __init__(self, short_half: int = 3, wide_half: int = 8, blend: float = 0.5):
        self.sh = max(1, short_half)
        self.wh = max(self.sh + 1, wide_half)
        self.blend = blend

    def apply(self, signal: np.ndarray) -> np.ndarray:
        n = signal.shape[0]
        if n < 2 * self.wh + 1:
            return signal
        # local slopes
        s_short = np.zeros(n)
        s_wide = np.zeros(n)
        for i in range(n):
            a = max(0, i - self.sh)
            b = min(n - 1, i + self.sh)
            s_short[i] = (signal[b] - signal[a]) / max(1, b - a)
            a2 = max(0, i - self.wh)
            b2 = min(n - 1, i + self.wh)
            s_wide[i] = (signal[b2] - signal[a2]) / max(1, b2 - a2)
        out = signal.astype(float).copy()
        # majority sign over wide window around each point
        for i in range(n):
            a = max(0, i - self.wh)
            b = min(n, i + self.wh + 1)
            votes = np.sign(s_short[a:b])
            majority = np.sign(votes.sum())
            if majority != 0 and np.sign(s_short[i]) != majority and s_wide[i] != 0:
                # spurious micro-reversal: soften it
                local_trend = signal[i - 1] + s_wide[i] if i > 0 else signal[i]
                out[i] = (1 - self.blend) * signal[i] + self.blend * local_trend
        return out


class TrendLockStage(FilterStage):
    """
    Preserve genuine trends: local linear fit over a medium window;
    where the smoothed signal deviates from the fitted trend beyond a
    robust sigma threshold, snap partially back toward the trend line.
    Prevents false trend changes without adding lag (fit is centered).
    """
    def __init__(self, half_width: int = 6, snap: float = 0.4, k_sigma: float = 2.0):
        self.h = max(2, half_width)
        self.snap = snap
        self.k = k_sigma

    def apply(self, signal: np.ndarray) -> np.ndarray:
        n = signal.shape[0]
        h = self.h
        if n < 2 * h + 1:
            return signal
        trend = np.empty(n)
        for i in range(n):
            lo = max(0, i - h)
            hi = min(n, i + h + 1)
            x = np.arange(lo, hi, dtype=float) - i
            y = signal[lo:hi]
            # centered linear fit
            slope = np.dot(x, y - y.mean()) / max(np.dot(x, x), 1e-12)
            trend[i] = y.mean() + 0.0  # value of fit at center
        resid = signal - trend
        sigma = np.std(resid) + 1e-12
        mask = np.abs(resid) > self.k * sigma
        out = signal.astype(float).copy()
        out[mask] = (1 - self.snap) * signal[mask] + self.snap * trend[mask]
        return out


class SignalPipeline:
    """
    Orchestrates stages. Core architecture: coarse-to-fine triple
    zero-phase S-G passes (progressively narrower windows) for maximal
    noise reduction, followed by slope stabilization and trend locking.
    """

    def __init__(self, base_half: int = 18):
        sg_half = max(4, base_half)
        h1 = sg_half
        h2 = (2 * sg_half) // 3          # widened second pass
        h3 = max(2, sg_half // 3)        # third pass: attack smoothness
        self.stages = [
            ZeroPhaseSavitzkyGolayStage(h1, poly_order=2),
            ZeroPhaseSavitzkyGolayStage(h2, poly_order=2),
            ZeroPhaseSavitzkyGolayStage(h3, poly_order=2),
            SlopeStabilizationStage(short_half=max(2, sg_half // 6),
                                    wide_half=max(4, sg_half // 2)),
            TrendLockStage(half_width=max(3, sg_half // 3)),
        ]

    def process(self, signal: np.ndarray) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim != 1 or x.shape[0] < 8:
            return x.copy()
        # reflection-pad entire signal once at pipeline boundary
        pad = max(st.h for st in self.stages if hasattr(st, 'h'))
        padded = np.concatenate([x[1:pad + 1][::-1], x, x[-pad - 1:-1][::-1]])
        y = padded
        for stage in self.stages:
            y = stage.apply(y)
        # remove pipeline-level padding
        return y[pad:pad + x.shape[0]]


def filter_signal(signal, window_half: int = 18):
    """
    Public entry point. Maintains original inputs/outputs:
    input: 1-D numpy array (or list) of raw samples
    output: filtered 1-D numpy array, same length, minimal phase delay.
    """
    pipeline = SignalPipeline(base_half=window_half)
    return pipeline.process(signal)

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
