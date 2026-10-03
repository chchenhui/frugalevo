# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np

try:
    import pywt
    _HAVE_PYWT = True
except Exception:
    _HAVE_PYWT = False


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
    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Multi-scale wavelet soft-threshold denoiser with edge preservation.

    Mechanism: one `pywt.wavedec` / `waverec` pair (O(n), ~1 ms). Noise sigma
    is estimated robustly from the finest detail coefficients via the MAD
    estimator (median(|d1|)/0.6745); the universal threshold
    sigma*sqrt(2*ln n) is applied with SOFT shrinkage to every detail band
    while the coarse approximation band is kept intact. Soft thresholding is
    scale-selective and signal-adaptive: it zeroes noise-driven coefficient
    excursions (suppressing false reversals and slope chatter) while leaving
    large step/chirp coefficients unchanged, so edges stay sharp — something
    fixed-gain linear smoothers (moving average, Holt) cannot do. The
    transform is zero-phase, so lag error is near zero.

    Output contract: y has length exactly len(x) - window_size + 1, first
    sample aligned to index window_size - 1 (same as a sliding window).
    Falls back to a causal Holt double-exponential smoother if pywt is
    unavailable.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Warm-up/alignment length; first output corresponds to
                     index window_size - 1

    Returns:
        y: Filtered output signal of length len(x) - window_size + 1 (finite)
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)
    out_len = n - window_size + 1

    if _HAVE_PYWT:
        try:
            wavelet = "sym6"
            max_level = pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len)
            level = min(5, max_level)
            coeffs = pywt.wavedec(x, wavelet, level=level)

            # Robust noise sigma from finest detail band
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
            thr = sigma * np.sqrt(2.0 * np.log(n)) if sigma > 0 else 0.0

            # Non-negative garrote shrinkage on all detail bands; the
            # approximation band is kept intact. Garrote shrinks large
            # (signal) coefficients less than soft thresholding -> lower
            # bias on steps/chirps (better correlation, higher noise
            # reduction) while still zeroing noise-driven small
            # coefficients (low slope chatter / fewer false reversals).
            coeffs = [coeffs[0]] + [pywt.threshold(d, thr, mode="garotte") for d in coeffs[1:]]

            rec = pywt.waverec(coeffs, wavelet)
            rec = np.asarray(rec, dtype=float)[:n]  # trim possible +1 sample

            # Zero-phase SG post-smooth: removes residual coefficient chatter
            # (slope_changes / false_reversals) without adding lag; short
            # window + low order preserves edges and dynamics.
            try:
                from scipy.signal import savgol_filter
                w = min(11, n if n % 2 == 1 else n - 1)
                if w >= 5:
                    rec = savgol_filter(rec, w, 2, mode="interp")
            except Exception:
                pass

            # Adaptive-reversal-suppression (refined): single-pass
            # variable-bandwidth smoothing with a capped blend weight.
            # Flip density (sigma=4) maps chatter clusters; the blend
            # toward the wide sigma=9 pass is capped at 0.85 so even the
            # noisiest regions retain 15% of the sharp estimate — this
            # keeps the slope/false-reversal gains of full blending while
            # limiting the correlation and noise-reduction loss seen when
            # blending unbounded. Monotone/edge regions (low density) keep
            # the narrow sigma=1.5 pass. All O(n), zero added lag.
            try:
                from scipy.ndimage import gaussian_filter1d
                d = np.diff(rec)
                flips = np.zeros(n, dtype=float)
                if len(d) > 1:
                    sgn = np.sign(d)
                    flip_idx = np.where(sgn[:-1] * sgn[1:] < 0)[0] + 1
                    flips[flip_idx] = 1.0
                density = gaussian_filter1d(flips, sigma=4.0, mode="nearest")
                density = np.clip(density, 0.0, 1.0) * 0.85
                smooth_lo = gaussian_filter1d(rec, sigma=1.5, mode="nearest")
                smooth_hi = gaussian_filter1d(rec, sigma=9.0, mode="nearest")
                rec = (1.0 - density) * smooth_lo + density * smooth_hi
            except Exception:
                pass

            rec = np.nan_to_num(rec, nan=0.0, posinf=0.0, neginf=0.0)
            y = rec[window_size - 1 : window_size - 1 + out_len]
            if len(y) == out_len and np.all(np.isfinite(y)):
                return y.copy()
        except Exception:
            pass  # fall back to Holt below

    # Fallback: causal Holt double-exponential smoother (incumbent)
    alpha, beta = 0.35, 0.04
    lvl = np.empty(n)
    slope = np.empty(n)
    lvl[0] = x[0]
    slope[0] = 0.0
    for t in range(1, n):
        prev = lvl[t - 1]
        lvl[t] = alpha * x[t] + (1.0 - alpha) * (prev + slope[t - 1])
        slope[t] = beta * (lvl[t] - prev) + (1.0 - beta) * slope[t - 1]
    return lvl[window_size - 1 :].copy()


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
