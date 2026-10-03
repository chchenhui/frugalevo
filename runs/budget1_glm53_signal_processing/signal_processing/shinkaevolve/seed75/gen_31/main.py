# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

Zero-phase multi-pass polynomial (Savitzky-Golay) smoothing cascade with
robust median prefilter and derivative-based lead compensation.
"""
import numpy as np


def _sg_kernel(half, order=3):
    """Savitzky-Golay smoothing kernel (center-tap value) for half-width `half`."""
    size = 2 * half + 1
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vstack([t ** k for k in range(order + 1)]).T
    # Row of pseudo-inverse that evaluates the fitted polynomial at t=0
    c = np.linalg.pinv(A)[0]
    return c


def _sg_smooth(sig, half, order=3):
    """Zero-phase SG smoothing with reflect-padding. Output length == input."""
    n = len(sig)
    half = max(1, min(half, max(1, n // 2)))
    if n < 2 * half + 1:
        return sig.copy()
    c = _sg_kernel(half, order)
    pad = np.concatenate([
        2 * sig[0] - sig[1:half + 1][::-1],
        sig,
        2 * sig[-1] - sig[n - 2:n - half - 2:-1][::-1],
    ])
    out = np.convolve(pad, c, mode="valid")
    return out


def _sg_deriv(sig, half):
    """Zero-phase SG first-derivative kernel evaluation."""
    n = len(sig)
    half = max(1, min(half, max(1, n // 2)))
    if n < 2 * half + 1:
        return np.gradient(sig)
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vstack([t ** k for k in range(3)]).T
    c = np.linalg.pinv(A)[1]  # derivative of fitted polynomial at t=0
    pad = np.concatenate([
        2 * sig[0] - sig[1:half + 1][::-1],
        sig,
        2 * sig[-1] - sig[n - 2:n - half - 2:-1][::-1],
    ])
    return np.convolve(pad, c, mode="valid")


def _median_filter(sig, k=3):
    """Short running-median filter (kills impulses, zero phase for odd k)."""
    if k <= 1 or len(sig) < k:
        return sig.copy()
    half = k // 2
    pad = np.concatenate([
        2 * sig[0] - sig[1:half + 1][::-1],
        sig,
        2 * sig[-1] - sig[n_ := len(sig) - 2: n_ - half - 2:-1][::-1] if len(sig) > half + 1 else sig[-1:] * half,
    ])
    out = np.array([np.median(pad[i:i + k]) for i in range(len(sig))])
    return out


def adaptive_filter(x, window_size=20):
    """
    Baseline adaptive filter: vectorized moving average (same contract).

    Returns y with length = len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})")
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[window_size:] - c[:-window_size]) / window_size
    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Zero-phase triple-pass Savitzky-Golay cascade with median prefilter
    and derivative-based lead compensation.

    Pipeline:
      1. 3-tap median prefilter (impulse rejection, zero phase).
      2. Three cascaded zero-phase SG passes with a widening window ladder
         derived from window_size (unconditional, full-signal).
      3. Small lead compensation using the zero-phase smoothed derivative
         to cancel residual windowing lag.
      4. End-aligned extraction (zero phase => no delay to compensate).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(
            f"Input signal length ({n}) must be >= window_size ({window_size})")
    output_length = n - window_size + 1

    # ---- Stage 1: robust median prefilter (impulse/outlier rejection) ----
    s = _median_filter(x, 3)

    # ---- Stage 2: cascaded zero-phase SG passes (widening ladder) ----
    sg_half = max(3, window_size // 2)
    h1 = max(2, sg_half // 3)       # narrow: kills high-freq noise
    h2 = max(2, sg_half // 2)       # medium: smooths residual jitter
    h3 = max(2, (2 * sg_half) // 3) # wide: polishes, targets smoothness

    y = _sg_smooth(s, h1, order=3)
    y = _sg_smooth(y, h2, order=3)
    y = _sg_smooth(y, h3, order=2)

    # ---- Stage 3: derivative-based lead compensation (recovers lag) ----
    hd = max(2, sg_half // 2)
    slope = _sg_deriv(y, hd)
    # Estimate noise level to gate the lead strength (avoid boosting noise)
    resid = x - y
    r = np.median(np.abs(resid)) * 1.4826 + 1e-12
    sig_scale = np.std(y) + 1e-12
    # lead scales with window but is capped; smaller when signal is noisy
    lead = min(3.0, max(0.0, window_size / 10.0)) * (1.0 / (1.0 + r / (sig_scale + 1e-12)))
    y = y + lead * slope

    # ---- Stage 4: end-aligned extraction (zero phase => minimal lag) ----
    out = y[-output_length:]
    return np.asarray(out)


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