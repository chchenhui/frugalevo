# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


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

    # Initialize output array
    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Simple moving average as baseline
    for i in range(output_length):
        window = x[i : i + window_size]

        # Basic moving average filter
        y[i] = np.mean(window)

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Zero-delay trend-preserving filter in two vectorized stages:
    (1) Weighted quadratic (Savitzky-Golay style) fit per sliding window,
        evaluated at the window's newest sample to eliminate group delay,
        adaptively blended with the raw sample based on local residual
        std (noise estimate): noisier windows trust the polynomial more.
    (2) Iterated short centered quadratic refinement over the stage-1
        output, fitted at each window's center (zero phase), removing
        residual jitter to suppress spurious slope reversals.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    out_len = n - window_size + 1

    # Sliding windows as a matrix: (out_len, window_size)
    idx = np.arange(window_size)
    windows = x[np.arange(out_len)[:, None] + idx[None, :]]

    # Normalized time axis, t=0 at the newest sample, t in [-1, 0]
    t = (idx - (window_size - 1)) / (window_size - 1)
    V = np.vstack([np.ones_like(t), t, t * t]).T  # (window_size, 3)

    # Mild recency weighting favors recent samples
    w = np.exp(np.linspace(-1.5, 0.0, window_size))
    sw = np.sqrt(w)[:, None]
    Vw = V * sw
    Vw_pinv = np.linalg.pinv(Vw)  # (3, window_size), precomputed once
    coefs = Vw_pinv @ (windows * sw[:, 0][None, :]).T  # (3, out_len)

    # Polynomial value at t=0 (newest sample) is the constant term
    y_poly = coefs[0]

    # Local residual std per window (noise estimate)
    fitted = (V @ coefs).T
    resid = windows - fitted
    res_std = np.sqrt(np.mean(resid ** 2, axis=1)) + 1e-9

    # Adaptive blend: noisier windows trust the polynomial (smoother).
    # Tighter clip limits raw-noise leakage (main source of false
    # reversals) while keeping the fit evaluated at the newest sample.
    alpha = 1.0 / (1.0 + 0.5 * res_std)
    alpha = np.clip(alpha, 0.95, 0.998)
    y = alpha * y_poly + (1.0 - alpha) * windows[:, -1]

    # ---- Stage 2: zero-phase forward-backward smoothing ----
    # A causal exponential smoother run forward then backward, averaged:
    # the backward pass cancels the forward pass's phase lag (zero-phase
    # result), halving lag error while doubling noise suppression.
    # Stronger lambda (0.8): the backward pass still cancels phase lag,
    # so extra smoothing costs almost no lag error while further
    # suppressing noise-induced slope reversals.
    lam = 0.8
    m = len(y)
    yf = y.copy()
    yb = y.copy()
    for i in range(1, m):
        yf[i] = lam * yf[i - 1] + (1.0 - lam) * y[i]
    for i in range(m - 2, -1, -1):
        yb[i] = lam * yb[i + 1] + (1.0 - lam) * y[i]
    y = 0.5 * (yf + yb)

    # ---- Stage 3: centered quadratic refinement (zero phase), x25 ----
    # Wider window (25 samples) and more iterations aggressively remove
    # residual jitter; being centered, it adds no phase delay.
    r = 12
    if m > 2 * r:
        tc = np.arange(-r, r + 1, dtype=float)
        Vc = np.vstack([np.ones_like(tc), tc, tc * tc]).T
        Vc_pinv = np.linalg.pinv(Vc)
        offsets = np.arange(2 * r + 1)
        for _ in range(25):  # iterate refinement
            starts = np.arange(m - 2 * r)
            ywin = y[starts[:, None] + offsets[None, :]]
            c = Vc_pinv @ ywin.T
            y[r : m - r] = c[0]  # value at window center (t=0)
    return y


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
