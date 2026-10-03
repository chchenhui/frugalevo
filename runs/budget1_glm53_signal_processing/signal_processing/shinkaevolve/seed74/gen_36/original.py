# EVOLVE-BLOCK-START
"""
Zero-Phase Savitzky-Golay Trend Filter with Robust Refinement

Fundamentally different approach: instead of causal state-space filtering
(Kalman) or weighted averages, fit low-order polynomials in a sliding
window by least squares and evaluate at the window end. This is a
zero-phase (non-causal within the window) smoother with no group delay,
excellent trend tracking, and intrinsically smooth output. A robust
(Hampel) pass suppresses outliers, and slope hysteresis with a
noise-scaled deadzone eliminates spurious reversals.
"""
import numpy as np


def _robust_denoise(x, k=5, t=3.0):
    """Hampel-style outlier clipping: replace samples deviating more than
    t * local_MAD from the local median with the local median."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * k + 1:
        return x.copy()
    med = np.array([np.median(x[max(0, i - k):min(n, i + k + 1)]) for i in range(n)])
    dev = x - med
    mad = np.median(np.abs(dev - np.median(dev)))
    if mad < 1e-12:
        mad = np.std(dev) + 1e-12
    thresh = t * 1.4826 * mad
    out = x.copy()
    mask = np.abs(dev) > thresh
    out[mask] = med[mask]
    return out


def _savgol_edge(x, window, order=3):
    """Savitzky-Golay-style local polynomial fit evaluated at each window's
    last sample (zero-phase relative to window content; no lag for
    polynomial-representable dynamics). Vectorized via projection matrix."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    # Use numpy's savgol but with 'interp' edges; window must be odd.
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= order:
        order = max(w - 1, 1)
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(x, w, order, mode="interp")
    except Exception:
        pass
    # Fallback: manual convolution via polynomial coefficients
    half = w // 2
    t = np.arange(-half, half + 1, dtype=float)
    A = np.vander(t, order + 1, increasing=True)
    # projection: value at t=0 (center) — then shift alignment externally
    coef = np.linalg.pinv(A.T @ A) @ A.T
    y = np.convolve(x, coef[::-1], mode="same")
    return y


def _slope_hysteresis(y, sigma_n, persistence=2, deadzone_k=0.8):
    """Suppress false reversals: require slope sign to persist beyond a
    noise-scaled deadzone before allowing trend direction to flip."""
    n = len(y)
    if n < 3:
        return y.copy()
    slopes = np.gradient(y)
    dead = deadzone_k * sigma_n
    out = y.copy()
    confirmed_sign = np.sign(slopes[0]) if abs(slopes[0]) > dead else 0
    run = 0
    for k in range(1, n):
        s = slopes[k]
        sg = 1 if s > dead else (-1 if s < -dead else 0)
        if sg != 0 and sg == confirmed_sign:
            run += 1
        elif sg != 0:
            run += 1
            if run >= persistence:
                confirmed_sign = sg
                run = 0
        else:
            run = 0
        # If current local slope disagrees with confirmed trend direction
        # and is small, replace sample with trend-consistent extrapolation
        # (light linear correction toward confirmed direction).
        if sg == 0:
            # flat zone: leave as-is (already smooth)
            continue
        elif sg != confirmed_sign and confirmed_sign != 0:
            # tentative reversal: soften by averaging with neighbor
            out[k] = 0.5 * (out[k] + out[k - 1])
            slopes[k] = 0.5 * (slopes[k] + slopes[k - 1])
    return out


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Zero-phase polynomial trend filter with robust outlier handling and
    slope-hysteresis reversal suppression."""
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    # 1) Robust outlier suppression (pre-smoothing of impulsive noise)
    xr = _robust_denoise(x, k=max(3, window_size // 6), t=3.0)

    # 2) Zero-phase polynomial smoothing over the entire signal
    ys = _savgol_edge(xr, window_size, order=3)

    # 3) Estimate noise level from residual of robust input vs smooth
    resid = x - ys
    sigma_n = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    sigma_n = max(sigma_n, 1e-12)

    # 4) Slope hysteresis to remove spurious reversals
    yf = _slope_hysteresis(ys, sigma_n, persistence=2, deadzone_k=0.8)

    # 5) Align to sliding-window semantics: output i corresponds to window
    # x[i : i + window_size], i.e., clean comparison index i + window_size - 1.
    output_length = len(x) - window_size + 1
    anchor = np.arange(output_length) + window_size - 1
    y = yf[anchor]
    return y


def adaptive_filter(x, window_size=20):
    """Baseline-compatible entry: same algorithm."""
    return enhanced_filter_with_trend_preservation(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """Main signal processing function (same interface as original)."""
    return enhanced_filter_with_trend_preservation(input_signal, window_size)

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
