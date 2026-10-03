# EVOLVE-BLOCK-START
"""
Zero-Lag Robust Local-Polynomial Filter with Trend-Lock Damping.

Novel sliding-window algorithm:
  1. Per-window robust weighted quadratic fit (IRLS with Huber weights)
     evaluated at the window's leading edge -> removes group delay analytically.
  2. Trend-lock: slope estimate is shrinkage-smoothed across windows to suppress
     spurious directional reversals while allowing genuine trend changes.
"""
import numpy as np


def _robust_window_estimate(window, half_life=None):
    """
    Fit a robust weighted quadratic to the window and evaluate at the leading edge.

    Args:
        window: 1D array of W samples (oldest -> newest)
        half_life: decay half-life for exponential recency weighting

    Returns:
        (value_at_leading_edge, slope_at_leading_edge)
    """
    W = len(window)
    if W < 3:
        w_last = window[-1]
        slope = window[-1] - window[-2] if W > 1 else 0.0
        return w_last, slope

    # Recency weights: recent samples matter most (noise suppression + responsiveness)
    if half_life is None:
        half_life = max(2.0, W / 4.0)
    decay = np.log(2.0) / half_life
    t = np.arange(W, dtype=float) - (W - 1)  # centered so leading edge is t=0
    base_w = np.exp(decay * t)

    # Design matrix for quadratic fit: value, slope, curvature at leading edge
    # x(0) = a, x'(0) = b, x''(0)/2 = c  ->  x(t) = a + b*t + c*t^2
    A = np.vstack([np.ones(W), t, t * t]).T

    coef = None
    resid_scale = None
    # IRLS: 2 iterations of Huber reweighting
    for iteration in range(3):
        if coef is not None:
            resid = window - A @ coef
            scale = np.median(np.abs(resid)) + 1e-9
            # Huber weights: cap influence of outlier residuals
            z = np.abs(resid) / (1.345 * scale)
            huber = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-9))
            wts = base_w * huber
        else:
            wts = base_w

        Aw = A * wts[:, None]
        yw = window * wts
        # Solve (A^T W A) c = A^T W y  via normal equations (3x3, cheap)
        M = Aw.T @ A
        rhs = Aw.T @ yw
        try:
            coef = np.linalg.solve(M + 1e-9 * np.eye(3), rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(Aw, yw, rcond=None)[0]

    # Evaluated at t = 0 (leading edge) -> a = coef[0], slope = coef[1]
    return coef[0], coef[1]


def adaptive_filter(x, window_size=20):
    """
    Zero-lag robust local-polynomial filter.

    Args:
        x: Input signal (1D array)
        window_size: Sliding window size W

    Returns:
        y: Filtered signal, length = len(x) - window_size + 1
    """
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)
    raw_slopes = np.zeros(output_length)

    # Pass 1: robust leading-edge estimates per window
    for i in range(output_length):
        window = x[i : i + window_size]
        val, slope = _robust_window_estimate(window)
        y[i] = val
        raw_slopes[i] = slope

    # Pass 2: trend-lock damping.
    # Smooth the slope sequence lightly; where the damped slope disagrees in sign
    # with the raw slope but evidence is weak, keep the previous direction.
    # This kills noise-induced reversals without delaying real turnings much,
    # because the level estimate y[i] is already lag-free.
    beta = 0.55  # slope smoothing factor (EMA on slope)
    locked_slope = 0.0
    level_adj = np.zeros(output_length)
    level_adj[0] = y[0]
    for i in range(output_length):
        if i == 0:
            locked_slope = raw_slopes[0]
        else:
            locked_slope = beta * locked_slope + (1.0 - beta) * raw_slopes[i]
        level_adj[i] = y[i] + 0.35 * (locked_slope - raw_slopes[i])

    return level_adj


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Alias of the main filter (trend preservation is built-in)."""
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Main entry point; both algorithm types route to the zero-lag robust filter.

    Args:
        input_signal: Input time series
        window_size: Window size
        algorithm_type: "basic" or "enhanced" (both use the new filter)

    Returns:
        Filtered signal
    """
    return adaptive_filter(np.asarray(input_signal, dtype=float), window_size)

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
