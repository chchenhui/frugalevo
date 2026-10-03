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
    Enhanced version with trend preservation using weighted moving average.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    # Exponential weights emphasizing recent samples (adaptive to non-stationarity)
    # Steeper recency weighting compensates the group delay of the downstream
    # SG stage (wider kernel), keeping lag error low.
    weights = np.exp(np.linspace(-1.5, 0, window_size))
    weights = weights / np.sum(weights)
    t = np.arange(window_size) - (window_size - 1)

    # Precompute weighted-normal-equation components that don't change
    Sw = np.sum(weights)
    St = np.sum(weights * t)
    Stt = np.sum(weights * t * t)
    denom = Sw * Stt - St * St

    xf = np.asarray(x, dtype=float)

    # --- Pre-convolution snap: 5-tap median on raw input, noise-scaled ---
    # Remove impulse-like, noise-sized steps from the raw signal before
    # window averaging so they cannot leak through the weighted regression.
    n = len(xf)
    if n >= 5:
        dx = np.diff(xf)
        sig_x = 1.4826 * np.median(np.abs(dx - np.median(dx))) + 1e-12
        # 5-tap running median of the input
        pad_l = np.concatenate((xf[:2][::-1], xf, xf[-2:][::-1]))
        stacked = np.stack([pad_l[0:n], pad_l[1:n+1], pad_l[2:n+2],
                           pad_l[3:n+3], pad_l[4:n+4]])
        med5 = np.median(stacked, axis=0)
        # Snap only samples whose deviation from the local median is
        # noise-sized (no genuine trend step is large enough to be caught).
        dev = np.abs(xf - med5)
        snap_mask = dev < 2.0 * sig_x
        xf = xf.copy()
        xf[snap_mask] = 0.5 * xf[snap_mask] + 0.5 * med5[snap_mask]

    # --- Vectorized pass 1: running weighted sums via convolution ---
    # Reversed kernel turns convolution into sliding correlation:
    # Sx[i] = sum(weights * x[i:i+W]), evaluated at every window at once.
    Sx = np.convolve(xf, weights[::-1], mode="valid")[:output_length]
    wt = weights * t
    Stx = np.convolve(xf, wt[::-1], mode="valid")[:output_length]

    if denom > 1e-12:
        y = (Stt * Sx - St * Stx) / denom  # fitted value at t = 0 per window
    else:
        y = Sx / Sw

    # --- Savitzky-Golay stage: order-2 polynomial smoothing, end-point evaluated ---
    # A wider (2W-1, odd) SG window suppresses sub-noise churn without the
    # over-smoothing of a moving average: the local quadratic fit preserves
    # genuine curvature/trends. Coefficients are precomputed once.
    M = 2 * window_size - 1  # kept odd
    if len(y) >= M:
        tm = np.arange(M, dtype=float) - (M - 1)
        V = np.stack([np.ones(M), tm, tm * tm], axis=1)
        # Least-squares fit evaluated at t = 0 (last sample) -> zero phase delay
        c = np.linalg.pinv(V)[0, :]  # row giving fitted value at t = 0
        # Edge-pad pass-1 output so convolution preserves output length
        half = (M - 1) // 2
        pad_y = np.concatenate((np.full(half, y[0]), y, np.full(half, y[-1])))
        ys = np.convolve(pad_y, c[::-1], mode="valid")
        # Blend SG smoothness with regression responsiveness
        y = 0.6 * ys + 0.4 * y

    # --- Pass 2 (targeted): robust reweighting only where pass-1 fits badly ---
    # Weighted RMS residual per window, computed vectorized:
    # sum w*(x - fit)^2 = sum(w*x^2) - 2*fit*sum(w*x) + fit^2*sum(w)
    win_sq = np.convolve(xf * xf, weights[::-1], mode="valid")[:output_length]
    resid_var = np.maximum(win_sq - 2.0 * y * Sx + y * y * Sw, 0.0)
    resid_scale = np.sqrt(resid_var / Sw)

    global_scale = np.median(resid_scale)
    bad = np.where(resid_scale > 2.0 * global_scale + 1e-12)[0]

    for i in bad:
        window = xf[i : i + window_size]
        base = y[i]
        if denom > 1e-12:
            slope_i = (Sw * Stx[i] - St * Sx[i]) / denom
        else:
            slope_i = 0.0
        fit = slope_i * t + base
        resid = window - fit
        scale = np.std(resid)
        if scale > 1e-12:
            rw = weights.copy()
            big = np.abs(resid) > 2.0 * scale
            rw[big] *= scale / (np.abs(resid[big]) + 1e-12)
            Sw2 = np.sum(rw)
            St2 = np.sum(rw * t)
            Stt2 = np.sum(rw * t * t)
            den2 = Sw2 * Stt2 - St2 * St2
            if den2 > 1e-12:
                Sx2 = np.sum(rw * window)
                Stx2 = np.sum(rw * t * window)
                y[i] = (Stt2 * Sx2 - St2 * Stx2) / den2  # value at t = 0
            else:
                y[i] = Sx2 / Sw2

    # --- Slope-consistency snap: kill sub-noise directional churn ---
    # Estimate noise sigma from first differences of the smoothed output
    # (trend contributes slowly-varying diffs; noise dominates the rest).
    if output_length > 2:
        d = np.diff(y)
        sigma = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
        thresh = 0.5 * sigma
        # 3-tap running median of the output
        pad = np.concatenate(([y[0]], y, [y[-1]]))
        med3 = np.median(np.stack([pad[:-2], pad[1:-1], pad[2:]]), axis=0)
        # Output steps below the noise threshold are indistinguishable from
        # noise -> snap those samples toward their local median value to
        # enforce inter-window slope consistency without touching real trends.
        small = np.abs(d) < thresh
        idx = np.where(small)[0] + 1  # diff d[k] affects y[k+1]
        y[idx] = 0.5 * y[idx] + 0.5 * med3[idx]

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