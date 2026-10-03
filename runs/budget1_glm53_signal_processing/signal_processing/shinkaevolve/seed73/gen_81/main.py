# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.

Key mutation: Stage 3.5 — a 5-tap median snap on the robust-SG output
(x_smooth) with a 0.5x local-sigma step threshold, blended 50/50 like the
final snap. This is the first stage targeting noise_reduction directly.
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

    output_length = len(x) - window_size + 1
    y = np.zeros(output_length)

    for i in range(output_length):
        window = x[i : i + window_size]
        y[i] = np.mean(window)

    return y


def _adaptive_kalman_filter(x, q_base=0.01, r_factor=1.0, forgetting=0.98):
    """
    1D Kalman filter with innovation-adaptive process noise.

    When the normalized innovation squared (NIS) indicates the prediction
    error is consistent with measurement noise, process noise stays small
    (heavy smoothing). When NIS is large (genuine dynamics change), process
    noise grows so the filter tracks the change quickly, minimizing lag.
    """
    n = len(x)
    y = np.zeros(n)

    if n > 2:
        d = np.diff(x)
        med = np.median(d)
        mad = np.median(np.abs(d - med)) / 0.6745
        r = max((r_factor * mad * mad) / 2.0, 1e-8)
    else:
        r = max(r_factor * np.var(x), 1e-8)

    x_state = np.array([x[0], 0.0])
    P = np.eye(2) * r
    I = np.eye(2)
    H = np.array([[1.0, 0.0]])

    for k in range(n):
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        x_state = F @ x_state
        P = F @ P @ F.T

        z = x[k]
        innov = z - x_state[0]
        S = P[0, 0] + r
        nis = innov * innov / S

        q_scale = q_base * (1.0 + 10.0 * min(nis, 50.0) / 10.0)
        Q = q_scale * np.array([[0.25, 0.5], [0.5, 1.0]])
        P = P + Q

        S = P[0, 0] + r
        K = P[:, 0] / S
        x_state = x_state + K * innov
        P = (I - np.outer(K, H)) @ P

        y[k] = x_state[0]

    return y


def _bidirectional_kalman(x):
    """
    Forward adaptive-Q Kalman recursion fused 50/50 with a backward pass
    (the same filter applied to the time-reversed input). Cancels much of
    the interior phase error of the causal forward pass.
    """
    x = np.asarray(x, dtype=float)
    fwd = _adaptive_kalman_filter(x)
    bwd = _adaptive_kalman_filter(x[::-1])[::-1]
    return 0.5 * fwd + 0.5 * bwd


def _robust_savgol(x_kalman, sg_window, poly_order, n_iter=2, clip_sigma=2.5):
    """
    Huber-style iterative reweighted Savitzky-Golay smoothing.

    Runs SG once, computes residuals against the (already heavily smoothed)
    fit, estimates a robust noise scale via MAD, winsorizes residuals to
    +/- clip_sigma * sigma to build an adjusted signal, and re-runs SG on it.
    Outlier noise samples that would otherwise pull the local polynomial fit
    (creating residual noise and spurious slope reversals) are down-weighted.
    Fully vectorized; adds no phase delay.
    """
    from scipy.signal import savgol_filter

    y_fit = savgol_filter(x_kalman, sg_window, poly_order)

    for _ in range(n_iter):
        residuals = x_kalman - y_fit
        med_r = np.median(residuals)
        mad_r = np.median(np.abs(residuals - med_r)) / 0.6745
        sigma = max(mad_r, 1e-12)
        clipped = np.clip(residuals - med_r, -clip_sigma * sigma, clip_sigma * sigma)
        x_adj = y_fit + clipped
        y_new = savgol_filter(x_adj, sg_window, poly_order)
        # Guard against divergence: accept only if fit stays finite.
        if not np.all(np.isfinite(y_new)):
            break
        y_fit = y_new

    return y_fit


def _median_snap(x, sigma, blend=0.5, thr_factor=0.5):
    """
    5-tap running-median snap with a local-sigma step threshold.

    Samples whose local step magnitude is below thr_factor * sigma (i.e.
    indistinguishable from noise) are blended toward the 5-tap running
    median. Genuine trend steps exceed the threshold and are untouched.
    Fully vectorized via sliding_window_view.
    """
    n = len(x)
    if n < 5:
        return x
    pad = np.pad(x, 2, mode="edge")
    swv = np.lib.stride_tricks.sliding_window_view(pad, 5)
    med = np.median(swv, axis=1)

    steps = np.zeros_like(x)
    steps[1:] = np.abs(np.diff(x))
    steps[:-1] = np.maximum(steps[:-1], steps[1:])

    # Broadcast sigma per-sample if array, else scalar
    thr = thr_factor * (sigma if np.isscalar(sigma) else sigma[:n])
    small = steps < thr
    return np.where(small, (1.0 - blend) * x + blend * med, x)


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version with trend preservation:

    Stage 1: bidirectional adaptive Kalman (phase-error cancellation).
    Stage 2: robust SG (Huber-style iterative reweighting).
    Stage 3.5: 5-tap median snap on x_smooth with 0.5x local-sigma
               threshold (targets noise_reduction directly).
    Stage 3: recency-weighted convolution (output convention + lag comp).
    Stage 4: level-domain median snap with locally adaptive threshold.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)

    # Stage 1: bidirectional adaptive Kalman filtering
    if len(x) >= 4:
        x_kalman = _bidirectional_kalman(x)
    else:
        x_kalman = _adaptive_kalman_filter(x)

    # Stage 2: robustly reweighted Savitzky-Golay polynomial smoothing
    # 2W+1 is the proven crossover sizing: 2W+3 buys no extra suppression
    # here but adds group delay that surfaces as extra lag-induced reversals.
    sg_window = 2 * window_size + 1
    if sg_window % 2 == 0:
        sg_window += 1
    sg_window = max(sg_window, 5)
    if sg_window > len(x_kalman):
        sg_window = len(x_kalman) if len(x_kalman) % 2 == 1 else len(x_kalman) - 1
        sg_window = max(sg_window, 5)
    poly_order = 2

    x_smooth = _robust_savgol(x_kalman, sg_window, poly_order, n_iter=2, clip_sigma=2.5)

    # Robust local noise scale from rolling MAD of diffs (shared by stages)
    d = np.diff(x_smooth)
    block = 51  # +/- 25 samples

    if len(d) >= block:
        swv = np.lib.stride_tricks.sliding_window_view(d, block)
        local_mad = np.median(np.abs(swv - np.median(swv, axis=1, keepdims=True)),
                              axis=1) / 0.6745
        pad_len = (block - 1) // 2
        local_mad = np.concatenate([
            local_mad[:1] * np.ones(pad_len), local_mad, local_mad[-1:] * np.ones(pad_len)
        ])
    else:
        med_d = np.median(d)
        local_mad = np.full(len(d), np.median(np.abs(d - med_d)) / 0.6745)
    sigma_local = np.maximum(local_mad / np.sqrt(2.0), 1e-12)

    # Stage 3.5: median snap on x_smooth (targets noise_reduction).
    # sigma for sample i corresponds to diffs; extend to n samples.
    sigma_x = np.concatenate([sigma_local, sigma_local[-1:]])[:len(x_smooth)]
    x_smooth = _median_snap(x_smooth, sigma_x, blend=0.6, thr_factor=0.5)

    # Stage 3: recency-weighted convolution (lag-compensation mechanism)
    w = np.exp(np.linspace(-1.5, 0.0, window_size))
    w = w / np.sum(w)
    y = np.convolve(x_smooth, w[::-1], mode="valid")  # length = n - W + 1

    # Stage 4: inter-window slope-consistency pass (locally adaptive threshold)
    if len(y) >= 5:
        y = _median_snap(y, np.concatenate([sigma_local[:1], sigma_local])[:len(y)],
                         blend=0.7, thr_factor=0.45)

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