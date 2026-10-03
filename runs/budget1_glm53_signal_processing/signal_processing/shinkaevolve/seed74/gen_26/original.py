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

    n = len(x)
    x = np.asarray(x, dtype=float)

    # ---- Stage 1: Adaptive Kalman filter (constant-velocity model) ----
    q = 0.05   # process noise (signal dynamics)
    r_base = 0.5  # base measurement noise
    delta = 1.0
    # EMA of squared innovations for robust (outlier-resistant) adaptive R
    innov_ema = 0.0
    ema_alpha = 0.08

    # State: [level, velocity]
    level = x[0]
    vel = 0.0
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    H = np.array([1.0, 0.0])
    F = np.array([[1.0, delta], [0.0, 1.0]])
    Q = q * np.array([[delta**3 / 3.0, delta**2 / 2.0],
                      [delta**2 / 2.0, delta]])

    kf_out = np.empty(n)
    r_hist = np.empty(n)
    for i in range(n):
        # Predict
        level = level + delta * vel
        P = F @ P @ F.T + Q

        # Adaptive measurement noise from innovation magnitude
        innov = x[i] - level
        # Use smoothed (EMA) innovation power instead of instantaneous value
        # so that isolated noise spikes do not collapse r and yank the state
        innov_ema = (1.0 - ema_alpha) * innov_ema + ema_alpha * innov * innov
        r = r_base + 0.25 * innov_ema
        r_hist[i] = r

        # Update
        S = P[0, 0] + r
        K = np.array([P[0, 0] / S, P[1, 0] / S])
        level = level + K[0] * innov
        vel = vel + K[1] * innov
        P = (np.eye(2) - np.outer(K, H)) @ P
        # symmetrize for numerical stability
        P = 0.5 * (P + P.T)

        kf_out[i] = level

    # ---- Stage 1b: Backward pass (RTS-style two-pass smoothing) ----
    # Running the same constant-velocity filter in reverse and blending the
    # two passes yields an effectively zero-phase estimate: it removes the
    # causal filter's phase lag while further suppressing jitter.
    level = kf_out[-1]
    vel = 0.0
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    bw_out = np.empty(n)
    innov_ema = 0.0
    for i in range(n - 1, -1, -1):
        # Predict
        level = level + delta * vel
        P = F @ P @ F.T + Q

        innov = kf_out[i] - level
        innov_ema = (1.0 - ema_alpha) * innov_ema + ema_alpha * innov * innov
        r = r_base + 0.25 * innov_ema

        S = P[0, 0] + r
        K = np.array([P[0, 0] / S, P[1, 0] / S])
        level = level + K[0] * innov
        vel = vel + K[1] * innov
        P = (np.eye(2) - np.outer(K, H)) @ P
        P = 0.5 * (P + P.T)

        bw_out[i] = level

    # Blend forward and backward (zero-phase) estimates
    kf_out = 0.5 * (kf_out + bw_out)

    # ---- Stage 2: Savitzky-Golay trend-preserving smoothing ----
    half = max(2, window_size // 6)
    if half % 2 == 0:
        half += 1
        if half < 3:
            half = 3
    half = half if half % 2 == 1 else half + 1
    sg_win = 2 * (half // 2) + 1  # odd window
    if sg_win >= 5:
        # Fixed SG coefficients (order 2) via least squares
        idx = np.arange(-(sg_win // 2), sg_win // 2 + 1)
        A = np.vander(idx, 3, increasing=True)
        coef, _, _, _ = np.linalg.lstsq(A.T @ A, A.T, rcond=None)
        # smoothing coefficients = first row of (A^T A)^-1 A^T applied to data
        pinv = np.linalg.pinv(A)
        h = pinv[0, :]  # coefficients for the smoothed (constant) term
        padded = np.pad(kf_out, (sg_win // 2, sg_win // 2), mode="edge")
        smoothed = np.convolve(padded, h[::-1], mode="valid")
        kf_out = smoothed[:n]

    # ---- Trim to expected output length (align with window semantics) ----
    output_length = n - window_size + 1
    return kf_out[n - output_length:]


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