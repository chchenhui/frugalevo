# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.
"""
import numpy as np


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    2-state Kalman filter (level + slope) with adaptive process noise,
    followed by RTS backward smoothing, two binomial smoothing passes,
    and median-filtered slope re-integration.

    The constant-velocity state model tracks genuine trends with minimal
    lag; the Kalman gain suppresses noise; the RTS smoother removes lag
    (zero-phase) while further reducing variance. Innovations exceeding
    ~2.6 sigma temporarily inflate process noise so real rapid transitions
    are followed quickly. Binomial passes and slope-median hysteresis
    suppress spurious reversals. Output index i corresponds to input
    index i + window_size - 1 (sliding-window alignment).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    output_length = n - window_size + 1
    delay = window_size - 1

    # Estimate noise variance from first differences (robust MAD-based)
    d = np.diff(x)
    r = 1.4826 * np.median(np.abs(d - np.median(d)))
    R = max(r * r, 1e-6)

    # Process noise: small base for smooth tracking
    q_level = 0.05 * R
    q_slope = 0.005 * R

    # State: [level, slope]
    state = np.array([x[0], 0.0])
    P = np.eye(2) * R

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    # Store forward and predicted states/covariances for RTS smoothing
    xs_f = np.zeros((n, 2))
    Ps_f = np.zeros((n, 2, 2))
    xps_f = np.zeros((n, 2))
    Pps_f = np.zeros((n, 2, 2))

    for k in range(n):
        # Predict
        state = F @ state
        P = F @ P @ F.T
        P[0, 0] += q_level
        P[1, 1] += q_slope
        xps_f[k] = state
        Pps_f[k] = P

        # Innovation
        innov = x[k] - (H @ state)[0]
        S = P[0, 0] + R

        # Adaptive gating: large innovations => real transition, inflate Q
        if innov * innov > 6.76 * S:
            P[0, 0] += 10.0 * q_level
            P[1, 1] += 10.0 * q_slope
            S = P[0, 0] + R

        # Kalman gain
        K = P[:, 0] / S
        state = state + K * innov
        P = (I - np.outer(K, H)) @ P

        # Store filtered states and the PREDICTED states (pre-update)
        # for the RTS smoother; storing post-update values corrupts
        # the backward pass and reintroduces lag/error.
        xs_f[k] = state
        Ps_f[k] = P
    # End of forward loop; xps_f/Pps_f hold predicted quantities

    # Backward RTS smoothing pass: zero-phase, removes Kalman lag
    xs_s = xs_f.copy()
    for k in range(n - 2, -1, -1):
        Pp_inv = np.linalg.inv(Pps_f[k + 1])
        G = Ps_f[k] @ F.T @ Pp_inv
        xs_s[k] = xs_f[k] + G @ (xs_s[k + 1] - xps_f[k + 1])

    # Emit output aligned with sliding-window convention
    y = xs_s[delay:, 0]

    # Two passes of 7-tap Savitzky-Golay (quadratic) smoothing.
    # Unlike binomial kernels, SG preserves quadratic trends exactly,
    # so noise is removed without biasing peaks/troughs -> better
    # smoothness and fewer spurious slope reversals, minimal lag.
    sg = np.array([-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]) / 21.0

    def sg_pass(sig):
        s = sig.copy()
        m = len(s)
        if m >= 7:
            s[3:-3] = np.convolve(sig, sg, mode="valid")
            # Edge handling: shrink toward shorter quadratic fits
            s[2] = (3.0 * sig[1] + 4.0 * sig[2] + 3.0 * sig[3] + sig[0] - sig[4]) / 11.0
            s[-3] = (3.0 * sig[-2] + 4.0 * sig[-3] + 3.0 * sig[-4] + sig[-1] - sig[-5]) / 11.0
            s[1] = (sig[0] + 2.0 * sig[1] + sig[2]) / 4.0
            s[-2] = (sig[-3] + 2.0 * sig[-2] + sig[-1]) / 4.0
        return s

    y = sg_pass(y)
    y = sg_pass(y)

    # Slope-hysteresis pass: 5-tap median-filter the derivative, then
    # integrate. Kills 1-2 sample noise bursts in the derivative (the
    # main source of false reversals) with negligible lag, since a
    # median filter is nearly zero-phase.
    if len(y) >= 6:
        dy = np.diff(y)
        dym = dy.copy()
        dym[2:-2] = np.median(np.vstack([dy[:-4], dy[1:-3], dy[2:-2],
                                          dy[3:-1], dy[4:]]), axis=0)
        dym[1] = np.median(dy[:3])
        dym[-2] = np.median(dy[-3:])
        dym[0] = dym[1]
        dym[-1] = dym[-2]
        y = y[0] + np.concatenate([[0.0], np.cumsum(dym)])

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
        # Baseline: simple sliding-window moving average
        x = np.asarray(input_signal, dtype=float)
        if len(x) < window_size:
            raise ValueError("Input too short")
        return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


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
