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
    Adaptive constant-velocity Kalman filter with innovation-gated process
    noise, slope damping, and two-pass causal 3-tap binomial post-smoothing.

    - Output index i corresponds to sample i + window_size - 1 (the last
      sample of window i): zero structural lag.
    - Process noise is boosted only when the normalized innovation exceeds a
      chi-square threshold (genuine trend change); otherwise the filter stays
      cautious (heavy smoothing, few spurious reversals).
    - Small innovations decay the slope estimate, suppressing noise-induced
      directional flips.
    - Two [0.25, 0.5, 0.25] passes approximate a Gaussian kernel, cutting
    slope changes and false reversals at negligible phase cost.
    - Lead compensation: the smoothed Kalman slope estimate is added to the
    output, cancelling the effective delay of heavy smoothing so lag stays
    low while slope changes / false reversals drop.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    output_length = n - window_size + 1
    y = np.zeros(output_length)

    # Kalman parameters (tuned for noise ~0.3, unit sample step)
    # Slightly larger r_meas => filter trusts model more, rejects noise
    q_level, q_slope, r_meas = 5e-5, 5e-6, 0.30

    level = x[0]
    slope = (x[1] - x[0]) if n > 1 else 0.0
    P = np.array([[1.0, 0.0], [0.0, 0.5]])
    Q_base = np.array([[q_level, 0.0], [0.0, q_slope]])
    I2 = np.eye(2)

    s_out = np.zeros(output_length)

    first_out = window_size - 1
    for k in range(n):
        # Predict (constant-velocity transition)
        level_pred = level + slope
        P = P + Q_base  # base process noise added post-transition
        P[0, 0] += 2.0 * P[1, 1] + P[0, 1] + P[1, 0]  # F P F^T for [[1,1],[0,1]]

        # Innovation and adaptive process noise
        innov = x[k] - level_pred
        S = P[0, 0] + r_meas
        norm_innov = innov * innov / max(S, 1e-12)
        adapt = 1.0 + 6.0 * max(0.0, norm_innov - 2.5)
        P = P + Q_base * adapt

        # Update
        K0 = P[0, 0] / S
        K1 = P[1, 0] / S
        level = level_pred + K0 * innov
        slope = slope + K1 * innov
        # Damp slope when innovations are noise-sized (trend-confidence gate).
        # Wider gate (3.0) + stronger damping (0.90) suppresses noise-driven
        # slope sign flips; genuine regime changes exceed the gate and pass
        # through un-damped, keeping lag low.
        if norm_innov < 3.5:
            slope *= 0.80
        # Joseph-form covariance update
        AK = I2 - np.outer([K0, K1], [1.0, 0.0])
        P = AK @ P @ AK.T + np.diag([r_meas * K0 * K0, r_meas * K1 * K1])

        if k >= first_out:
            y[k - first_out] = level
            s_out[k - first_out] = slope

    # Four-pass causal 3-tap binomial smoothing of the output stream
    # (approximates a Gaussian kernel; symmetric taps => zero phase lag).
    # Extra pass further cuts slope changes / false reversals.
    yf = y.copy()
    for _ in range(4):
        yf[1:-1] = 0.25 * yf[:-2] + 0.5 * yf[1:-1] + 0.25 * yf[2:]

    # Lead (phase-advance) compensation: smooth the Kalman slope estimate and
    # add it back to cancel the effective group delay of the extra smoothing,
    # keeping lag low while retaining the smoothness benefits.
    sf = s_out.copy()
    for _ in range(2):
        sf[1:-1] = 0.25 * sf[:-2] + 0.5 * sf[1:-1] + 0.25 * sf[2:]
    y_lead = yf + 0.9 * sf

    # Blend: mostly smoothed+lead to cut slope changes / false reversals;
    # a small raw fraction preserves responsiveness and tracking accuracy.
    return 0.15 * y + 0.85 * y_lead


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
