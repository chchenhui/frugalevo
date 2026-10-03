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
    """Causal robust endpoint regression with trend-accelerated and persistence-confirmed alpha-beta tracking."""
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    n = len(x) - window_size + 1
    y = np.zeros(n)
    span = max(window_size - 1, 1)
    t = np.linspace(-1.0, 0.0, window_size)
    base = np.exp(np.linspace(-2.2, 0.0, window_size))
    base /= base.sum()

    def fit(values, weights):
        """Estimate weighted local-linear level and slope at the newest endpoint."""
        tw = np.sum(weights * t)
        vw = np.sum(weights * values)
        dt = t - tw
        slope = np.sum(weights * dt * (values - vw)) / (np.sum(weights * dt * dt) + 1e-12)
        return vw - slope * tw, slope

    level = 0.0
    velocity = 0.0
    turn_evidence = 0
    for i in range(n):
        window = np.asarray(x[i:i + window_size], dtype=float)
        endpoint, slope = fit(window, base)
        residual = window - (endpoint + slope * t)
        scale = 1.4826 * np.median(np.abs(residual)) + 1e-8

        weights = base / (1.0 + (residual / (2.8 * scale)) ** 2)
        endpoint, slope = fit(window, weights)
        mean_estimate = np.sum(weights * window) / np.sum(weights)

        strength = abs(slope) / (scale * span)
        blend = np.clip(0.44 + 0.10 * strength, 0.44, 0.66)
        candidate = blend * endpoint + (1.0 - blend) * mean_estimate
        measured_velocity = slope / span

        if i == 0:
            level = candidate
            velocity = measured_velocity
            y[i] = level
            continue

        prediction = level + velocity
        innovation = candidate - prediction

        # A small noise-scaled dead band avoids converting endpoint jitter into slope flips.
        if abs(innovation) < 0.055 * scale:
            innovation = 0.0
        else:
            # Limit isolated endpoint excursions while retaining legitimate large moves.
            innovation = np.clip(innovation, -3.0 * scale, 3.0 * scale)

        alpha = np.clip(0.36 + 0.24 * strength, 0.36, 0.64)
        beta = np.clip(0.045 + 0.060 * strength, 0.045, 0.105)

        # Trend-consistent innovations can be followed slightly faster without
        # introducing a directional reversal. Opposing evidence remains gated.
        opposing = innovation * velocity < 0.0
        if not opposing:
            follow_gain = 1.06 + 0.08 * np.clip(strength, 0.0, 1.0)
            alpha = min(alpha * follow_gain, 0.70)

        # A turn is credible only when the independently fitted local slope points
        # with the innovation, and that evidence persists across adjacent windows.
        slope_agrees = innovation * measured_velocity > 0.0
        material_turn = abs(innovation) > 0.12 * scale
        if opposing and slope_agrees and material_turn:
            turn_evidence = min(turn_evidence + 1, 3)
        else:
            turn_evidence = max(turn_evidence - 1, 0)

        if opposing:
            coherence = abs(measured_velocity) / (abs(velocity) + 0.08 * scale + 1e-12)
            reversal_confidence = np.clip(0.30 * strength + 0.70 * coherence, 0.0, 1.0)
            # First-sample opposition remains heavily damped. Once two robust
            # window estimates agree, restore most gain to reduce true-turn lag.
            confirmation = 0.45 if turn_evidence < 2 else 0.80 + 0.20 * reversal_confidence
            alpha *= confirmation
            beta *= confirmation

        level = prediction + alpha * innovation
        velocity = (1.0 - beta) * velocity + beta * (measured_velocity + innovation)
        y[i] = level

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
