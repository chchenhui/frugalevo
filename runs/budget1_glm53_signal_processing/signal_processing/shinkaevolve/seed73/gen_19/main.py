# EVOLVE-BLOCK-START
"""
Real-Time Adaptive Signal Processing Algorithm for Non-Stationary Time Series

This algorithm implements a sliding window approach to filter volatile, non-stationary
time series data while minimizing noise and preserving signal dynamics.

Hybrid approach: robust exponentially-weighted local quadratic regression
(zero phase lag, high tracking accuracy) combined with slope-gated reversal
suppression (median-gated slopes + level-anchored reintegration) to minimize
spurious slope changes and false reversals.
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

    # Vectorized simple moving average (baseline)
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Enhanced version: robust, exponentially-weighted local quadratic regression
    evaluated at the most recent sample of each window (zero phase lag),
    followed by slope-gated reversal suppression (median-gated slopes with
    level-anchored reintegration) and a light 3-tap smoothing pass.

    Args:
        x: Input signal (1D array of real-valued samples)
        window_size: Size of the sliding window

    Returns:
        y: Filtered output signal
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    n = len(x)
    output_length = n - window_size + 1
    y = np.zeros(output_length)
    b_fit = np.zeros(output_length)  # fitted slope at window end (t = 0)

    # Exponential weights emphasizing recent samples (adaptive to non-stationarity)
    weights = np.exp(np.linspace(-2, 0, window_size))
    weights = weights / np.sum(weights)

    # Time axis anchored so the LAST sample in the window is t = 0:
    # evaluating the fitted quadratic at t=0 gives zero trend-lag output.
    t = np.arange(window_size, dtype=float) - (window_size - 1)

    # Quadratic basis columns: t^2, t, 1
    t2 = t * t
    ones = np.ones(window_size)

    # Precompute weighted moment sums that do not depend on the data window
    S1 = np.dot(weights, ones)
    St = np.dot(weights, t)
    St2 = np.dot(weights, t2)
    St3 = np.dot(weights, t2 * t)
    St4 = np.dot(weights, t2 * t2)
    A = np.array([[St4, St3, St2],
                  [St3, St2, St],
                  [St2, St, S1]])
    w_t = weights * t
    w_t2 = weights * t2

    def _fit(rw, win):
        """Solve weighted normal equations for [a, b, c] of a*t^2 + b*t + c."""
        Ar = A
        if rw is not weights:
            rSt = np.dot(rw, t)
            rSt2 = np.dot(rw, t2)
            rSt3 = np.dot(rw, t2 * t)
            rSt4 = np.dot(rw, t2 * t2)
            Ar = np.array([[rSt4, rSt3, rSt2],
                           [rSt3, rSt2, rSt],
                           [rSt2, rSt, np.dot(rw, ones)]])
        rhs = np.array([np.dot(rw * t2, win),
                        np.dot(rw * t, win),
                        np.dot(rw, win)])
        try:
            return np.linalg.solve(Ar, rhs)
        except np.linalg.LinAlgError:
            s1 = np.dot(rw, ones)
            return np.array([0.0, 0.0, np.dot(rw, win) / s1 if s1 > 1e-12 else np.mean(win)])

    for i in range(output_length):
        win = x[i : i + window_size]

        # --- Pass 1: weighted local quadratic fit, evaluated at window end ---
        sol = _fit(weights, win)
        a, b, c0 = sol
        fit = a * t2 + b * t + c0

        # --- Pass 2: Huber-style robust reweighting to suppress outliers ---
        resid = win - fit
        scale = np.std(resid)
        if scale > 1e-12:
            rw = weights.copy()
            big = np.abs(resid) > 2.0 * scale
            rw[big] *= scale / (np.abs(resid[big]) + 1e-12)
            sol2 = _fit(rw, win)
            y[i] = sol2[2]
            b_fit[i] = sol2[1]
        else:
            y[i] = c0
            b_fit[i] = b

    if output_length < 2:
        return y

    # ---- Stage 3: slope-gated reversal suppression (Kalman-inspired) ----
    # Suppress sign flips of the estimated slope unless the flip is large
    # relative to local slope noise (kills noise-induced false reversals).
    slope_noise = np.std(np.diff(y)) + 1e-12
    gate = 0.5 * slope_noise
    est_slope = np.zeros(output_length)
    est_slope[1:] = np.diff(y)
    # Median filter on slope (window 3) with gating
    if output_length >= 3:
        for i in range(1, output_length - 1):
            med = np.median(est_slope[i - 1 : i + 2])
            if abs(est_slope[i] - med) < gate:
                est_slope[i] = med
    # Reintegrate gated slopes, anchored to levels via convex blend.
    # Pure reintegration accumulates slope biases as drift; blending with
    # the level-anchored regression output keeps low-frequency content
    # locked to the measurements while suppressing false reversals.
    reint = np.empty(output_length)
    reint[0] = y[0]
    reint[1:] = y[0] + np.cumsum(est_slope[1:])
    y = 0.75 * y + 0.25 * reint

    # Light 3-tap post-smoothing: negligible added phase delay.
    if output_length >= 3:
        y[1:-1] = 0.25 * y[:-2] + 0.5 * y[1:-1] + 0.25 * y[2:]

    # ---- Lag compensation: small forward extrapolation along fitted slope ----
    y = y + 1.0 * b_fit

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