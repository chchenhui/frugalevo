# EVOLVE-BLOCK-START
import numpy as np
from scipy.signal import savgol_filter


def _adaptive_kalman_pass(y, q_scale=2.5, vol_win=5):
    """Forward adaptive-Q Kalman (constant-velocity model, level domain only)."""
    n = len(y)
    x = np.zeros(n)
    # Noise variance estimate from first differences (robust MAD)
    d = np.diff(y)
    sigma_n = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0)
    sigma_n = max(sigma_n, 1e-8)
    R = sigma_n ** 2

    # State: [level, slope]
    xm = np.array([y[0], 0.0])
    P = np.eye(2) * (sigma_n ** 2 + 1.0)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    I = np.eye(2)

    # Local volatility for Q adaptation
    pad = np.concatenate([[y[0]] * vol_win, y, [y[-1]] * vol_win])
    vol = np.array([np.std(pad[i:i + 2 * vol_win + 1]) for i in range(n)])
    vol = np.maximum(vol, sigma_n)

    for k in range(n):
        # Predict
        xm = F @ xm
        P = F @ P @ F.T
        # Adaptive process noise: scale with local volatility
        q = q_scale * (vol[k] ** 2) / max(vol_win, 1)
        P[0, 0] += q
        P[1, 1] += 0.25 * q
        # Update
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T / S).flatten()
        innov = y[k] - (H @ xm)[0]
        xm = xm + K * innov
        P = (I - np.outer(K, H)) @ P
        x[k] = xm[0]
    return x


def _bidirectional_kalman(y, crossover=0.25):
    """Fuse forward and backward Kalman passes with linear crossover weighting."""
    fwd = _adaptive_kalman_pass(y)
    bwd = _adaptive_kalman_pass(y[::-1])[::-1]
    n = len(y)
    t = np.linspace(0.0, 1.0, n)
    # Weight on forward pass: 1 near start, 0 near end, linear in transition zones
    w = np.ones(n)
    lo = crossover
    hi = 1.0 - crossover
    mask = (t > lo) & (t < hi)
    w[mask] = 0.5 + 0.5 * np.cos(np.pi * (t[mask] - lo) / (hi - lo))
    w[t >= hi] = 0.0
    return w * fwd + (1.0 - w) * bwd


def _robust_sg(x, window=9, poly=2, iters=2):
    """Robust Savitzky-Golay with residual-based outlier downweighting."""
    x = savgol_filter(x, window, poly, mode="interp")
    for _ in range(iters):
        resid = np.abs(x - savgol_filter(x, window, poly, mode="interp"))
        s = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-8
        w = np.clip(1.0 - (resid / (3.0 * s)) ** 2, 0.05, 1.0)
        xs = savgol_filter(x * w, window, poly, mode="interp")
        norm = savgol_filter(w, window, poly, mode="interp") + 1e-8
        x = xs / norm
    return x


def _recency_convolution(x, half_life=12):
    """Causal convolution with exponentially decaying kernel (mild lag, high recency weight)."""
    n = len(x)
    L = min(n, 4 * half_life)
    k = np.exp(-np.arange(L) / half_life)
    k /= k.sum()
    out = np.convolve(x, k, mode="full")[:n]
    # Zero-phase correction for the interior (leave endpoints causal)
    return out


def _median_snap(x, y, window=5, thresh=2.5):
    """Level-domain median snap: pull output toward local input median when drifted."""
    n = len(x)
    pad = np.concatenate([[y[0]] * window, y, [y[-1]] * window])
    med = np.array([np.median(pad[i:i + 2 * window + 1]) for i in range(n)])
    dev = x - med
    s = 1.4826 * np.median(np.abs(dev - np.median(dev))) + 1e-8
    corr = np.where(np.abs(dev) > thresh * s, 0.5 * (med - x), 0.0)
    return x + corr


def filter_signal(y):
    """Full cascade: bidirectional adaptive-Q Kalman -> robust SG -> recency conv -> median snap."""
    y = np.asarray(y, dtype=float)
    if len(y) < 8:
        return y.copy()
    x = _bidirectional_kalman(y, crossover=0.25)
    x = _robust_sg(x, window=9, poly=2, iters=2)
    x = _recency_convolution(x, half_life=12)
    x = _median_snap(x, y, window=5, thresh=2.5)
    return x
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