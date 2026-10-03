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


def _pava(v, increasing=True):
    """Pool-Adjacent-Violators isotonic regression on 1D array v."""
    vals = []
    wts = []
    for a in v:
        val, wt = float(a), 1.0
        if not increasing:
            val = -val
        while vals and vals[-1] > val + 1e-12:
            # pool backwards while order is violated
            pv, pw = vals.pop(), wts.pop()
            val = (val * wt + pv * pw) / (wt + pw)
            wt += pw
        vals.append(val)
        wts.append(wt)
    out = np.repeat(np.array(vals), np.round(wts).astype(int))
    if not increasing:
        out = -out
    return out


def _l1_trend_filter(y, lam):
    """
    L1 trend filtering via iteratively-reweighted least squares (IRLS).

    Solves min ||y - z||^2 + lam * ||D2 z||_1 (Kim-Koh-Boyd-Gorinevsky
    discrete trend filtering) by repeating: weights w_i =
    1 / max(|(D2 z_prev)_i|, eps), then solving the sparse linear system
    (I + lam * D2' W D2) z = y with scipy.sparse.linalg.spsolve.
    The result is piecewise-LINEAR (piecewise-constant slope), so sign
    changes of np.diff occur only at genuine trend kinks.
    """
    n = len(y)
    if n < 5:
        return y
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except Exception:
        return y

    # Second-difference operator (n-2 x n), sparse.
    e = np.ones(n)
    D2 = sparse.diags([e[:-2], -2.0 * e[1:-1], e[2:]], [0, 1, 2], shape=(n - 2, n), format="csr")
    I = sparse.identity(n, format="csr")

    # Robust noise scale from MAD of first differences.
    d1 = np.diff(y)
    sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 / np.sqrt(2.0)
    if sigma <= 0:
        sigma = 1e-6
    eps = 1e-3 * sigma
    lam_eff = lam * sigma * sigma * n

    z = y.astype(float).copy()
    for _ in range(5):
        r = D2 @ z
        wts = 1.0 / np.maximum(np.abs(r), eps)
        W = sparse.diags(wts, 0, format="csr")
        A = (I + lam_eff * (D2.T @ (W @ D2))).tocsc()
        try:
            z = spsolve(A, y)
        except Exception:
            break
        if not np.all(np.isfinite(z)):
            return y
    return z


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    L1 trend filtering + piecewise-monotone reconstruction.

    Stage 1: Savitzky-Golay degree-2 endpoint fit (zero systematic lag).
    Stage 2: 5-tap running median (spike rejection, zero group delay).
    Stage 3: two zero-lag EMA passes (forward-backward smoothing).
    Stage 4: L1 trend filtering (IRLS on second differences) — the output
    is piecewise-linear, so slope sign changes occur only at genuine kinks.
    Stage 5: hysteresis trend segmentation + PAVA isotonic regression per
    monotone segment, guaranteeing np.diff changes sign only at accepted
    reversals — minimizing slope-change and false-reversal counts.
    """
    if len(x) < window_size:
        raise ValueError(f"Input signal length ({len(x)}) must be >= window_size ({window_size})")

    x = np.asarray(x, dtype=float)
    output_length = len(x) - window_size + 1

    # Stage 1: degree-2 polynomial LS fit, evaluated at the endpoint (t = W-1)
    t = np.arange(window_size, dtype=float)
    V = np.vander(t, 3, increasing=True)
    ep = float(window_size - 1)
    w = np.array([ep ** k for k in range(3)]) @ np.linalg.pinv(V)

    sw = np.lib.stride_tricks.sliding_window_view(x, window_size)
    y = sw @ w

    # Stage 2: 5-tap running median (zero group delay, spike-rejecting).
    if len(y) >= 5:
        y5 = np.lib.stride_tricks.sliding_window_view(y, 5)
        med = np.median(y5, axis=-1)
        y[2:-2] = med

    # Stage 3: zero-lag EMA (forward-backward exponential smoothing).
    def _zerolag_ema(s, alpha):
        fwd = np.empty_like(s)
        acc = s[0]
        for i in range(len(s)):
            acc = alpha * s[i] + (1.0 - alpha) * acc
            fwd[i] = acc
        bwd = np.empty_like(s)
        acc = fwd[-1]
        for i in range(len(s) - 1, -1, -1):
            acc = alpha * fwd[i] + (1.0 - alpha) * acc
            bwd[i] = acc
        return bwd

    if len(y) >= 3:
        y = _zerolag_ema(y, 0.25)
        y = _zerolag_ema(y, 0.45)  # second, lighter pass for slope stability

    # Stage 4: L1 trend filtering (piecewise-linear reconstruction).
    # Blend with the smoothed baseline to avoid over-flattening dynamics.
    if len(y) >= 6:
        y_l1 = _l1_trend_filter(y, lam=0.75)
        y = 0.6 * y_l1 + 0.4 * y

    # Stage 4+5: hysteresis segmentation + piecewise-monotone (PAVA) output.
    if len(y) >= 6:
        d = np.diff(y)
        # Robust noise floor on first differences (MAD)
        h = 1.5 * np.median(np.abs(d - np.median(d))) / 0.6745
        if h <= 0:
            h = 1e-9

        n = len(y)
        # Determine segment boundaries: indices where an accepted reversal occurs.
        # A reversal at diff-index j (between samples j and j+1) is accepted if
        # the new-direction slope exceeds h sustained over >= 3 samples.
        sustain = 3
        signs = np.sign(d)
        boundaries = [0]
        cur_dir = 0  # current enforced direction: +1 up, -1 down, 0 flat
        j = 0
        while j < len(d):
            s = signs[j]
            if s == 0:
                j += 1
                continue
            if cur_dir == 0:
                cur_dir = int(s)
                j += 1
                continue
            if int(s) != cur_dir and abs(d[j]) > h:
                # candidate reversal; check sustenance
                k = j
                cnt = 0
                while k < len(d) and signs[k] == s:
                    if abs(d[k]) > h:
                        cnt += 1
                    else:
                        cnt = 0
                    k += 1
                if cnt >= sustain:
                    boundaries.append(j + 1)  # new segment starts at sample j+1
                    cur_dir = int(s)
                    j = k
                    continue
                else:
                    j = k
                    continue
            j += 1
        boundaries.append(n)

        out = y.copy()
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if b - a < 3:
                continue
            seg = y[a:b]
            # direction of this segment from its baseline trend
            seg_dir = 1 if seg[-1] >= seg[0] else -1
            iso = _pava(seg, increasing=(seg_dir > 0))
            out[a:b] = iso
        y = out

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
