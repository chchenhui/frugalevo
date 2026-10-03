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

    # Vectorized moving average via cumulative sum, O(n)
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[window_size:] - c[:-window_size]) / window_size


def _monotone_run_projection(y, min_run=3, amp_k=2.0, max_iter=200):
    """
    Monotone-run projection: post-process a smoother's output into a
    piecewise-monotone signal by merging spurious wiggle runs.

    Diffs of the base output are run-length encoded by sign. A run is a
    spurious wiggle when it is short (duration < min_run samples) AND its
    net displacement is below a noise-scaled threshold (amp_k * MAD of
    the diffs). The wiggle's diffs are replaced by the average slope of
    the dominant neighbouring run, and the displacement excess is
    redistributed over that neighbour so the merged segment's net
    displacement (hence the signal level) is preserved exactly. Output
    is rebuilt with np.cumsum. Genuine reversals (long runs and/or large
    displacements, e.g. step changes) fail at least one criterion and
    are left untouched.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return y

    mod = np.diff(y).copy()
    m = len(mod)
    med = np.median(mod)
    amp_thresh = amp_k * max(np.median(np.abs(mod - med)), 1e-12)

    for _ in range(max_iter):
        s = np.sign(mod)
        bounds = np.flatnonzero(np.diff(s)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [m]))
        nruns = len(starts)
        if nruns < 3:
            break
        merged = False
        for i in range(nruns):
            lo, hi = starts[i], ends[i]
            if hi - lo >= min_run or s[lo] == 0:
                continue
            disp_i = mod[lo:hi].sum()
            if abs(disp_i) >= amp_thresh:
                continue
            # dominant neighbouring run (larger net displacement)
            cands = [j for j in (i - 1, i + 1)
                     if 0 <= j < nruns and s[starts[j]] != 0]
            if not cands:
                continue
            dom = max(cands, key=lambda j: abs(mod[starts[j]:ends[j]].sum()))
            dlo, dhi = starts[dom], ends[dom]
            disp_d = mod[dlo:dhi].sum()
            if abs(disp_d) <= abs(disp_i):
                continue
            repl = disp_d / (dhi - dlo)
            adj = ((hi - lo) * repl - disp_i) / (dhi - dlo)
            new_dom = mod[dlo:dhi] - adj
            # only apply if the dominant run's sign is preserved
            if (disp_d > 0 and np.all(new_dom > 0)) or \
               (disp_d < 0 and np.all(new_dom < 0)):
                mod[lo:hi] = repl
                mod[dlo:dhi] = new_dom
                merged = True
                break
        if not merged:
            break

    out = np.empty(n)
    out[0] = y[0]
    out[1:] = y[0] + np.cumsum(mod)
    return out


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Two-pass Rauch-Tung-Striebel (RTS) Kalman smoother with a
    constant-velocity state model and adaptive process noise.

    Forward pass: constant-velocity Kalman filter (state = [level, slope]).
    The velocity state acts as a smoothness prior, suppressing spurious
    slope reversals. Process noise Q is inflated online when the
    normalized innovation exceeds a 3-sigma gate (detected level shifts),
    so genuine jumps are tracked without ringing or lag. Joseph-form
    covariance updates keep the recursion numerically stable.

    Backward pass: the true RTS recursion (not forward/backward averaging)
    optimally fuses future information into each estimate, eliminating
    phase lag while keeping piecewise-smooth dynamics.

    Projection pass: the smoothed level is post-processed by a
    monotone-run projection that merges short, low-amplitude wiggle
    runs into the dominant neighbouring trend (displacement-preserving),
    directly reducing spurious slope reversals without adding lag.

    Output length contract: len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < window_size:
        raise ValueError(f"Input signal length ({n}) must be >= window_size ({window_size})")

    # Robust measurement-noise estimate from first differences
    d = np.diff(x)
    r_var = max(np.median(np.abs(d - np.median(d))) ** 2, 1e-6)
    q_base = max(r_var * 0.005, 1e-9)  # tight smoothness prior
    gate, q_boost = 3.0, 50.0

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Ft = F.T
    I2 = np.eye(2)
    Q0 = np.array([[0.25, 0.5], [0.5, 1.0]])

    # Storage for RTS backward pass
    xf = np.empty((n, 2))
    Pf = np.empty((n, 2, 2))
    xp_arr = np.empty((n, 2))
    Pp_arr = np.empty((n, 2, 2))

    level, slope = x[0], 0.0
    P = r_var * I2
    xf[0] = (level, slope)
    Pf[0] = P

    for k in range(1, n):
        # Predict
        xp = F @ xf[k - 1]
        Pp = F @ Pf[k - 1] @ Ft
        # Adaptive Q via innovation gating (3-sigma)
        innov = x[k] - xp[0]
        S = Pp[0, 0] + r_var
        nis = innov * innov / S
        q = q_base * (q_boost if nis > gate * gate else 1.0)
        Pp = Pp + q * Q0
        # Update (Joseph form)
        K = Pp[:, 0] / S
        xf[k] = xp + K * innov
        A = I2 - np.outer(K, (1.0, 0.0))
        Pf[k] = A @ Pp @ A.T + np.outer(K, K) * r_var
        xp_arr[k] = xp
        Pp_arr[k] = Pp

    # RTS backward pass
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        Cg = Pf[k] @ Ft @ np.linalg.inv(Pp_arr[k + 1])
        xs[k] = xf[k] + Cg @ (xs[k + 1] - xp_arr[k + 1])

    # Monotone-run projection: remove short, small-amplitude wiggle runs
    # (spurious slope reversals) while preserving displacement and length.
    return _monotone_run_projection(xs[:, 0][window_size - 1:])


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
