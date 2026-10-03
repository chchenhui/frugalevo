import numpy as np


def _median3(x):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return x.copy()
    z = np.empty_like(x)
    z[0] = x[0]
    z[-1] = x[-1]
    z[1:-1] = np.median(np.stack((x[:-2], x[1:-1], x[2:]), axis=0), axis=0)
    return z


def _reflect_convolve(x, kernel):
    r = len(kernel) // 2
    if r == 0:
        return x.copy()
    xp = np.pad(x, (r, r), mode="reflect")
    return np.convolve(xp, kernel, mode="valid")


def _gaussian(sigma, radius):
    t = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / np.sum(k)


def adaptive_filter(x, window_size=20):
    x = np.asarray(x, dtype=float)
    if len(x) < window_size:
        raise ValueError("input shorter than window")
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Apply an adaptive constant-acceleration Kalman filter and RTS smoother."""
    x = np.asarray(x, dtype=float).copy()
    if len(x) < window_size:
        raise ValueError("input shorter than window")

    # Interpolate missing observations before state estimation.
    if not np.all(np.isfinite(x)):
        good = np.isfinite(x)
        if not np.any(good):
            x[:] = 0.0
        else:
            idx = np.arange(len(x))
            x = np.interp(idx, idx[good], x[good])

    n = len(x)
    if n == 0:
        return np.empty(0, dtype=float)

    # Constant-acceleration dynamics with unit sample spacing.
    F = np.array([[1.0, 1.0, 0.5],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]], dtype=float)
    H = np.array([1.0, 0.0, 0.0], dtype=float)
    I = np.eye(3, dtype=float)

    # First differences provide a robust measurement-noise scale.
    d = np.diff(x)
    if d.size:
        dm = np.median(d)
        scale = 1.4826 * np.median(np.abs(d - dm)) / np.sqrt(2.0)
    else:
        scale = 0.0
    scale = max(float(scale), 1e-4)
    R = max(scale * scale, 1e-6)

    # Initialize level, velocity, and acceleration conservatively.
    level = float(x[0])
    velocity = float(np.median(d[-min(5, d.size):])) if d.size else 0.0
    state = np.array([level, velocity, 0.0], dtype=float)
    P = np.diag([R * 4.0, R * 2.0, R],).astype(float)

    filtered = np.empty((n, 3), dtype=float)
    covariances = np.empty((n, 3, 3), dtype=float)
    transitions = np.repeat(F[None, :, :], max(n - 1, 1), axis=0)
    process_covariances = np.empty((max(n - 1, 1), 3, 3), dtype=float)

    # Base process covariance permits gradual acceleration changes.  A
    # deliberately conservative baseline suppresses isolated reversals; the
    # innovation-dependent multiplier below still increases agility at genuine
    # bends and abrupt regime changes.
    q = max(R * 0.015, 1e-7)
    # White-jerk covariance for the constant-acceleration model.  Compared
    # with the loosely coupled baseline, this preserves smooth level/velocity
    # evolution while allowing acceleration to change coherently at bends.
    Q_base = q * np.array([[1.0 / 36.0, 1.0 / 12.0, 1.0 / 6.0],
                           [1.0 / 12.0, 1.0 / 4.0, 1.0 / 2.0],
                           [1.0 / 6.0, 1.0 / 2.0, 1.0]], dtype=float)

    for k in range(n):
        if k:
            innovation_prediction = float(x[k] - H @ (F @ state))
            multiplier = min(9.0, 1.0 + max(0.0,
                abs(innovation_prediction) / (3.0 * scale) - 1.0) ** 2)
            Q = Q_base * multiplier
            predicted = F @ state
            predicted_cov = F @ P @ F.T + Q
            transitions[k - 1] = F
            process_covariances[k - 1] = Q
        else:
            predicted = state
            predicted_cov = P
            process_covariances[0] = Q_base

        predicted_cov = (predicted_cov + predicted_cov.T) * 0.5
        predicted_cov.flat[::4] += 1e-10

        innovation = float(x[k] - H @ predicted)
        innovation_var = max(float(H @ predicted_cov @ H + R), 1e-9)
        gain = (predicted_cov @ H) / innovation_var
        state = predicted + gain * innovation

        # Joseph-form covariance update preserves positive semidefiniteness.
        A = I - np.outer(gain, H)
        P = A @ predicted_cov @ A.T + R * np.outer(gain, gain)
        P = (P + P.T) * 0.5
        P.flat[::4] = np.maximum(P.flat[::4], 1e-10)

        filtered[k] = state
        covariances[k] = P

    # Rauch–Tung–Striebel backward pass removes forward-filter lag.
    smoothed = filtered.copy()
    for k in range(n - 2, -1, -1):
        predicted_cov = F @ covariances[k] @ F.T + process_covariances[k]
        predicted_cov = (predicted_cov + predicted_cov.T) * 0.5
        predicted_cov.flat[::4] += 1e-10
        # Solve the RTS system instead of forming a pseudoinverse.  Explicit
        # symmetrization and diagonal regularization improve stability when
        # adaptive process noise makes the covariance nearly ill-conditioned.
        predicted_cov = (predicted_cov + predicted_cov.T) * 0.5
        predicted_cov.flat[::4] += 1e-9
        gain = np.linalg.solve(predicted_cov, F @ covariances[k]).T
        smoothed[k] = filtered[k] + gain @ (smoothed[k + 1] - F @ filtered[k])

    return np.asarray(smoothed[window_size - 1:, 0], dtype=float)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
    return adaptive_filter(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, length)
    clean_signal = (
        2 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2 * t)
        + 0.5 * np.sin(2 * np.pi * 5 * t)
        + 0.8 * np.exp(-t / 5) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean_signal += 0.1 * t * np.sin(0.2 * t)
    clean_signal += np.cumsum(rng.randn(length) * 0.05)
    return clean_signal + rng.normal(0, noise_level, length), clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level)
    else:
        noisy_signal = np.asarray(noisy_signal, dtype=float)
        clean_signal = None

    filtered = process_signal(noisy_signal, window_size, "enhanced")
    result = {
        "filtered_signal": filtered,
        "clean_signal": None,
        "noisy_signal": None,
        "correlation": 0,
        "noise_reduction": 0,
        "signal_length": len(filtered),
    }

    if clean_signal is not None:
        clean = clean_signal[window_size - 1:]
        noisy = noisy_signal[window_size - 1:]
        m = min(len(filtered), len(clean))
        filtered = filtered[:m]
        clean = clean[:m]
        noisy = noisy[:m]
        corr = np.corrcoef(filtered, clean)[0, 1] if m > 1 else 0.0
        before = np.var(noisy - clean)
        after = np.var(filtered - clean)
        result.update({
            "filtered_signal": filtered,
            "clean_signal": clean,
            "noisy_signal": noisy,
            "correlation": float(corr),
            "noise_reduction": float((before - after) / before)
                if before > 0 else 0.0,
            "signal_length": m,
        })
    return result


if __name__ == "__main__":
    r = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {r['correlation']:.3f}")
    print(f"Noise reduction: {r['noise_reduction']:.3f}")
    print(f"Processed signal length: {r['signal_length']}")