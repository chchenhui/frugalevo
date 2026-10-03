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
    """Apply a robust adaptive constant-acceleration Kalman/RTS smoother with conservative jerk adaptation."""
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

    # Robust regularized spline continuation: solve three fixed-span
    # curvature-penalized fits, reweighting large residuals twice and selecting
    # the candidate with the best residual/reversal tradeoff.
    if n == 1:
        return x[window_size - 1:].copy()

    d = np.diff(x)
    dm = float(np.median(d)) if d.size else 0.0
    sigma = max(1.4826 * float(np.median(np.abs(d - dm))) / np.sqrt(2.0),
                 1e-4)

    # Allocate the observation across coarse and residual scales.  The
    # reflected Gaussian component carries persistent trend, while the
    # residual is retained conservatively and attenuated near derivative
    # reversals where isolated noise is most likely.
    time_scales = (5.0, 9.0, 15.0)
    residual_budget = np.empty((len(time_scales), n), dtype=float)
    coarse_components = np.empty((len(time_scales), n), dtype=float)
    for i, scale in enumerate(time_scales):
        radius = max(1, int(np.ceil(2.5 * scale)))
        coarse = _reflect_convolve(x, _gaussian(scale, radius))
        residual = x - coarse
        residual_slope = np.diff(residual)
        if residual_slope.size > 1:
            turns = np.zeros(residual_slope.size, dtype=bool)
            turns[1:] = (
                np.sign(residual_slope[1:]) !=
                np.sign(residual_slope[:-1])
            )
            turns = np.pad(turns, (0, 1), mode="edge")
        else:
            turns = np.zeros(n, dtype=bool)
        residual_scale = max(float(np.std(residual)), 1e-8)
        bounded = np.clip(
            residual,
            -0.55 * residual_scale,
            0.55 * residual_scale,
        )
        gain = np.where(turns, 0.25, 0.75)
        coarse_components[i] = coarse
        residual_budget[i] = coarse + gain * bounded

    # D2.T @ D2 is pentadiagonal; constructing it explicitly keeps this
    # implementation NumPy-only and deterministic for the modest input sizes.
    D2 = np.zeros((max(n - 2, 1), n), dtype=float)
    if n > 2:
        rows = np.arange(n - 2)
        D2[rows, rows] = 1.0
        D2[rows, rows + 1] = -2.0
        D2[rows, rows + 2] = 1.0
    curvature = D2.T @ D2
    identity = np.eye(n, dtype=float)
    candidates = []

    # Three robust curvature passes.  Tukey weights redescend on isolated
    # outliers, while the short-run projection suppresses unsupported turns.
    derivative = np.diff(x)
    if derivative.size:
        padded = np.pad(derivative, (2, 2), mode="edge")
        robust_derivative = np.median(
            np.stack([padded[i:i + derivative.size] for i in range(5)]),
            axis=0)
        supported = np.sign(robust_derivative)
    else:
        supported = np.empty(0, dtype=float)

    for candidate_index, span in enumerate((9.0, 15.0, 23.0)):
        """Solve one scale-allocated robust curvature fit and project weak turns."""
        regularization = (span ** 4) / 16.0
        target = residual_budget[min(candidate_index, len(time_scales) - 1)]
        weights = np.ones(n, dtype=float)
        estimate = target.copy()
        system = weights[:, None] * identity + regularization * curvature
        estimate = np.linalg.solve(system, weights * target)
        if not np.all(np.isfinite(estimate)):
            estimate = x.copy()
        residual = x - estimate
        ratio = residual / max(4.0 * sigma, 1e-12)
        weights = np.where(
            np.abs(ratio) < 1.0, (1.0 - ratio * ratio) ** 2, 0.0)
        weights = np.maximum(weights, 0.04)
        system = weights[:, None] * identity + regularization * curvature
        estimate = np.linalg.solve(system, weights * target)
        if not np.all(np.isfinite(estimate)):
            estimate = x.copy()

        slope = np.diff(estimate)
        signs = np.sign(slope)
        if signs.size and supported.size:
            # Project only unsupported short reversals.  A reversal is
            # retained when both neighboring raw derivative runs have at
            # least three samples and agree with their fitted directions.
            runs = []
            start = 0
            for j in range(1, signs.size + 1):
                if j == signs.size or signs[j] != signs[start]:
                    runs.append((start, j, signs[start]))
                    start = j

            supported_runs = []
            start = 0
            for j in range(1, supported.size + 1):
                if j == supported.size or supported[j] != supported[start]:
                    supported_runs.append((start, j, supported[start]))
                    start = j

            for j in range(1, len(runs) - 1):
                left = runs[j - 1]
                current = runs[j]
                right = runs[j + 1]
                left_support = [
                    a for a, b, s in supported_runs
                    if b > left[0] and a < left[1] and s != 0
                    and (b - a) >= 3
                ]
                right_support = [
                    a for a, b, s in supported_runs
                    if b > right[0] and a < right[1] and s != 0
                    and (b - a) >= 3
                ]
                keep = bool(left_support and right_support)
                if not keep:
                    a, b = current[0], current[1]
                    end = min(b + 1, n - 1)
                    if end > a:
                        estimate[a:end] = np.linspace(
                            estimate[a], estimate[end], end - a)
            slope = np.diff(estimate)

        residual = target - estimate
        signs = np.sign(slope)
        signs = signs[signs != 0.0]
        reversals = (np.sum(signs[1:] != signs[:-1])
                     if signs.size > 1 else 0)
        residual_variance = float(np.mean(residual * residual))
        departure = float(np.mean(np.abs(slope - derivative)))
        objective = (residual_variance +
                     0.50 * sigma * sigma * reversals / max(n, 1) +
                     0.08 * departure)
        candidates.append((objective, estimate))

    estimate = min(candidates, key=lambda item: item[0])[1]
    return np.asarray(estimate[window_size - 1:], dtype=float)

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
    q = max(R * 0.0085, 1e-7)
    # White-jerk covariance for the constant-acceleration model.  Compared
    # with the loosely coupled baseline, this preserves smooth level/velocity
    # evolution while allowing acceleration to change coherently at bends.
    Q_base = q * np.array([[1.0 / 36.0, 1.0 / 12.0, 1.0 / 6.0],
                           [1.0 / 12.0, 1.0 / 4.0, 1.0 / 2.0],
                           [1.0 / 6.0, 1.0 / 2.0, 1.0]], dtype=float)

    for k in range(n):
        if k:
            innovation_prediction = float(x[k] - H @ (F @ state))
            # Require a distinctly large innovation before increasing process
            # noise, and use a lower cap to avoid fitting isolated impulses.
            multiplier = min(7.0, 1.0 + max(0.0,
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

        # Robustify isolated spikes without suppressing regime changes.  The
        # adaptive process covariance above responds to sustained bends, while
        # this Huber-like measurement inflation prevents one-sample outliers
        # from creating artificial velocity reversals.
        normalized_innovation = abs(innovation) / max(scale, 1e-6)
        measurement_scale = 1.0 + max(0.0, normalized_innovation - 2.5) ** 2
        effective_R = R * min(measurement_scale, 16.0)

        innovation_var = max(float(H @ predicted_cov @ H + effective_R), 1e-9)
        gain = (predicted_cov @ H) / innovation_var
        state = predicted + gain * innovation

        # Joseph-form covariance update preserves positive semidefiniteness.
        A = I - np.outer(gain, H)
        P = A @ predicted_cov @ A.T + effective_R * np.outer(gain, gain)
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