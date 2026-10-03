# EVOLVE-BLOCK-START


# ----------------------------------------------------------------------------
# Stage interface
# ----------------------------------------------------------------------------
class Stage:
    """Base class for zero-phase pipeline stages."""
    def apply(self, x: np.ndarray, state: dict) -> np.ndarray:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# Stage 1: two-pass RTS smoother core
# ----------------------------------------------------------------------------
class RTSSmootherStage(Stage):
    """
    Constant-acceleration Kalman filter + RTS backward smoothing, run twice.
    Pass 1 estimates noise statistics; pass 2 re-smooths with adapted gains.
    """
    def __init__(self, q_scale=0.02, r_frac=0.10, warmup=8):
        self.q_scale = q_scale
        self.r_frac = r_frac
        self.warmup = warmup

    @staticmethod
    def _forward(x, q, r):
        n = len(x)
        # State: [pos, vel, acc]
        F = np.array([[1, 1, 0.5],
                      [0, 1, 1.0],
                      [0, 0, 1.0]])
        H = np.array([[1.0, 0.0, 0.0]])
        P = np.eye(3)
        s = np.zeros(3)
        s[0] = x[0]
        xs_f = np.zeros((n, 3))
        Pf = np.zeros((n, 3, 3))
        I3 = np.eye(3)
        for k in range(n):
            if k > 0:
                s = F @ s
                P = F @ P @ F.T
                P[2, 2] += q
            innov = x[k] - (H @ s)[0]
            S = (H @ P @ H.T)[0, 0] + r
            K = (P @ H.T / S).ravel()
            s = s + K * innov
            A = I3 - np.outer(K, H)
            P = A @ P @ A.T + (r / S) * np.outer(K, K)  # Joseph form
            xs_f[k] = s
            Pf[k] = P
        return xs_f, Pf, F

    @staticmethod
    def _backward(xs_f, Pf, F):
        n = len(xs_f)
        xs = xs_f.copy()
        for k in range(n - 2, -1, -1):
            P_pred = F @ Pf[k] @ F.T
            P_pred[2, 2] += 1e-9
            G = Pf[k] @ F.T @ np.linalg.inv(P_pred)
            xs[k] = xs_f[k] + G @ (xs[k + 1] - F @ xs_f[k])
        return xs

    def _one_pass(self, x, q, r):
        xs_f, Pf, F = self._forward(x, q, r)
        return self._backward(xs_f, Pf, F)

    def apply(self, x: np.ndarray, state: dict) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 3:
            return x.copy()

        sigma_r = max(np.std(np.diff(x)) / np.sqrt(2.0), 1e-8) if n > 2 else 1e-8
        r0 = max(self.r_frac * sigma_r**2, 1e-12)
        q0 = max(self.q_scale * r0, 1e-12)

        # Pass 1
        xs1 = self._one_pass(x, q0, r0)
        # Adapt measurement noise from residuals
        resid = x - xs1[:, 0]
        r1 = max(np.var(resid[self.warmup:]) if n > self.warmup else r0, 1e-12)
        r1 = 0.5 * (r0 + r1)
        q1 = max(self.q_scale * r1, 1e-12)

        # Pass 2
        xs2 = self._one_pass(x, q1, r1)

        state['sigma_r'] = float(np.sqrt(r1))
        state['vel'] = xs2[:, 1].copy()
        state['acc'] = xs2[:, 2].copy()
        return xs2[:, 0].copy()


# ----------------------------------------------------------------------------
# Stage 2: zero-phase velocity clamp (Gen 41 fusion)
# ----------------------------------------------------------------------------
class ZeroPhaseVelocityClampStage(Stage):
    """
    Limit per-step displacement to 2*|velocity| + slack*sigma_r.
    Applied forward and mirrored backward to stay zero-phase. A correlation
    guard checks fidelity; if corr < 0.97 the slack doubles and the clamp
    is retried once.
    """
    def __init__(self, slack=0.5, min_corr=0.97, max_ratio=2.0):
        self.slack = slack
        self.min_corr = min_corr
        self.max_ratio = max_ratio

    @staticmethod
    def _clamp_sweep(y, vel, sigma_r, slack):
        n = len(y)
        for k in range(1, n):
            limit = 2.0 * abs(vel[k]) + slack * sigma_r
            if limit <= 0:
                continue
            d = y[k] - y[k - 1]
            if abs(d) > limit:
                y[k] = y[k - 1] + np.sign(d) * limit
        return y

    def _clamp(self, y, vel, sigma_r, slack):
        # forward sweep
        yf = self._clamp_sweep(y.copy(), vel, sigma_r, slack)
        # mirrored backward sweep (zero-phase)
        yb = self._clamp_sweep(yf[::-1].copy(), vel[::-1], sigma_r, slack)[::-1]
        # average the two sweeps' corrections to keep symmetric behavior
        return 0.5 * (yf + yb)

    @staticmethod
    def _corr(a, b):
        sa, sb = np.std(a), np.std(b)
        if sa < 1e-12 or sb < 1e-12:
            return 1.0
        return float(np.corrcoef(a, b)[0, 1])

    def apply(self, x: np.ndarray, state: dict) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        vel = state.get('vel')
        sigma_r = state.get('sigma_r', 1e-8)
        if vel is None or len(vel) != len(x) or len(x) < 3:
            return x.copy()

        slack = self.slack
        y = self._clamp(x, vel, sigma_r, slack)
        if self._corr(y, x) < self.min_corr:
            y = self._clamp(x, vel, sigma_r, min(slack * 2, self.max_ratio * slack))
        return y


# ----------------------------------------------------------------------------
# Stage 3: light bilateral damper
# ----------------------------------------------------------------------------
class BilateralDamperStage(Stage):
    """
    Edge-preserving exponential damper applied forward and backward and
    averaged (zero-phase). Cheap: O(n * passes).
    """
    def __init__(self, alpha=0.35, edge_thresh=1.5):
        self.alpha = alpha
        self.edge_thresh = edge_thresh

    @staticmethod
    def _ewma_one_sided(x, alpha, thresh, sigma_r):
        y = x.copy()
        n = len(y)
        for k in range(1, n):
            diff = y[k] - y[k - 1]
            a = alpha
            if abs(diff) > thresh * sigma_r:
                a = min(1.0, alpha * 3.0)  # preserve dynamics at edges
            y[k] = y[k - 1] + a * (x[k] - y[k - 1])
        return y

    def apply(self, x: np.ndarray, state: dict) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = len(x)
        if n < 3:
            return x.copy()
        sigma_r = state.get('sigma_r', 1e-8)
        fwd = self._ewma_one_sided(x, self.alpha, self.edge_thresh, sigma_r)
        bwd = self._ewma_one_sided(x[::-1], self.alpha, self.edge_thresh, sigma_r)[::-1]
        return 0.5 * (fwd + bwd)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class SmoothingPipeline:
    def __init__(self):
        self.stages = [
            RTSSmootherStage(),
            ZeroPhaseVelocityClampStage(),
            BilateralDamperStage(),
        ]

    def process(self, x: np.ndarray) -> np.ndarray:
        state = {}
        y = np.asarray(x, dtype=float)
        for stage in self.stages:
            y = stage.apply(y, state)
        return y


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