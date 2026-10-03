# EVOLVE-BLOCK-START
"""
Causal adaptive multi-scale robust endpoint filter.

The filter emits one estimate for every complete input window and aligns each
estimate with the newest observation in that window.  It combines robust local
regressions with asymmetric, direction-aware hysteresis: continuation is easy,
while a reversal needs stronger and persistent evidence.
"""
import numpy as np


def _robust_scale(values, fallback=1e-6):
    """Finite MAD scale with standard-deviation and fallback protection."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return float(fallback)

    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))

    if not np.isfinite(scale) or scale < 1e-10:
        scale = np.std(values)

    if not np.isfinite(scale) or scale < 1e-10:
        scale = max(float(fallback), 1e-6)

    return float(scale)


def _robust_endpoint_fit(window, decay, fallback_scale):
    """
    Robust causal exponentially weighted local-linear endpoint regression.

    Returns:
        endpoint estimate, slope per sample, residual noise scale
    """
    window = np.asarray(window, dtype=float)
    valid = np.isfinite(window)

    if not np.any(valid):
        return np.nan, 0.0, float(fallback_scale)

    if np.count_nonzero(valid) == 1:
        value = float(window[valid][0])
        return value, 0.0, float(fallback_scale)

    length = len(window)
    time_full = np.arange(length, dtype=float) - float(length - 1)
    time = time_full[valid]
    values = window[valid]

    tau = max(float(decay), 1.0)
    weights = np.exp(time / tau)
    weights /= max(np.sum(weights), 1e-12)

    def solve(current_weights):
        sw = np.sum(current_weights)
        st = np.sum(current_weights * time)
        stt = np.sum(current_weights * time * time)
        sz = np.sum(current_weights * values)
        stz = np.sum(current_weights * time * values)

        determinant = sw * stt - st * st
        if determinant <= 1e-14:
            return float(np.average(values, weights=current_weights)), 0.0

        intercept = (stt * sz - st * stz) / determinant
        slope = (sw * stz - st * sz) / determinant
        return float(intercept), float(slope)

    intercept, slope = solve(weights)

    # Two inexpensive robustification passes provide useful protection from
    # impulsive observations without materially increasing latency.
    for _ in range(2):
        residual = values - (intercept + slope * time)
        scale = _robust_scale(residual, fallback=fallback_scale)
        huber_limit = max(1.60 * scale, 1e-10)
        huber_weights = np.minimum(
            1.0, huber_limit / np.maximum(np.abs(residual), 1e-12)
        )
        intercept, slope = solve(weights * huber_weights)

    residual = values - (intercept + slope * time)
    scale = _robust_scale(residual, fallback=fallback_scale)
    return float(intercept), float(slope), float(scale)


def _model_consensus(levels, slopes, scales, global_noise):
    """
    Blend fast, medium, and slow endpoint models.

    Fast-model weight rises only when its slope agrees with the more stable
    models.  This retains rapid response for genuine moves but reduces the
    effect of short-lived noisy excursions.
    """
    levels = np.asarray(levels, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    scales = np.asarray(scales, dtype=float)
    valid = np.isfinite(levels)

    if not np.any(valid):
        return np.nan, 0.0, float(global_noise)

    levels = levels[valid]
    slopes = slopes[valid]
    scales = scales[valid]

    if levels.size == 1:
        return float(levels[0]), float(slopes[0]), float(scales[0])

    local_scale = max(float(np.median(scales)), 0.25 * global_noise, 1e-8)

    if levels.size == 2:
        slope_ref = max(0.20 * global_noise, 0.35 * np.sum(np.abs(slopes)), 1e-9)
        agreement = np.tanh((slopes[0] * slopes[1]) / (slope_ref * slope_ref))
        fast_weight = float(np.clip(0.46 + 0.18 * agreement, 0.28, 0.64))
        level = fast_weight * levels[0] + (1.0 - fast_weight) * levels[1]
        slope = fast_weight * slopes[0] + (1.0 - fast_weight) * slopes[1]
        return float(level), float(slope), local_scale

    fast_level, medium_level, slow_level = levels[:3]
    fast_slope, medium_slope, slow_slope = slopes[:3]

    slope_ref = max(
        0.16 * global_noise,
        0.22 * (abs(fast_slope) + abs(medium_slope) + abs(slow_slope)),
        1e-9,
    )

    fast_agreement = 0.5 * (
        np.tanh((fast_slope * medium_slope) / (slope_ref * slope_ref))
        + np.tanh((fast_slope * slow_slope) / (slope_ref * slope_ref))
    )
    stable_agreement = np.tanh(
        (medium_slope * slow_slope) / (slope_ref * slope_ref)
    )

    # Fast horizon receives up to 0.66 weight only on broad agreement.
    fast_weight = float(np.clip(0.43 + 0.23 * fast_agreement, 0.24, 0.66))

    # In stable-model conflict, favor the slow fit.  In agreement, preserve
    # more medium-scale detail.
    medium_share = float(np.clip(0.57 + 0.15 * stable_agreement, 0.38, 0.72))
    remaining = 1.0 - fast_weight
    medium_weight = remaining * medium_share
    slow_weight = remaining - medium_weight

    level = (
        fast_weight * fast_level
        + medium_weight * medium_level
        + slow_weight * slow_level
    )
    slope = (
        fast_weight * fast_slope
        + medium_weight * medium_slope
        + slow_weight * slow_slope
    )

    return float(level), float(slope), local_scale


def adaptive_filter(x, window_size=20):
    """
    Causal multi-scale robust filter with asymmetric reversal confirmation.

    Args:
        x: One-dimensional real-valued input signal.
        window_size: Required history before the first emitted endpoint value.

    Returns:
        Filtered endpoint estimates, length len(x) - window_size + 1.
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )

    n = len(x)
    output_length = n - window_size + 1

    if not np.any(np.isfinite(x)):
        return np.full(output_length, np.nan)

    if window_size == 1:
        return x.copy()

    finite_x = x[np.isfinite(x)]
    if finite_x.size > 2:
        global_noise = _robust_scale(np.diff(finite_x), fallback=1e-5) / np.sqrt(2.0)
    else:
        global_noise = _robust_scale(finite_x, fallback=1e-5)
    global_noise = max(float(global_noise), 1e-6)

    # Three horizons: rapid local tracking, intermediate trend, stable context.
    fast_size = max(4, min(window_size, int(np.ceil(0.38 * window_size))))
    medium_size = max(fast_size + 1, min(window_size, int(np.ceil(0.62 * window_size))))
    slow_size = window_size

    fast_decay = max(1.4, 0.31 * fast_size)
    medium_decay = max(1.8, 0.38 * medium_size)
    slow_decay = max(2.3, 0.48 * slow_size)

    y = np.empty(output_length, dtype=float)

    previous = np.nan
    velocity = 0.0
    accepted_direction = 0.0
    pending_direction = 0.0
    reversal_count = 0

    for output_index, endpoint in enumerate(range(window_size - 1, n)):
        fast_window = x[endpoint - fast_size + 1:endpoint + 1]
        medium_window = x[endpoint - medium_size + 1:endpoint + 1]
        slow_window = x[endpoint - slow_size + 1:endpoint + 1]

        fast = _robust_endpoint_fit(fast_window, fast_decay, global_noise)
        medium = _robust_endpoint_fit(medium_window, medium_decay, global_noise)
        slow = _robust_endpoint_fit(slow_window, slow_decay, global_noise)

        candidate, candidate_slope, local_scale = _model_consensus(
            [fast[0], medium[0], slow[0]],
            [fast[1], medium[1], slow[1]],
            [fast[2], medium[2], slow[2]],
            global_noise,
        )

        if not np.isfinite(candidate):
            candidate = previous if np.isfinite(previous) else 0.0
            candidate_slope = 0.0

        # Small causal prediction compensates regression attenuation.  It is
        # deliberately lower than the previous 0.12 correction to avoid
        # amplifying slope noise near turning points.
        candidate += 0.09 * candidate_slope

        if not np.isfinite(previous):
            filtered = candidate
            velocity = candidate_slope
            y[output_index] = filtered
            previous = filtered
            continue

        predicted = previous + velocity
        innovation = candidate - predicted

        slope_strength = abs(candidate_slope) / max(local_scale, global_noise, 1e-8)
        confidence = float(np.clip(slope_strength / 1.5, 0.0, 1.0))

        # Adaptive alpha-beta tracker: high gain during strong real movement,
        # lower gain in ambiguous/noisy regions.
        alpha = 0.42 + 0.40 * confidence
        beta = 0.05 + 0.14 * confidence
        tracked = predicted + alpha * innovation
        next_velocity = velocity + beta * innovation

        delta = tracked - previous
        delta_sign = np.sign(delta)

        forward_deadband = max(0.075 * local_scale, 0.014 * global_noise)
        reversal_deadband = max(0.19 * local_scale, 0.032 * global_noise)

        if accepted_direction == 0.0:
            if abs(delta) >= forward_deadband:
                filtered = tracked
                accepted_direction = delta_sign
            else:
                filtered = previous
        elif delta_sign == 0.0 or delta_sign == accepted_direction:
            # Trend continuation has a low threshold, avoiding excess lag.
            if abs(delta) >= forward_deadband:
                filtered = tracked
            else:
                filtered = previous
            pending_direction = 0.0
            reversal_count = 0
        else:
            # Opposing movement needs a larger directional deadband and two
            # consecutive confirmations.  Strong slope opposition can satisfy
            # the same confirmation path, preserving genuine sharp turns.
            opposing_slope = np.sign(candidate_slope) == delta_sign
            enough_evidence = (
                abs(delta) >= reversal_deadband
                and (opposing_slope or abs(candidate_slope) >= 0.10 * local_scale)
            )

            if enough_evidence:
                if pending_direction == delta_sign:
                    reversal_count += 1
                else:
                    pending_direction = delta_sign
                    reversal_count = 1
            else:
                pending_direction = 0.0
                reversal_count = 0

            if reversal_count >= 2:
                filtered = tracked
                accepted_direction = delta_sign
                pending_direction = 0.0
                reversal_count = 0
            else:
                filtered = previous
                # Decay stale velocity while held against a suspected reversal.
                next_velocity *= 0.55

        actual_delta = filtered - previous
        if abs(actual_delta) >= forward_deadband:
            velocity = 0.70 * next_velocity + 0.30 * actual_delta
        else:
            velocity = 0.82 * next_velocity

        y[output_index] = filtered
        previous = filtered

    return y


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """Backward-compatible trend-preserving public entry point."""
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected signal-processing algorithm.

    Args:
        input_signal: Input time series data.
        window_size: Sliding history size.
        algorithm_type: "basic" or "enhanced".

    Returns:
        Filtered signal with length len(input_signal) - window_size + 1.
    """
    if algorithm_type == "enhanced":
        return enhanced_filter_with_trend_preservation(input_signal, window_size)
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
