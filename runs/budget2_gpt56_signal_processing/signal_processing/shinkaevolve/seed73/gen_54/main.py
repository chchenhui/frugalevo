# EVOLVE-BLOCK-START
"""
Hysteretic endpoint trend filter for volatile non-stationary signals.

The enhanced filter uses:
1. Causal multi-scale endpoint linear regressions.
2. Innovation-based robust sample limiting for isolated spikes.
3. A hysteretic confidence state machine with flat, transition, and trend modes.
4. A predictive alpha-beta endpoint tracker with persistence-aware reversal gating.

Output[i] is aligned with x[i + window_size - 1].
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient trailing moving-average baseline.

    Args:
        x: One-dimensional input signal.
        window_size: Trailing window size.

    Returns:
        Moving-average output aligned to trailing window endpoints.
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

    cumulative = np.concatenate(([0.0], np.cumsum(x, dtype=float)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def _endpoint_linear_coefficients(span, emphasis):
    """
    Return FIR coefficients for a weighted linear fit evaluated at the endpoint.

    The returned filters estimate the level and slope at the newest sample in a
    causal span.  Increasing emphasis gives newer observations more influence.
    """
    t = np.arange(span, dtype=float) - float(span - 1)
    weights = np.exp(emphasis * t / float(max(span - 1, 1)))

    s0 = np.sum(weights)
    s1 = np.sum(weights * t)
    s2 = np.sum(weights * t * t)
    determinant = max(s0 * s2 - s1 * s1, 1e-12)

    level = weights * (s2 - s1 * t) / determinant
    slope = weights * (s0 * t - s1) / determinant
    return level, slope


def _robust_causal_samples(x, window_size):
    """
    Suppress only extreme isolated innovations before endpoint fitting.

    The limiter is intentionally loose: ordinary dynamics and genuine turns are
    retained, while large impulsive observations cannot dominate a short local
    regression and create an artificial reversal.
    """
    n = len(x)
    result = np.empty(n, dtype=float)

    initial_diff = np.diff(x[:max(2, min(window_size, n))])
    if initial_diff.size:
        median_diff = np.median(initial_diff)
        scale = 1.4826 * np.median(np.abs(initial_diff - median_diff))
    else:
        scale = 0.0
    scale = max(float(scale), 1e-6)

    level = float(x[0])
    velocity = 0.0
    result[0] = level

    for i in range(1, n):
        predicted = level + velocity
        innovation = float(x[i] - predicted)

        limit = 4.25 * scale + 1e-10
        bounded = float(np.clip(innovation, -limit, limit))
        observation = predicted + bounded
        result[i] = observation

        step = observation - level
        level += 0.46 * (observation - level)
        velocity = 0.78 * velocity + 0.22 * step

        clipped_for_scale = min(abs(innovation), 4.0 * scale)
        scale = np.sqrt(0.975 * scale * scale + 0.025 * clipped_for_scale * clipped_for_scale)
        scale = max(scale, 1e-6)

    return result


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Causal hysteretic multi-scale endpoint filter.

    Local regressions at short, medium, and long causal spans estimate the
    newest signal value.  Their directional agreement drives a discrete
    confidence mode rather than continuously changing gain every sample.
    Persistent coherent motion receives a high tracking gain; noisy or
    conflicting windows use lower gain and reversal confirmation.

    Args:
        x: Input one-dimensional signal.
        window_size: Output trailing-window alignment and maximum context span.

    Returns:
        Filtered signal of length len(x) - window_size + 1.
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
    if window_size == 1:
        return x.copy()

    n = len(x)
    n_out = n - window_size + 1

    # The spans are deliberately nested.  The short fit is responsive, while
    # the long fit supplies a stable trend reference and reversal validator.
    short_span = max(3, min(window_size, window_size // 3 + 1))
    medium_span = max(short_span + 1, min(window_size, (2 * window_size) // 3))
    long_span = window_size

    robust_x = _robust_causal_samples(x, window_size)

    spans_and_emphasis = (
        (short_span, 4.8),
        (medium_span, 2.8),
        (long_span, 1.35),
    )
    levels = []
    slopes = []

    for span, emphasis in spans_and_emphasis:
        level_coeff, slope_coeff = _endpoint_linear_coefficients(span, emphasis)
        levels.append(np.convolve(robust_x, level_coeff[::-1], mode="valid"))
        slopes.append(np.convolve(robust_x, slope_coeff[::-1], mode="valid"))

    # Align every regression result to output endpoints x[window_size - 1:].
    short_offset = long_span - short_span
    medium_offset = long_span - medium_span
    short_level = levels[0][short_offset:short_offset + n_out]
    medium_level = levels[1][medium_offset:medium_offset + n_out]
    long_level = levels[2]
    short_slope = slopes[0][short_offset:short_offset + n_out]
    medium_slope = slopes[1][medium_offset:medium_offset + n_out]
    long_slope = slopes[2]

    # A rolling absolute first-difference statistic tracks changing noise and
    # avoids mistaking higher variance periods for persistent trend evidence.
    differences = np.abs(np.diff(robust_x))
    noise_width = max(2, min(window_size - 1, medium_span))
    noise_kernel = np.ones(noise_width, dtype=float) / float(noise_width)
    rolling_difference = np.convolve(differences, noise_kernel, mode="valid")

    noise = np.empty(n_out, dtype=float)
    endpoint_start = window_size - 1
    for k in range(n_out):
        diff_index = endpoint_start + k - noise_width
        diff_index = max(0, min(diff_index, len(rolling_difference) - 1))
        noise[k] = max(rolling_difference[diff_index] / 1.128379167, 1e-6)

    output = np.empty(n_out, dtype=float)

    # State-machine modes: 0=flat/noisy, 1=transition, 2=persistent trend.
    mode = 0
    high_count = 0
    low_count = 0

    filtered = float(long_level[0])
    velocity = 0.0
    direction = 0
    reverse_count = 0

    for k in range(n_out):
        scale = noise[k]

        ss = short_slope[k]
        ms = medium_slope[k]
        ls = long_slope[k]

        sign_short = 1 if ss > 0.0 else (-1 if ss < 0.0 else 0)
        sign_medium = 1 if ms > 0.0 else (-1 if ms < 0.0 else 0)
        sign_long = 1 if ls > 0.0 else (-1 if ls < 0.0 else 0)

        coherent = (
            sign_short != 0
            and sign_short == sign_medium
            and sign_medium == sign_long
        )

        # Trend evidence uses displacement over each fit span rather than raw
        # slope alone, making the criterion comparable across window sizes.
        coherent_motion = min(
            abs(ss) * short_span,
            abs(ms) * medium_span,
            abs(ls) * long_span,
        )
        trend_ratio = coherent_motion / (scale + 1e-12)

        spread = max(short_level[k], medium_level[k], long_level[k]) - min(
            short_level[k], medium_level[k], long_level[k]
        )
        agreement = 1.0 / (1.0 + spread / (2.4 * scale + 1e-12))
        confidence = trend_ratio * agreement if coherent else 0.0

        # Hysteresis prevents rapid gain oscillation when confidence is near a
        # decision boundary.
        if confidence > 0.72:
            high_count += 1
        else:
            high_count = 0

        if confidence < 0.30:
            low_count += 1
        else:
            low_count = 0

        if mode < 2 and high_count >= 2:
            mode = 2
            low_count = 0
        elif mode == 2 and low_count >= 3:
            mode = 1
            high_count = 0
        elif mode == 0 and confidence > 0.36:
            mode = 1
        elif mode == 1 and confidence < 0.16:
            mode = 0

        # Median-like selection during disagreement is robust to a single
        # endpoint fit corrupted by a local burst.  Persistent trends bias
        # toward the short endpoint fit for low phase delay.
        ordered_levels = np.sort(np.array([short_level[k], medium_level[k], long_level[k]]))
        robust_center = ordered_levels[1]

        if coherent and mode == 2:
            local_target = (
                0.54 * short_level[k]
                + 0.30 * medium_level[k]
                + 0.16 * long_level[k]
            )
        elif mode == 1:
            local_target = (
                0.28 * short_level[k]
                + 0.42 * medium_level[k]
                + 0.30 * long_level[k]
            )
        else:
            local_target = 0.72 * robust_center + 0.28 * long_level[k]

        # Quantized gains are more stable than sample-by-sample gain tuning.
        if mode == 2:
            alpha, beta = 0.70, 0.105
        elif mode == 1:
            alpha, beta = 0.55, 0.062
        else:
            alpha, beta = 0.45, 0.030

        predicted = filtered + velocity
        residual = local_target - predicted

        candidate_step = alpha * residual + velocity
        candidate_direction = (
            1 if candidate_step > 0.0 else (-1 if candidate_step < 0.0 else 0)
        )

        deadband = 0.040 * scale + 0.10 * abs(velocity)
        opposing = (
            direction != 0
            and candidate_direction != 0
            and candidate_direction != direction
        )

        # Counter-direction movement must be supported by multi-scale fits or
        # have sufficiently large amplitude. This avoids suppressing genuine
        # sharp turns while removing weak noise-driven reversals.
        supports_turn = (
            coherent
            and sign_short == candidate_direction
            and confidence > 0.42
        )
        strong_turn = abs(candidate_step) > (2.25 * deadband + 0.22 * scale)

        if opposing:
            if supports_turn or strong_turn:
                reverse_count += 1
            else:
                reverse_count = 0

            if reverse_count < 2 and not strong_turn:
                candidate_step = 0.0
                residual *= 0.35
            else:
                direction = candidate_direction
                reverse_count = 0
        elif abs(candidate_step) <= deadband:
            candidate_step = 0.0
            reverse_count = 0
        else:
            reverse_count = 0
            if candidate_direction != 0:
                direction = candidate_direction

        filtered += candidate_step
        velocity = (1.0 - beta) * velocity + beta * residual

        # Prevent accumulated sub-threshold velocity from causing delayed
        # alternating output movement in quiet sections.
        if mode == 0 and abs(velocity) < 0.018 * scale:
            velocity = 0.0

        output[k] = filtered

    return output


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply either the basic moving average or enhanced hysteretic filter.

    Args:
        input_signal: Input time series.
        window_size: Sliding window length.
        algorithm_type: "basic" or "enhanced".

    Returns:
        Filtered output signal.
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