# EVOLVE-BLOCK-START
"""
Robust low-lag adaptive filter for volatile non-stationary time series.

The enhanced filter combines a bounded constant-velocity tracker with robust
causal endpoint regression and a pending-reversal confirmation state.
"""
import numpy as np


def adaptive_filter(x, window_size=20):
    """
    Efficient sliding-window moving-average baseline.

    Returns an array aligned with x[window_size - 1:].
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


def enhanced_filter_with_trend_preservation(x, window_size=20):
    """
    Robust predictive tracker with causal endpoint regression.

    The output is aligned with x[window_size - 1:].  A robust local-linear
    endpoint fit provides low-lag trend information, while an adaptive
    constant-velocity tracker supplies stable state estimation.  Opposing
    movements are held in a lightweight pending-reversal state and are released
    only after a second residual-noise-supported, slope-consistent observation.
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
    output_length = n - window_size + 1

    # Robust causal endpoint regression.  MAD clipping suppresses isolated
    # spikes before estimating the local level and local trend.
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
    medians = np.median(windows, axis=1)
    raw_mad = np.median(np.abs(windows - medians[:, None]), axis=1)
    raw_scale = np.maximum(1.4826 * raw_mad, 1e-8)

    clipped_windows = np.clip(
        windows,
        (medians - 3.25 * raw_scale)[:, None],
        (medians + 3.25 * raw_scale)[:, None],
    )

    time = np.arange(window_size, dtype=float) - (window_size - 1)
    design = np.column_stack((np.ones(window_size), time))

    # Recency decay of 2.6 retains trend responsiveness while keeping the
    # regression sufficiently stable at noisy extrema.
    regression_weights = np.exp(np.linspace(-2.6, 0.0, window_size))
    regression_map = np.linalg.solve(
        design.T @ (regression_weights[:, None] * design),
        design.T * regression_weights,
    )

    coefficients = clipped_windows @ regression_map.T
    endpoint_levels = coefficients[:, 0]
    endpoint_slopes = coefficients[:, 1]

    fitted_windows = (
        endpoint_levels[:, None] + endpoint_slopes[:, None] * time[None, :]
    )
    fit_noise = 1.4826 * np.median(
        np.abs(clipped_windows - fitted_windows), axis=1
    )
    fit_noise = np.maximum(np.maximum(fit_noise, 0.10 * raw_scale), 1e-8)

    initial_diff = np.diff(x[:window_size])
    if initial_diff.size:
        diff_center = np.median(initial_diff)
        diff_scale = 1.4826 * np.median(np.abs(initial_diff - diff_center))
    else:
        diff_scale = 0.0

    # Adaptive Kalman-style tracker parameters.
    measurement_var = max(0.52 * diff_scale * diff_scale, 1e-8)
    innovation_var = measurement_var
    process_var = max(0.032 * measurement_var, 1e-10)

    level = float(x[0])
    velocity = float(endpoint_slopes[0])
    p00 = 4.0 * measurement_var
    p01 = 0.0
    p11 = measurement_var

    consensus_level = float(x[0])
    consensus_velocity = float(endpoint_slopes[0])
    output = np.empty(output_length, dtype=float)

    # Warm-up reference is used only before a complete regression window is
    # available; no warm-up samples are emitted.
    warm_decay = np.exp(-2.3 / max(window_size - 1, 1))
    warm_reference = float(x[0])

    previous_step = 0.0
    pending_direction = 0.0
    pending_strength = 0.0
    pending_age = 0

    for index, sample in enumerate(x):
        # Bounded constant-velocity prediction/update.
        predicted_level = level + velocity
        q = process_var

        pp00 = p00 + 2.0 * p01 + p11 + 0.25 * q
        pp01 = p01 + p11 + 0.50 * q
        pp11 = p11 + q

        innovation = sample - predicted_level
        prediction_var = max(pp00 + measurement_var, 1e-12)
        innovation_limit = 3.15 * np.sqrt(prediction_var)
        bounded_innovation = np.clip(
            innovation, -innovation_limit, innovation_limit
        )

        gain_level = pp00 / prediction_var
        gain_velocity = pp01 / prediction_var

        level = predicted_level + gain_level * bounded_innovation
        velocity = velocity + gain_velocity * bounded_innovation

        p00 = max(pp00 - gain_level * pp00, 1e-12)
        p01 = pp01 - gain_level * pp01
        p11 = max(pp11 - gain_velocity * pp01, 1e-12)

        innovation_var = (
            0.958 * innovation_var
            + 0.042 * bounded_innovation * bounded_innovation
        )
        measurement_var = max(
            0.992 * measurement_var + 0.008 * innovation_var,
            1e-8,
        )
        process_var = max(
            0.994 * process_var + 0.006 * innovation_var * 0.035,
            1e-10,
        )

        warm_reference = (
            warm_decay * warm_reference + (1.0 - warm_decay) * sample
        )

        if index >= window_size - 1:
            local_index = index - window_size + 1
            local_level = endpoint_levels[local_index]
            local_slope = endpoint_slopes[local_index]
            local_noise = fit_noise[local_index]
            trend_confidence = abs(local_slope) / (
                abs(local_slope) + 0.65 * local_noise + 1e-12
            )
        else:
            local_level = warm_reference
            local_slope = velocity
            local_noise = np.sqrt(max(innovation_var, 1e-10))
            trend_confidence = 0.0

        scale = max(
            np.sqrt(max(innovation_var, 1e-10)),
            local_noise,
            1e-8,
        )

        # Local endpoint information is used most strongly when it agrees with
        # the stable tracker and has a residual-supported slope.
        disagreement = abs(level - local_level)
        agreement = 1.0 / (
            1.0 + (disagreement / (1.85 * scale + 1e-12)) ** 2
        )
        local_weight = (0.12 + 0.26 * trend_confidence) * agreement
        target = (1.0 - local_weight) * level + local_weight * local_level

        previous_consensus = consensus_level
        predicted_consensus = consensus_level + consensus_velocity
        target_residual = target - predicted_consensus

        # Prediction-centered update limits phase delay relative to a static
        # smoother, while avoiding overreaction to isolated innovations.
        alpha = 0.67 + 0.15 * min(
            1.0, abs(target_residual) / (2.6 * scale + 1e-12)
        )
        candidate = predicted_consensus + alpha * target_residual
        candidate_step = candidate - previous_consensus
        candidate_direction = np.sign(candidate_step)

        prior_direction = np.sign(previous_step)
        opposing = (
            prior_direction != 0.0
            and candidate_direction != 0.0
            and candidate_direction == -prior_direction
        )

        reversal_band = 0.17 * scale + 0.06 * abs(consensus_velocity)
        slope_supported = (
            candidate_direction != 0.0
            and np.sign(local_slope) == candidate_direction
            and abs(local_slope) >= 0.095 * scale
        )

        # A large, slope-supported turn may pass immediately.  Smaller
        # countertrend moves are pre-committed but must be confirmed by the
        # next sample, preventing alternating noise from becoming output turns.
        decisive_turn = (
            slope_supported
            and abs(candidate_step) >= 0.78 * scale
        )
        confirmation_threshold = (
            0.30 * scale + 0.07 * abs(consensus_velocity)
        )

        accepted = True
        if opposing and not decisive_turn:
            pending_eligible = (
                abs(candidate_step) >= reversal_band
                and candidate_direction != 0.0
            )

            confirmed = (
                pending_age == 1
                and candidate_direction == pending_direction
                and slope_supported
                and abs(candidate_step) >= confirmation_threshold
                and pending_strength >= reversal_band
            )

            if confirmed:
                consensus_level = candidate
                pending_direction = 0.0
                pending_strength = 0.0
                pending_age = 0
            else:
                consensus_level = previous_consensus
                candidate_step = 0.0
                accepted = False

                if pending_eligible:
                    pending_direction = candidate_direction
                    pending_strength = abs(candidate - predicted_consensus)
                    pending_age = 1
                else:
                    pending_direction = 0.0
                    pending_strength = 0.0
                    pending_age = 0
        else:
            consensus_level = candidate
            pending_direction = 0.0
            pending_strength = 0.0
            pending_age = 0

        if candidate_step != 0.0:
            previous_step = candidate_step

        # During a held countertrend candidate, decay the previous velocity
        # instead of allowing stale momentum to manufacture a later overshoot.
        if not accepted:
            consensus_velocity *= 0.62

        slope_weight = 0.20 + 0.48 * trend_confidence
        velocity_target = (
            (1.0 - slope_weight) * candidate_step
            + slope_weight * local_slope
        )

        acceleration_limit = (
            0.070 * scale
            + 0.14 * abs(local_slope)
            + 0.018 * abs(consensus_velocity)
        )
        velocity_change = np.clip(
            velocity_target - consensus_velocity,
            -acceleration_limit,
            acceleration_limit,
        )
        consensus_velocity += (
            0.44 + 0.14 * trend_confidence
        ) * velocity_change

        velocity_deadband = 0.020 * scale
        if abs(consensus_velocity) < velocity_deadband:
            consensus_velocity = 0.0

        if index >= window_size - 1:
            output[index - window_size + 1] = consensus_level

    return output


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    """
    Apply the selected filtering algorithm.

    Args:
        input_signal: Input 1D time series.
        window_size: Sliding-window length.
        algorithm_type: "enhanced" or "basic".

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