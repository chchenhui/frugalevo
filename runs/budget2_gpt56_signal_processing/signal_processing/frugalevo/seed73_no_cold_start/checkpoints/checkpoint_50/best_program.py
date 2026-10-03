"""
Batch robust adaptive denoising for volatile non-stationary time series.

Returned samples are aligned to input indices window_size - 1 through end.
"""
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt, savgol_filter
except Exception:
    butter = None
    sosfiltfilt = None
    savgol_filter = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


def _finite_signal(x):
    z = np.asarray(x, dtype=float).reshape(-1).copy()
    if z.size == 0:
        return z
    good = np.isfinite(z)
    if not np.all(good):
        if np.any(good):
            ind = np.arange(z.size)
            z[~good] = np.interp(ind[~good], ind[good], z[good])
        else:
            z.fill(0.0)
    return z


def _odd_at_most(value, n):
    w = min(int(value), int(n))
    return w if w % 2 else w - 1


def _savgol_safe(x, preferred=11):
    x = _finite_signal(x)
    n = len(x)
    if savgol_filter is None or n < 5:
        return x.copy()
    w = _odd_at_most(preferred, n)
    if w < 5:
        return x.copy()
    try:
        return savgol_filter(x, w, min(3, w - 2), mode="interp")
    except Exception:
        return x.copy()


def _median3(x):
    """Apply two finite bilateral passes with temporal and range-domain weighting."""
    base = _finite_signal(x)
    if len(base) < 3:
        return base.copy()

    fallback = base.copy()
    sigma_d = max(_mad_sigma(np.diff(base)), 1e-12)
    if not np.isfinite(sigma_d) or sigma_d <= 1e-12:
        return fallback

    offsets = np.arange(-5, 6, dtype=int)
    temporal = np.exp(-0.5 * (offsets / 2.4) ** 2)
    current = base.copy()

    for pass_index in range(2):
        if pass_index == 1:
            sigma_d = max(_mad_sigma(np.diff(current)), 1e-12)
            if not np.isfinite(sigma_d) or sigma_d <= 1e-12:
                return current

        range_scale = 2.2 * sigma_d
        result = np.empty_like(current)

        for i in range(len(current)):
            indices = np.clip(i + offsets, 0, len(current) - 1)
            values = current[indices]
            range_weights = np.exp(
                -0.5 * ((values - current[i]) / range_scale) ** 2
            )
            weights = temporal * range_weights
            total = float(np.sum(weights))
            result[i] = (
                float(np.sum(weights * values) / total)
                if np.isfinite(total) and total > 1e-12
                else current[i]
            )

        current = np.where(np.isfinite(result), result, current)

    return current


def _mad_sigma(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) / 0.67448975)


def _spectral_cutoff(x):
    n = len(x)
    if n < 8:
        return 0.18
    power = np.abs(np.fft.rfft(x)) ** 2
    bins = len(power)
    tail = power[max(1, int(0.70 * bins)):]
    floor = float(np.median(tail)) if len(tail) else 0.0
    useful = np.maximum(power - floor, 0.0)
    useful[0] = 0.0
    total = float(np.sum(useful))
    if not np.isfinite(total) or total <= 1e-14:
        return 0.18
    k = int(np.searchsorted(np.cumsum(useful), 0.94 * total))
    k = min(max(k, 1), bins - 1)
    return float(np.clip(2.0 * k / n, 0.07, 0.28))


def _hysteretic_extrema(y, excursion):
    """Alternating extrema confirmed only by a scale-supported reversal."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        return np.arange(n, dtype=int)

    excursion = max(float(excursion), 1e-12)
    anchors = [0]

    direction = 1 if y[1] >= y[0] else -1
    candidate = 1

    for i in range(2, n):
        if direction > 0:
            if y[i] >= y[candidate]:
                candidate = i
            elif y[candidate] - y[i] >= excursion:
                if candidate > anchors[-1]:
                    anchors.append(candidate)
                direction = -1
                candidate = i
        else:
            if y[i] <= y[candidate]:
                candidate = i
            elif y[i] - y[candidate] >= excursion:
                if candidate > anchors[-1]:
                    anchors.append(candidate)
                direction = 1
                candidate = i

    if anchors[-1] != n - 1:
        anchors.append(n - 1)

    return np.asarray(anchors, dtype=int)


def _adaptive_full_record_filter(x):
    """Denoise with change-point-segmented local SSA and overlap consensus."""
    x = _finite_signal(x)
    n = len(x)
    if n < 2:
        return x.copy()

    robust = _median3(x)

    # Isolate pronounced level changes before local low-rank reconstruction.
    # This prevents a global Hankel basis from ringing across discontinuities.
    if n >= 14:
        d = np.diff(robust)
        d_scale = max(_mad_sigma(d), 1e-12)
        level_scale = max(_mad_sigma(robust), 1e-12)
        threshold = max(4.5 * d_scale, 0.12 * level_scale)
        candidates = np.flatnonzero(np.abs(d) > threshold)
        boundaries = []
        for boundary in candidates:
            left = robust[max(0, boundary - 2):boundary + 1]
            right = robust[boundary + 1:min(n, boundary + 4)]
            if len(left) and len(right):
                separation = float(np.median(right) - np.median(left))
                if abs(separation) >= 0.5 * threshold:
                    boundaries.append((abs(float(d[boundary])), boundary + 1))

        # Retain only the strongest boundaries and enforce usable segments.
        boundaries = [
            b for _, b in sorted(boundaries, reverse=True)[:5]
        ]
        boundaries = sorted(set(boundaries))
        cleaned = []
        for boundary in boundaries:
            if boundary >= 14 and n - boundary >= 14:
                if not cleaned or boundary - cleaned[-1] >= 14:
                    cleaned.append(boundary)
        boundaries = cleaned

        edges = [0] + boundaries + [n]

        def local_ssa(segment):
            """Reconstruct one segment with compact low-rank Hankel averaging."""
            length = len(segment)
            if length < 8:
                return _savgol_safe(segment, 9)
            width = min(40, max(8, length // 3), length)
            columns = length - width + 1
            try:
                matrix = np.lib.stride_tricks.sliding_window_view(
                    segment, width
                ).T.astype(float, copy=True)
                u, singular, vt = np.linalg.svd(
                    matrix, full_matrices=False
                )
                energy = singular * singular
                tail = energy[max(1, (2 * len(energy)) // 3):]
                floor = float(np.median(tail)) if len(tail) else 0.0
                useful = np.maximum(energy - floor, 0.0)
                total = float(np.sum(useful))
                if not np.isfinite(total) or total <= 1e-12:
                    return _savgol_safe(segment, 9)

                cumulative = np.cumsum(useful)
                rank = min(4, len(singular))
                for candidate in range(2, min(7, len(singular)) + 1):
                    if cumulative[candidate - 1] >= 0.94 * total:
                        rank = candidate
                        break

                reconstructed = (
                    u[:, :rank] * singular[:rank][None, :]
                ) @ vt[:rank, :]
                output = np.zeros(length, dtype=float)
                counts = np.zeros(length, dtype=float)
                for row in range(width):
                    end = min(columns, length - row)
                    if end:
                        output[row:row + end] += reconstructed[row, :end]
                        counts[row:row + end] += 1.0
                output /= np.maximum(counts, 1.0)
                return np.where(np.isfinite(output), output, segment)
            except Exception:
                return _savgol_safe(segment, 9)

        pieces = [
            local_ssa(robust[start:end])
            for start, end in zip(edges[:-1], edges[1:])
        ]
        preliminary = pieces[0].copy()
        for index in range(1, len(pieces)):
            boundary = edges[index]
            left = preliminary
            right = pieces[index]
            overlap = min(4, boundary, n - boundary)
            if overlap:
                # Raised-cosine consensus keeps the transition responsive
                # while avoiding an abrupt reconstruction-dependent jump.
                weights = 0.5 - 0.5 * np.cos(
                    np.pi * (np.arange(overlap) + 1) / (overlap + 1)
                )
                left_start = boundary - overlap
                right_end = overlap
                left[left_start:boundary] = (
                    (1.0 - weights) * left[left_start:boundary]
                    + weights * right[:right_end]
                )
                preliminary = np.concatenate(
                    (left, right[overlap:])
                )
            else:
                preliminary = np.concatenate((left, right))

        # Continue through the incumbent extrema projection below, preserving
        # its hysteresis and finite-output safeguards.
    else:
        preliminary = None
    if n < 8:
        preliminary = _savgol_safe(robust, 11)
    else:
        width = min(40, max(8, n // 3), n)
        columns = n - width + 1
        try:
            # H[row, column] = robust[row + column]; this orientation makes
            # anti-diagonal averaging recover the original sample indices.
            trajectory = np.lib.stride_tricks.sliding_window_view(
                robust, width
            ).T.astype(float, copy=True)

            u, singular, vt = np.linalg.svd(
                trajectory, full_matrices=False
            )
            energy = singular * singular
            tail = energy[max(1, (2 * len(energy)) // 3):]
            floor = float(np.median(tail)) if len(tail) else 0.0
            useful = np.maximum(energy - floor, 0.0)
            total = float(np.sum(useful))
            if not np.isfinite(total) or total <= 1e-12:
                raise ValueError("degenerate trajectory energy")

            # Stability-selected paired-SSA rank.  Each candidate must retain
            # sufficient de-biased trajectory energy.  The score combines
            # robust slope/detail mismatch, explicit slope-change complexity,
            # and hysteretic false-reversal complexity.
            cumulative = np.cumsum(useful)
            ranks = range(2, min(8, len(singular)) + 1)
            best_rank = None
            best_cost = np.inf
            minimum_energy = 0.92 * total

            for candidate_rank in ranks:
                if cumulative[candidate_rank - 1] < minimum_energy:
                    continue

                candidate_matrix = (
                    u[:, :candidate_rank] *
                    singular[:candidate_rank][None, :]
                ) @ vt[:candidate_rank, :]

                candidate_series = np.zeros(n, dtype=float)
                candidate_counts = np.zeros(n, dtype=float)
                for row in range(width):
                    end = min(columns, n - row)
                    if end:
                        candidate_series[row:row + end] += (
                            candidate_matrix[row, :end]
                        )
                        candidate_counts[row:row + end] += 1.0
                candidate_series /= np.maximum(candidate_counts, 1.0)

                if not np.all(np.isfinite(candidate_series)):
                    continue

                candidate_scale = max(_mad_sigma(candidate_series), 1e-12)
                if n > 2:
                    slope_error = (
                        np.diff(candidate_series, n=2) -
                        np.diff(robust, n=2)
                    )
                    detail_error = (
                        np.diff(candidate_series) - np.diff(robust)
                    )
                    detail_cost = _mad_sigma(detail_error)
                    slope_cost = _mad_sigma(slope_error)
                elif n > 1:
                    detail_cost = _mad_sigma(
                        np.diff(candidate_series) - np.diff(robust)
                    )
                    slope_cost = 0.0
                else:
                    detail_cost = 0.0
                    slope_cost = 0.0

                residual_sigma = _mad_sigma(robust - candidate_series)
                reversal_scale = max(1.2 * residual_sigma, 1e-12)
                reversal_cost = (
                    max(len(_hysteretic_extrema(
                        candidate_series, reversal_scale
                    )) - 2, 0) / max(n, 1)
                )

                # The small rank regularizer resolves nearly equal scores in
                # favor of the lower-complexity reconstruction without
                # materially sacrificing tracking accuracy.
                cost = (
                    detail_cost / candidate_scale
                    + 0.35 * slope_cost / candidate_scale
                    + 2.5 * reversal_cost
                    + 0.002 * candidate_rank
                )

                if np.isfinite(cost) and cost < best_cost:
                    best_cost = cost
                    best_rank = candidate_rank

            if best_rank is None:
                best_rank = min(max(2, 1), 8, len(singular))
            low_rank = (
                u[:, :best_rank] * singular[:best_rank][None, :]
            ) @ vt[:best_rank, :]

            preliminary = np.zeros(n, dtype=float)
            counts = np.zeros(n, dtype=float)
            for row in range(width):
                end = min(columns, n - row)
                if end:
                    preliminary[row:row + end] += low_rank[row, :end]
                    counts[row:row + end] += 1.0
            preliminary /= np.maximum(counts, 1.0)
            preliminary = np.where(
                np.isfinite(preliminary), preliminary, robust
            )
        except Exception:
            preliminary = _savgol_safe(robust, 11)

    # Select between the aligned low-rank reconstruction and a responsive
    # polynomial reconstruction independently in bounded overlapping blocks.
    if n >= 12:
        def consensus_blocks(ssa, reference):
            """Blend blockwise SSA/polynomial choices using residual and reversal costs."""
            ssa = np.asarray(ssa, dtype=float).copy()
            reference = np.asarray(reference, dtype=float)
            polynomial = _savgol_safe(reference, 9)
            if len(polynomial) != n:
                polynomial = reference.copy()

            block_length = int(np.clip(n // 6, 24, 48))
            overlap = min(6, max(2, block_length // 4))
            step = max(1, block_length - overlap)
            starts = list(range(0, max(1, n - block_length + 1), step))
            last = max(0, n - block_length)
            if not starts or starts[-1] != last:
                starts.append(last)
            # A final endpoint block is more useful than silently truncating
            # the record when the bounded block budget is reached.
            starts = starts[:7]
            if last not in starts:
                starts[-1] = last

            output = np.zeros(n, dtype=float)
            weights = np.zeros(n, dtype=float)
            difference_scale = max(_mad_sigma(np.diff(reference)), 1e-12)

            for block_index, start in enumerate(starts):
                stop = min(n, start + block_length)
                if stop <= start:
                    continue
                sl = slice(start, stop)
                target = reference[sl]
                candidates = (ssa[sl], polynomial[sl])
                costs = []

                for candidate in candidates:
                    candidate = np.asarray(candidate, dtype=float)
                    residual_scale = max(
                        _mad_sigma(candidate - target), 1e-12
                    )
                    reversal_scale = max(
                        1.25 * residual_scale, 0.35 * difference_scale
                    )
                    reversals = max(
                        len(_hysteretic_extrema(
                            candidate, reversal_scale
                        )) - 2,
                        0,
                    )
                    endpoint = (
                        abs(candidate[0] - target[0]) +
                        abs(candidate[-1] - target[-1])
                    ) / (2.0 * difference_scale)
                    costs.append(
                        residual_scale / difference_scale
                        + 0.10 * reversals
                        + 0.20 * endpoint
                    )

                # Prefer the lower-reversal model unless its residual is
                # materially worse; this preserves oscillatory detail while
                # suppressing block-local noise reversals.
                if costs[0] <= 1.15 * costs[1]:
                    selected = candidates[0]
                else:
                    selected = candidates[1]
                selected = np.asarray(selected, dtype=float).copy()

                length = stop - start
                taper = np.ones(length, dtype=float)
                left = min(overlap, length)
                right = min(overlap, length)
                if block_index:
                    taper[:left] = 0.5 - 0.5 * np.cos(
                        np.pi * (np.arange(left) + 1) / (left + 1)
                    )
                if block_index < len(starts) - 1:
                    taper[-right:] *= 0.5 + 0.5 * np.cos(
                        np.pi * (np.arange(right) + 1) / (right + 1)
                    )
                output[sl] += taper * selected
                weights[sl] += taper

            consensus = output / np.maximum(weights, 1e-12)
            return np.where(weights > 0, consensus, ssa)

        preliminary = consensus_blocks(preliminary, robust)

    if n < 5:
        return preliminary

    """Blend local polynomial scales using curvature and residual-noise gates."""
    residual = x - preliminary
    residual_sigma = _mad_sigma(residual)
    signal_scale = max(_mad_sigma(preliminary), 1e-12)

    # The three zero-phase polynomial estimates act as overlapping local
    # consensus models.  Savitzky-Golay interpolation provides triangular
    # overlap-like confidence through its progressively different supports.
    candidates = []
    for width in (7, 11, 17):
        if savgol_filter is not None and n >= width:
            try:
                estimate = savgol_filter(
                    preliminary, width, 2, mode="interp"
                )
            except Exception:
                estimate = preliminary.copy()
        else:
            estimate = preliminary.copy()
        candidates.append(np.asarray(estimate, dtype=float))

    short, medium, long = candidates
    curvature = np.abs(np.gradient(np.gradient(medium)))
    curvature_scale = max(_mad_sigma(curvature), 1e-12)
    curvature_z = curvature / curvature_scale

    # Estimate local residual noise with a compact overlapping confidence
    # envelope; this avoids allowing a noisy flat region to select the short
    # polynomial merely because its pointwise curvature is nonzero.
    local_noise = np.convolve(
        np.abs(residual), np.ones(9, dtype=float) / 9.0, mode="same"
    )
    noise_gate = local_noise > max(1.15 * residual_sigma, 0.01 * signal_scale)

    short_weight = np.clip((curvature_z - 0.5) / 1.0, 0.0, 1.0)
    long_weight = (
        np.clip((0.5 - curvature_z) / 0.5, 0.0, 1.0) *
        noise_gate.astype(float)
    )
    # Keep the weights convex and favor the responsive medium estimate in the
    # transition band, limiting phase-like displacement of narrow peaks.
    long_weight *= 0.65
    short_weight = np.minimum(short_weight, 1.0 - long_weight)
    medium_weight = 1.0 - short_weight - long_weight

    y = (
        short_weight * short +
        medium_weight * medium +
        long_weight * long
    )

    # Reject only reversals whose excursion is locally below the residual
    # noise floor.  Genuine extrema remain untouched because their local
    # excursion exceeds the adaptive hysteresis threshold.
    local_sigma = np.maximum(
        np.convolve(
            np.abs(residual),
            np.ones(7, dtype=float) / 7.0,
            mode="same",
        ),
        1.15 * residual_sigma,
    )
    extrema = _hysteretic_extrema(y, float(np.median(local_sigma)))
    if len(extrema) >= 3:
        weak = []
        for index in extrema[1:-1]:
            left = abs(y[index] - y[extrema[list(extrema).index(index) - 1]])
            right = abs(y[index] - y[extrema[list(extrema).index(index) + 1]])
            if min(left, right) < 1.25 * local_sigma[index]:
                weak.append(index)
        if weak:
            y = y.copy()
            y[weak] = medium[weak]

    return np.where(np.isfinite(y), y, preliminary)


def adaptive_filter(x, window_size=20):
    """Apply Hankel low-rank denoising followed by hysteretic trend projection."""
    x = _finite_signal(x)
    if len(x) < window_size:
        raise ValueError(
            f"Input signal length ({len(x)}) must be >= window_size ({window_size})"
        )
    return _adaptive_full_record_filter(x)[window_size - 1:]


def enhanced_filter_with_trend_preservation(x, window_size=20):
    return adaptive_filter(x, window_size)


def process_signal(input_signal, window_size=20, algorithm_type="enhanced"):
    return enhanced_filter_with_trend_preservation(input_signal, window_size)


def generate_test_signal(length=1000, noise_level=0.3, seed=42):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 10, length)
    clean_signal = (
        2.0 * np.sin(2 * np.pi * 0.5 * t)
        + 1.5 * np.sin(2 * np.pi * 2.0 * t)
        + 0.5 * np.sin(2 * np.pi * 5.0 * t)
        + 0.8 * np.exp(-t / 5.0) * np.sin(2 * np.pi * 1.5 * t)
    )
    clean_signal += 0.1 * t * np.sin(0.2 * t)
    clean_signal += np.cumsum(rng.randn(length) * 0.05)
    noisy_signal = clean_signal + rng.normal(0, noise_level, length)
    return noisy_signal, clean_signal


def run_signal_processing(noisy_signal=None, signal_length=1000,
                          noise_level=0.3, window_size=20):
    if noisy_signal is None:
        noisy_signal, clean_signal = generate_test_signal(
            signal_length, noise_level
        )
    else:
        noisy_signal = _finite_signal(noisy_signal)
        clean_signal = None

    filtered_signal = process_signal(noisy_signal, window_size, "enhanced")

    if clean_signal is None:
        return {
            "filtered_signal": filtered_signal,
            "clean_signal": None,
            "noisy_signal": None,
            "correlation": 0,
            "noise_reduction": 0,
            "signal_length": len(filtered_signal),
        }

    aligned_clean = clean_signal[window_size - 1:]
    aligned_noisy = noisy_signal[window_size - 1:]
    m = min(len(filtered_signal), len(aligned_clean))
    filtered_signal = filtered_signal[:m]
    aligned_clean = aligned_clean[:m]
    aligned_noisy = aligned_noisy[:m]

    correlation = (
        float(np.corrcoef(filtered_signal, aligned_clean)[0, 1])
        if m > 1 and np.std(filtered_signal) > 0 and np.std(aligned_clean) > 0
        else 0.0
    )
    noise_before = np.var(aligned_noisy - aligned_clean)
    noise_after = np.var(filtered_signal - aligned_clean)
    noise_reduction = (
        float((noise_before - noise_after) / noise_before)
        if noise_before > 0 else 0.0
    )

    return {
        "filtered_signal": filtered_signal,
        "clean_signal": aligned_clean,
        "noisy_signal": aligned_noisy,
        "correlation": correlation,
        "noise_reduction": noise_reduction,
        "signal_length": m,
    }


if __name__ == "__main__":
    results = run_signal_processing()
    print("Signal processing completed!")
    print(f"Correlation with clean signal: {results['correlation']:.3f}")
    print(f"Noise reduction: {results['noise_reduction']:.3f}")
    print(f"Processed signal length: {results['signal_length']}")