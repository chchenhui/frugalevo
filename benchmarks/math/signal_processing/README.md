# Real-Time Adaptive Signal Processing

Evolve a real-time adaptive filtering algorithm for non-stationary time series data. The algorithm must filter noise while preserving signal dynamics and minimizing computational latency.

## Problem

**Input**: Univariate time series with non-linear dynamics, non-stationary statistics, and rapidly changing spectral characteristics.

**Constraints**: Causal processing (finite sliding window), fixed latency, real-time capability.

**Multi-objective function**:
```
J(theta) = 0.3*S + 0.2*L_recent + 0.2*L_avg + 0.3*R
```
- **S**: Slope change penalty (directional reversals in filtered signal)
- **L_recent**: Mean absolute tracking error over the most recent processing window
- **L_avg**: Average tracking error
- **R**: False reversal penalty (noise-induced trend changes)

The evaluator tests on 5 synthetic signals: sinusoidal, multi-frequency, non-stationary, step changes, and random walk.

## Run

```bash
# From repo root
uv run skydiscover-run \
  benchmarks/math/signal_processing/initial_program.py \
  benchmarks/math/signal_processing/evaluator.py \
  -c benchmarks/math/signal_processing/config.yaml \
  -s [your_algorithm] \
  -i 100
```

## Scoring

- **combined_score**: Composite J(theta) metric (higher is better)
- Also reports: slope changes, correlation, lag error, noise reduction, processing time

## Evaluator hardening (2026-09-02)

Two minimal safeguards were added to `evaluator/evaluator.py`:

1. **Exact output-length contract**: for an input of length `N` and a window of
   size `W`, `filtered_signal` must contain exactly `N - W + 1` samples. This is
   enforced in both Stage 1 and Stage 2. A candidate with an empty, truncated,
   or otherwise incorrect-length output now receives a zero score instead of
   being evaluated on only the returned prefix.
2. **Window-based recent lag**: `L_recent` is now the mean absolute tracking
   error over the latest `min(W, output_length)` aligned samples. Previously it
   compared only the final filtered sample with the corresponding input sample,
   which allowed a candidate to obtain zero recent-lag error by copying that one
   input point without tracking the surrounding recent signal.

These changes preserve the existing scoring weights and five-signal test set;
they only tighten output validity and make the recent-response metric represent
a complete recent window.

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Seed: basic moving average / weighted exponential filters |
| `evaluator.py` | Multi-objective evaluation across 5 synthetic test signals |
| `config.yaml` | LLM and evaluator settings |
| `requirements.txt` | Python dependencies |
