"""EPLB evaluate() entry point that survives being copied into an adapter.

The baseline launchers build ShinkaEvolve's CLI adapter by keeping everything
above the first `if __name__ == "__main__":` line of the evaluator file and
appending their own argparse entry point. `scripts/eplb_serial_evaluator.py`
cannot be used directly for that: its own entry point is spelled with single
quotes, so the split would keep it, and it resolves the repository root from
`__file__`, which points at the copy once the adapter is written elsewhere.

This module keeps only a delegating evaluate() above the marker and locates the
real serial evaluator through EPLB_SERIAL_EVALUATOR, falling back to a walk up
from wherever the copy lives, so the adapter behaves exactly like the original.
"""
import importlib.util
import os
from pathlib import Path


def _load_serial_evaluator():
    relative = Path("scripts") / "eplb_serial_evaluator.py"
    override = os.environ.get("EPLB_SERIAL_EVALUATOR")
    candidates = [Path(override)] if override else []
    # An adapter copy can land anywhere under the run root, so search upward too.
    candidates += [parent / relative for parent in Path(__file__).resolve().parents]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "eplb_serial_evaluator_delegate", candidate
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError(
        f"Could not locate {relative}; set EPLB_SERIAL_EVALUATOR to its absolute path"
    )


_SERIAL_EVALUATOR = _load_serial_evaluator()


def evaluate(program_path, **kwargs):
    """Run one EPLB evaluation in the shared serialized worker subprocess."""
    return _SERIAL_EVALUATOR.evaluate(program_path, **kwargs)


if __name__ == "__main__":
    import argparse
    import json
    import math
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    result = evaluate(args.program_path)

    combined_score = float(result.get("combined_score", 0.0) or 0.0)
    explicit_correct = result.get("correct")
    if isinstance(explicit_correct, dict):
        explicit_correct = explicit_correct.get("correct")
    validity = result.get("validity")
    runs_successfully = result.get("runs_successfully")
    if explicit_correct is not None:
        is_correct = bool(explicit_correct)
    elif validity is not None:
        is_correct = bool(validity) and math.isfinite(combined_score)
    elif runs_successfully is not None:
        is_correct = bool(runs_successfully) and math.isfinite(combined_score)
    else:
        is_correct = not result.get("error") and math.isfinite(combined_score)

    metrics = {
        "combined_score": combined_score,
        "public": result,
        "private": {},
        "text_feedback": str(result.get("text_feedback", result.get("error", ""))),
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as output:
        json.dump(metrics, output)
    with open(os.path.join(args.results_dir, "correct.json"), "w") as output:
        json.dump({"correct": is_correct}, output)
