import fcntl
import runpy
import sys

sys.path.insert(0, '/home/chenhui/efficient-evolve/benchmarks/ADRS/prism/evaluator')
_original = runpy.run_path('/home/chenhui/efficient-evolve/benchmarks/ADRS/prism/evaluator/evaluator.py')

def evaluate(program_path):
    with open('/tmp/frugalevo-adrs-1029.lock', "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _original["evaluate"](program_path)

def evaluate_stage1(program_path):
    return evaluate(program_path)

def evaluate_stage2(program_path):
    return evaluate(program_path)

def evaluate_stage3(program_path):
    return evaluate(program_path)


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
