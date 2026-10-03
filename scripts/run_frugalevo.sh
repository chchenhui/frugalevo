#!/usr/bin/env bash
# Run FrugalEvo using the repository's uv environment.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# You can use your own initial program and evaluator
initial_program="$repo_root/benchmarks/math/circle_packing/initial_program.py"
evaluator="$repo_root/benchmarks/math/circle_packing/evaluator.py"
# You can use your own config file
config="$repo_root/configs/budget2_gpt56_circle_packing/circle_packing/frugalevo.yaml"
extra_args=()

usage() {
  cat <<'EOF'
Usage: bash scripts/run_frugalevo.sh [options] [skydiscover-run options]

Run FrugalEvo on circle packing by default.
Before running, install dependencies with `uv sync --extra math` and configure
the API credentials required by the models in your YAML configuration.

Options:
  --initial-program FILE  Initial program (default: circle packing)
  --evaluator PATH        Evaluator file or benchmark directory (default: circle packing)
  --config, -c FILE       YAML configuration (default: budget2_gpt56_circle_packing)
  --help, -h             Show this help without starting a run

Other CLI options are forwarded, e.g. --iterations 10 --output runs/my_run.
The search algorithm is always frugalevo. Relative paths are resolved from your
current working directory. Edit the YAML to customize models and budget.
EOF
}

while (( $# )); do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --initial-program|--evaluator|--config|-c)
      if (( $# < 2 )) || [[ -z "$2" || "$2" == -* ]]; then
        printf 'Error: %s requires a path.\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --initial-program) initial_program="$2" ;;
        --evaluator) evaluator="$2" ;;
        --config|-c) config="$2" ;;
      esac
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

for file in "$initial_program" "$config"; do
  if [[ ! -f "$file" ]]; then
    printf 'Error: file not found: %s\n' "$file" >&2
    exit 2
  fi
done
if [[ ! -f "$evaluator" && ! -d "$evaluator" ]]; then
  printf 'Error: evaluator not found: %s\n' "$evaluator" >&2
  exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'Error: uv is required. Install it from https://docs.astral.sh/uv/.\n' >&2
  exit 127
fi

exec uv run --project "$repo_root" skydiscover-run \
  "$initial_program" "$evaluator" --config "$config" \
  "${extra_args[@]}" --search frugalevo
