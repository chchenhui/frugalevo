"""Run one reproducible EE experiment with conservative per-request cost admission.

Uses the project's price table and UTF-8 byte counts (an upper bound for the
text tokenizer), reserves all concurrent requests, disables automatic retries,
and retains reservations for requests whose usage cannot be established.
The unchanged evaluator and initial program are supplied by the caller.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skydiscover.llm.openai import OpenAILLM
from skydiscover.llm.pricing import load_pricing, provider_from_api_base
from skydiscover.search.frugalevo.controller import FrugalEvoController
from scripts.ee_campaign_budget import reserve_run, settle_run


class RequestBudget:
    def __init__(self, limit, path):
        self.limit = limit
        self.path = Path(path)
        self.charged = 0.0
        self.reserved = 0.0
        self.stopped = False
        self.records = []
        self.deadline = float("inf")
        self.stop_file = self.path.parent / "STOP"

    async def generate(self, original, model, system, messages, **kwargs):
        if time.monotonic() >= self.deadline or self.stop_file.exists():
            self.stopped = True
            raise RuntimeError("Experiment stopped before buying another API call")
        price = load_pricing().resolve(model.model, provider_from_api_base(model.api_base))
        if price is None or not price.is_priced:
            self.stopped = True
            raise RuntimeError(f"Cannot enforce USD budget for unpriced model {model.model}")
        max_tokens = int(kwargs.get("max_tokens", model.max_tokens) or 8192)
        payload = json.dumps([system, messages], ensure_ascii=False)
        input_bound = len(payload.encode("utf-8")) + 4096
        max_input_rate = max(price.rate_for(field) or 0 for field in (
            "input_tokens", "cache_read_tokens", "cache_write_tokens", "cache_write_1h_tokens"
        ))
        bound = (input_bound * max_input_rate + max_tokens * price.output_per_million) / 1e6
        # No await between testing and reserving: atomic within the asyncio loop.
        if self.stopped or self.charged + self.reserved + bound > self.limit:
            self.stopped = True
            raise RuntimeError("Conservative request budget exhausted before API call")
        self.reserved += bound
        billed = bound
        usage_known = False
        started_at = time.time()
        try:
            response = await original(model, system, messages, **{**kwargs, "retries": 0})
            if response.usage is not None:
                billed = price.cost(response.usage)
                usage_known = True
            return response
        finally:
            self.reserved -= bound
            self.charged += billed
            self.records.append({"model": model.model, "upper_bound": bound,
                                 "charged_or_reserved": billed, "cumulative": self.charged,
                                 "usage_known": usage_known, "started_at": started_at,
                                 "finished_at": time.time()})
            self.path.write_text(json.dumps({"limit": self.limit, "charged": self.charged,
                                            "reserved": self.reserved, "calls": self.records}, indent=2))


def resume_cost_before(checkpoint, *, initial, evaluator, seed):
    checkpoint = Path(checkpoint)
    parent_run = checkpoint.parent.parent
    completion = parent_run / "completed.json"
    if not checkpoint.is_dir() or not completion.exists():
        raise ValueError("Resume requires a saved checkpoint from a normally completed segment")
    previous = json.loads((parent_run / "experiment.json").read_text())
    expected = {"initial": initial, "evaluator": evaluator, "seed": seed}
    if any(previous.get(k) != v for k, v in expected.items()):
        raise ValueError("Resume must keep the original task, evaluator and seed")
    cost = float(previous.get("cost_before", 0.0)) + float(json.loads(completion.read_text())["charged"])
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Invalid prior cost; manual accounting audit required")
    return cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial", default="benchmarks/math/third_autocorr_ineq/initial_program.py")
    parser.add_argument("--evaluator", default="benchmarks/math/third_autocorr_ineq/evaluator/evaluator.py")
    parser.add_argument("--plateau", type=int, default=12, help="Stalled-round limit; 0 disables")
    parser.add_argument("--plateau-min-spend", type=float, default=0.3)
    parser.add_argument("--wall-seconds", type=float, default=0, help="Run time limit; 0 disables (default)")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--iterations", type=int, default=200, help="Iteration limit; 0 runs until budget stop")
    parser.add_argument("--sequential", type=int, choices=(0, 1), default=1)
    parser.add_argument("--parallel-after-valid-miss", type=int, choices=(0, 1), default=0)
    parser.add_argument("--cooldown", type=int, default=2)
    parser.add_argument("--search-incumbent", type=int, choices=(0, 1), default=1)
    parser.add_argument("--strategies", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--implementations",
        type=int,
        help="implementations per strategy (defaults to the value in the base config)",
    )
    parser.add_argument(
        "--initial-strategies", type=int, choices=(1, 2, 3),
        help="strategy breadth for the first guide portfolio (defaults to --strategies)",
    )
    parser.add_argument("--strategy-when", choices=("stalled", "reuse_on_improvement", "always"))
    parser.add_argument(
        "--strategy-reference", type=int, choices=(0, 1), default=1,
        help="request and evaluate executable code bundled with each guide portfolio",
    )
    parser.add_argument("--models-config", help="YAML containing an llm mapping; task and evaluator stay unchanged")
    parser.add_argument(
        "--prompt-config",
        help="YAML whose prompt.system_message replaces the base task prompt",
    )
    parser.add_argument("--lazy-guidance", type=int, choices=(0, 1), default=1)
    parser.add_argument("--spend-limit", type=float, default=1.0)
    parser.add_argument("--campaign-ledger")
    parser.add_argument("--opening-guide-hedge", type=int, choices=(0, 1), default=0)
    parser.add_argument("--post-guide-cheap", type=int, choices=(0, 1), default=0)
    parser.add_argument("--opening-failure-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--opening-candidates", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--independent-opening", type=int, choices=(0, 1), default=0)
    parser.add_argument("--escalation-patience", type=int,
                        help="Fresh-run stagnation threshold for adaptive guide reasoning")
    parser.add_argument("--escalation-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--checkpoint", help="Continue a completed segment; prior cost remains on the curve")
    args = parser.parse_args()
    if args.iterations < 0 or args.plateau < 0 or args.wall_seconds < 0:
        parser.error("iterations, plateau and wall-seconds must be non-negative")
    if args.implementations is not None and args.implementations < 1:
        parser.error("implementations must be positive")
    iteration_limit = args.iterations or sys.maxsize
    if args.escalation_patience is not None:
        if args.escalation_patience < 1 or args.checkpoint:
            parser.error("escalation-patience must be positive and currently requires a fresh run")
    output = Path(args.output)
    args.cost_before = 0.0
    if args.checkpoint:
        try:
            args.cost_before = resume_cost_before(args.checkpoint, initial=args.initial,
                                                 evaluator=args.evaluator, seed=args.seed)
        except ValueError as exc:
            parser.error(str(exc))
    if not 0 < args.spend_limit <= 2:
        parser.error("spend-limit must be > 0 and <= 2")
    if args.cost_before + args.spend_limit > 2.0 + 1e-12:
        parser.error("Prior cost plus this segment's reservation exceeds the USD 2 trajectory cap")
    if args.campaign_ledger:
        reserve_run(args.campaign_ledger, output, args.spend_limit)
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for path in ("skydiscover/config.py", "skydiscover/search/frugalevo/controller.py",
                 "skydiscover/search/frugalevo/strategy.py", "skydiscover/search/frugalevo/database.py",
                 "scripts/ee_budget_experiment.py", "scripts/ee_campaign_budget.py",
                 "skydiscover/llm/models.yaml", "skydiscover/search/utils/budget_curve.py",
                 args.initial, args.evaluator):
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / source
        source = source.resolve()
        try:
            snapshot_path = source.relative_to(ROOT)
        except ValueError:
            snapshot_path = Path("external_inputs") / source.name
        destination = snapshots / snapshot_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config = yaml.safe_load(Path(args.config).read_text())
    config["max_iterations"] = iteration_limit
    if args.models_config:
        config["llm"] = yaml.safe_load(Path(args.models_config).read_text())["llm"]
    if args.prompt_config:
        prompt_source = yaml.safe_load(Path(args.prompt_config).read_text())
        config.setdefault("prompt", {})["system_message"] = prompt_source["prompt"]["system_message"]
    config["llm"]["retries"] = 0
    config["llm"].setdefault("max_tokens", 8192)
    for pool in ("models", "guide_models", "evaluator_models"):
        for model in config["llm"].get(pool, []):
            model["retries"] = 0
            model.setdefault("max_tokens", config["llm"]["max_tokens"])
    config["checkpoint_interval"] = 5
    database = config["search"]["database"]
    database.setdefault("budget_usd", 1.0)
    database.update(random_seed=args.seed,
                    sequential_implementation_feedback=bool(args.sequential),
                    sequential_parallel_after_valid_miss=bool(args.parallel_after_valid_miss),
                    search_incumbent_enabled=bool(args.search_incumbent),
                    search_incumbent_elite_ratio=0.35,
                    search_incumbent_stagnation_patience=2, search_incumbent_cooldown=args.cooldown,
                    strategy_reference_candidate=bool(args.strategy_reference),
                    opening_failure_retries=args.opening_failure_retries,
                    opening_cheap_candidates=args.opening_candidates,
                    opening_independent_sampling=bool(args.independent_opening),
                    opening_direct_guide_hedge=bool(args.opening_guide_hedge),
                    opening_post_guide_cheap_probe=bool(args.post_guide_cheap),
                    strategies_per_guide_call=args.strategies,
                    initial_strategies_per_guide_call=(
                        args.initial_strategies
                        if args.initial_strategies is not None
                        else args.strategies
                    ),
                    lazy_strategy_guidance=bool(args.lazy_guidance))
    if args.implementations is not None:
        database["implementations_per_strategy"] = args.implementations
    if args.strategy_when:
        database["strategy_when"] = args.strategy_when
    if args.escalation_patience is not None:
        database.update(strategy_escalation_patience=args.escalation_patience,
                        strategy_escalation_interval=args.escalation_patience,
                        strategy_escalation_reasoning_effort=args.escalation_effort)
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "experiment.json").write_text(json.dumps(vars(args), indent=2))
    (output / "pid.txt").write_text(str(os.getpid()))
    budget = RequestBudget(args.spend_limit, output / "request_budget.json")
    original_generate = OpenAILLM.generate
    original_plan = FrugalEvoController._plan_iteration
    started = time.monotonic()
    if args.wall_seconds > 0:
        budget.deadline = started + args.wall_seconds

    async def generate(model, system_message, messages, **kwargs):
        return await budget.generate(original_generate, model, system_message, messages, **kwargs)

    def plan(controller, iteration):
        reason = None
        if budget.stopped or budget.stop_file.exists():
            reason = "request admission stop (budget, deadline, or STOP file)"
        elif args.plateau > 0 and controller._stagnation >= args.plateau and controller.curve.spent() >= args.plateau_min_spend:
            reason = f"{controller._stagnation} consecutive non-improving rounds"
        elif args.wall_seconds > 0 and time.monotonic() - started >= args.wall_seconds:
            reason = "experiment wall-clock limit"
        if reason:
            controller.early_stopping_triggered = True
            (output / "stop_reason.json").write_text(json.dumps({"iteration": iteration, "reason": reason}))
            logging.getLogger(__name__).info("Experiment early stop: %s", reason)
            return None
        return original_plan(controller, iteration)

    OpenAILLM.generate = generate
    FrugalEvoController._plan_iteration = plan
    from skydiscover.cli import main as cli_main
    sys.argv = ["skydiscover-run", args.initial, args.evaluator, "--config", str(config_path),
                "--search", "frugalevo", "--iterations", str(iteration_limit), "--output", str(output)]
    if args.checkpoint:
        sys.argv.extend(["--checkpoint", args.checkpoint])
    try:
        exit_code = cli_main()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
        exit_code = 0
    (output / "completed.json").write_text(json.dumps({"exit_code": exit_code, "charged": budget.charged}))
    if args.campaign_ledger:
        settle_run(args.campaign_ledger, output, budget.charged)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
