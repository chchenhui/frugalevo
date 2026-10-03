"""
Thin wrapper around ShinkaEvolve (https://github.com/ShinkaiEvolve).

Delegates entirely to ShinkaEvolve's public API so upstream updates are
picked up automatically.
"""

import asyncio
import json
import logging
import os
import threading
from typing import Any, Dict, Optional

from skydiscover.api import DiscoveryResult
from skydiscover.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# LLM usage and pricing bridge
# ------------------------------------------------------------------


def _write_usage_summary(tracker, output_dir: str, lock) -> None:
    """Atomically persist the usage seen so far for crash-safe partial runs."""
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "llm_usage.json")
    temporary_path = summary_path + ".tmp"
    with lock:
        with open(temporary_path, "w", encoding="utf-8") as summary_file:
            json.dump(tracker.to_dict(), summary_file, indent=2)
            summary_file.write("\n")
        os.replace(temporary_path, summary_path)


def _merge_shinka_request_policy(
    kwargs: Dict[str, Any], request_extra_body: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Apply SkyDiscover's OpenRouter policy to a Shinka Responses request."""
    if not request_extra_body:
        return kwargs

    merged = dict(kwargs)
    policy = dict(request_extra_body)
    configured_reasoning = policy.pop("reasoning", None)
    if isinstance(configured_reasoning, dict):
        reasoning = dict(merged.get("reasoning") or {})
        reasoning.update(configured_reasoning)
        merged["reasoning"] = reasoning

    if policy:
        extra_body = dict(merged.get("extra_body") or {})
        for key, value in policy.items():
            if isinstance(value, dict) and isinstance(extra_body.get(key), dict):
                extra_body[key] = {**extra_body[key], **value}
            else:
                extra_body[key] = value
        merged["extra_body"] = extra_body
    return merged


def _install_usage_bridge(
    output_dir: str,
    api_base: Optional[str] = None,
    request_extra_body: Optional[Dict[str, Any]] = None,
):
    """Make Shinka use SkyDiscover's ``models.yaml`` API token prices.

    Shinka's OpenAI provider reduces the raw SDK response to a ``QueryResult``
    and normally prices it with Shinka's bundled pricing snapshot. Replacing
    that narrow conversion function lets Shinka's own budget bookkeeping use
    the same prices as the other SkyDiscover backends, while preserving
    Shinka's expected return shape.

    Returns the tracker and a cleanup callback which restores the upstream
    function. The caller must invoke the callback even when evolution fails.
    """
    import importlib

    from shinka.embed import embedding as shinka_embedding
    from shinka.llm.providers import openai as shinka_openai

    shinka_query = importlib.import_module("shinka.llm.query")

    from skydiscover.llm.pricing import CostTracker, Usage, resolve_pricing

    tracker = CostTracker()
    summary_lock = threading.Lock()
    original_get_costs = shinka_openai.get_openai_costs
    original_get_embedding_cost = shinka_embedding._get_embedding_cost
    original_query_openai = shinka_query.query_openai
    original_query_openai_async = shinka_query.query_openai_async

    def query_openai_with_policy(*args, **kwargs):
        return original_query_openai(
            *args,
            **_merge_shinka_request_policy(kwargs, request_extra_body),
        )

    async def query_openai_async_with_policy(*args, **kwargs):
        return await original_query_openai_async(
            *args,
            **_merge_shinka_request_policy(kwargs, request_extra_body),
        )

    def persist_summary() -> None:
        try:
            _write_usage_summary(tracker, output_dir, summary_lock)
        except Exception:
            # Accounting must never turn a successful generation into a
            # failed Shinka iteration.
            logger.debug("Failed to persist Shinka LLM usage summary", exc_info=True)

    def get_costs_from_models_yaml(response, model):
        usage = tracker.record_response(model, response, api_base=api_base)
        pricing = resolve_pricing(model, api_base=api_base)
        if usage is None or pricing is None:
            # Keep Shinka usable for responses without usage and for models
            # the user's price table does not contain.
            return original_get_costs(response, model)

        input_usage = Usage(
            input_tokens=usage.input_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            cache_write_1h_tokens=usage.cache_write_1h_tokens,
        )
        output_usage = Usage(output_tokens=usage.output_tokens)
        input_cost = pricing.cost(input_usage) or 0.0
        output_cost = pricing.cost(output_usage) or 0.0

        # Write after every response so an interrupted experiment still has
        # a usable partial cost report.
        persist_summary()
        return {
            # Shinka expects total prompt tokens here, including cache hits.
            "input_tokens": usage.prompt_tokens,
            # Shinka stores reasoning tokens separately from visible output.
            "output_tokens": max(0, usage.output_tokens - usage.reasoning_tokens),
            "thinking_tokens": usage.reasoning_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "cost": input_cost + output_cost,
        }

    def get_embedding_cost_from_models_yaml(response, model_name):
        usage = tracker.record_response(model_name, response, api_base=api_base)
        pricing = resolve_pricing(model_name, api_base=api_base)
        if usage is None or pricing is None:
            return original_get_embedding_cost(response, model_name)

        cost = pricing.cost(usage)
        persist_summary()
        return cost or 0.0

    shinka_openai.get_openai_costs = get_costs_from_models_yaml
    shinka_embedding._get_embedding_cost = get_embedding_cost_from_models_yaml
    shinka_query.query_openai = query_openai_with_policy
    shinka_query.query_openai_async = query_openai_async_with_policy

    def close_bridge() -> None:
        # Avoid overwriting a newer wrapper if two bridges were accidentally
        # installed in the same process.
        if shinka_openai.get_openai_costs is get_costs_from_models_yaml:
            shinka_openai.get_openai_costs = original_get_costs
        if shinka_embedding._get_embedding_cost is get_embedding_cost_from_models_yaml:
            shinka_embedding._get_embedding_cost = original_get_embedding_cost
        if shinka_query.query_openai is query_openai_with_policy:
            shinka_query.query_openai = original_query_openai
        if shinka_query.query_openai_async is query_openai_async_with_policy:
            shinka_query.query_openai_async = original_query_openai_async
        persist_summary()
        for line in tracker.format_summary().splitlines():
            logger.info(line)
        logger.info(
            "Saved Shinka LLM usage priced from models.yaml to %s",
            os.path.join(output_dir, "llm_usage.json"),
        )

    return tracker, close_bridge


# ------------------------------------------------------------------
# Config mapping
# ------------------------------------------------------------------


def _runner_concurrency_kwargs(config: Config) -> Dict[str, int]:
    limit = config.evaluator.max_concurrent
    if limit is None:
        return {}
    if limit < 1:
        raise ValueError("evaluator.max_concurrent must be positive")
    return {"max_evaluation_jobs": limit, "max_proposal_jobs": limit}


def _map_config(config: Config, iterations: Optional[int], evaluator_path: str, output_dir: str):
    """Convert SkyDiscover Config to ShinkaEvolve's three config objects."""
    from dataclasses import fields as dc_fields

    from shinka.core import EvolutionConfig
    from shinka.database import DatabaseConfig as ShinkaDBC
    from shinka.launch import LocalJobConfig

    # Power-user escape hatch
    ext = getattr(config, "external_config", None)
    if ext is not None and isinstance(ext, dict):
        evo = ext.get("evo_config", EvolutionConfig())
        job = ext.get("job_config", LocalJobConfig(eval_program_path=evaluator_path))
        dbc = ext.get("db_config", ShinkaDBC())
        if iterations is not None:
            evo.num_generations = iterations
        return evo, job, dbc

    # Load tuned backend defaults
    from skydiscover.extras.external.defaults import load_defaults

    defaults = load_defaults("shinkaevolve_default.yaml")
    evo_defaults = defaults.get("evolution", {})
    db_defaults = defaults.get("database", {})

    # EvolutionConfig
    evo_kwargs: Dict[str, Any] = {
        "num_generations": iterations or config.max_iterations,
        "results_dir": output_dir,
        "job_type": "local",
        "language": getattr(config, "language", None) or "python",
        # Code embeddings are not needed for the standard Shinka baseline and
        # otherwise introduce a second provider/model into a GLM-only run.
        "embedding_model": None,
    }

    # ShinkaEvolve has its own proactive committed-cost guard.  Forward the
    # common SkyDiscover budget field so it actually stops rather than merely
    # drawing a reporting curve after the run.
    budget_usd = getattr(config.search.database, "budget_usd", None)
    if budget_usd is not None:
        evo_kwargs["max_api_costs"] = float(budget_usd)

    # Map LLM model names (from --model / -c config).  SkyDiscover removes
    # its provider prefix while parsing a config and keeps the provider in
    # ``api_base``; Shinka needs the prefix to select its OpenRouter backend
    # when the model is absent from Shinka's independent catalog.
    def shinka_model_name(model_config) -> str:
        name = model_config.name
        api_base = (getattr(model_config, "api_base", None) or "").lower()
        if "openrouter" in api_base and not name.startswith("openrouter/"):
            return f"openrouter/{name}"
        return name

    if config.llm.models:
        evo_kwargs["llm_models"] = [shinka_model_name(m) for m in config.llm.models]

    # Apply tuned defaults for evolution (patch types, LLM kwargs, meta, etc.)
    valid_evo_fields = {f.name for f in dc_fields(EvolutionConfig)}
    for key, value in evo_defaults.items():
        if key in valid_evo_fields and key not in evo_kwargs:
            evo_kwargs[key] = value

    # Keep Shinka's native sampling policy, but honor the request-level
    # reasoning effort used by the comparison run.  The upstream config uses
    # a list because it can sample across multiple reasoning efforts.
    reasoning_effort = getattr(config.llm, "reasoning_effort", None)
    if config.llm.models:
        reasoning_effort = config.llm.models[0].reasoning_effort or reasoning_effort
    if reasoning_effort:
        llm_kwargs = dict(evo_kwargs.get("llm_kwargs", {}))
        llm_kwargs["reasoning_efforts"] = [reasoning_effort]
        evo_kwargs["llm_kwargs"] = llm_kwargs

        meta_llm_kwargs = dict(evo_kwargs.get("meta_llm_kwargs", {}))
        meta_llm_kwargs["reasoning_efforts"] = [reasoning_effort]
        evo_kwargs["meta_llm_kwargs"] = meta_llm_kwargs

    # Cap parallelism to iteration count to avoid over-submission when
    # target generations is small (ShinkaEvolve auto-scales proposal jobs
    # from max_parallel_jobs; too many slots for few generations causes
    # directory collisions and stuck detection).
    iters = evo_kwargs["num_generations"]
    if "max_parallel_jobs" in evo_kwargs and iters < evo_kwargs["max_parallel_jobs"]:
        evo_kwargs["max_parallel_jobs"] = max(1, iters)

    # Meta model follows the main model
    if config.llm.models and evo_defaults.get("meta_rec_interval"):
        evo_kwargs["meta_llm_models"] = [shinka_model_name(config.llm.models[0])]

    # System prompt -> task_sys_msg
    sys_prompt = config.system_prompt_override
    if sys_prompt is None and hasattr(config, "context_builder"):
        sp = config.context_builder.system_message
        if sp and sp not in ("system_message", "evaluator_system_message"):
            sys_prompt = sp
    if sys_prompt:
        evo_kwargs["task_sys_msg"] = sys_prompt

    evo = EvolutionConfig(**evo_kwargs)

    # --- JobConfig ---
    job_kwargs: Dict[str, Any] = {"eval_program_path": evaluator_path}
    # Current ShinkaEvolve releases otherwise derive this value from the host
    # For example: CPU count (for example, 192 CPUs / 4 evaluation jobs = 48 BLAS threads
    # per child). Honour an explicit numeric-library cap from the launcher so
    # several baseline runs can coexist without severe oversubscription.
    for env_name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = os.environ.get(env_name)
        if value:
            try:
                job_kwargs["numeric_threads_per_job"] = max(1, int(value))
            except ValueError:
                logger.warning("Ignoring invalid %s=%r", env_name, value)
            break
    if hasattr(config, "evaluator") and config.evaluator.timeout:
        secs = int(config.evaluator.timeout)
        h, m, s = secs // 3600, (secs % 3600) // 60, secs % 60
        job_kwargs["time"] = f"{h:02d}:{m:02d}:{s:02d}"
    job = LocalJobConfig(**job_kwargs)

    # --- DatabaseConfig ---
    valid_db_fields = {f.name for f in dc_fields(ShinkaDBC)}
    dbc_kwargs = {k: v for k, v in db_defaults.items() if k in valid_db_fields}
    dbc = ShinkaDBC(**dbc_kwargs)

    return evo, job, dbc


# ------------------------------------------------------------------
# Initial score extraction
# ------------------------------------------------------------------


def _get_initial_score(all_programs: list) -> float:
    """Extract initial (generation 0) score from ShinkaEvolve programs list."""
    initial_score = 0.0
    for p in all_programs:
        gen = getattr(p, "generation", 0)
        if gen == 0:
            score = getattr(p, "combined_score", None)
            if score is not None:
                initial_score = max(initial_score, float(score))
    return initial_score


def _evaluation_is_correct(result: Dict[str, Any], combined_score: float) -> bool:
    """Interpret evaluator validity without rejecting legitimate zero scores."""
    import math

    explicit_correct = result.get("correct")
    if isinstance(explicit_correct, dict):
        explicit_correct = explicit_correct.get("correct")
    if explicit_correct is not None:
        is_correct = bool(explicit_correct)
    elif result.get("validity") is not None:
        is_correct = bool(result["validity"])
    else:
        is_correct = True

    # A finite score (including zero) does not imply successful evaluation.
    # Failure signals veto even an explicit correct=True from the evaluator.
    runs_successfully = result.get("runs_successfully")
    return (
        is_correct
        and math.isfinite(combined_score)
        and not bool(result.get("error"))
        and (runs_successfully is None or bool(runs_successfully))
    )


def _prepare_evaluator_source(evaluator_path: str, eval_str: str) -> str:
    """Adapt a SkyDiscover evaluator for Shinka's copied CLI evaluator.

    Shinka writes this source to ``results_dir/evaluate.py`` before running it.
    Keep ``__file__`` anchored to the original evaluator so task resources such
    as ``datasets/`` continue to resolve beside the source evaluator rather
    than beside the copied file.
    """
    main_marker = '\nif __name__ == "__main__":'
    if main_marker in eval_str:
        eval_str = eval_str.split(main_marker, 1)[0].rstrip()

    evaluator_path = os.path.abspath(evaluator_path)
    evaluator_dir = os.path.dirname(evaluator_path)
    eval_str = eval_str.replace(
        "sched_dir = os.path.dirname(os.path.abspath(__file__))",
        f"sched_dir = {evaluator_dir!r}",
    )
    return (
        "import sys\n"
        f"__file__ = {evaluator_path!r}\n"
        f"sys.path.insert(0, {evaluator_dir!r})\n\n"
        + eval_str
    )


# ------------------------------------------------------------------
# Program conversion
# ------------------------------------------------------------------


def _to_skydiscover_program(sp):
    """Convert a ShinkaEvolve Program to SkyDiscover's Program dataclass."""
    from skydiscover.search.base_database import Program

    metrics = dict(sp.public_metrics) if sp.public_metrics else {}
    metrics["combined_score"] = float(sp.combined_score or 0.0)
    metrics["correct"] = sp.correct

    return Program(
        id=sp.id,
        solution=sp.code,
        language=getattr(sp, "language", "python"),
        metrics=metrics,
        iteration_found=getattr(sp, "generation", 0),
        parent_id=getattr(sp, "parent_id", None),
        generation=getattr(sp, "generation", 0),
        timestamp=getattr(sp, "timestamp", 0.0),
    )


# ------------------------------------------------------------------
# SkyDiscover-format progress reporting
# ------------------------------------------------------------------


class _ShinkaProgressReporter:
    """Mirror Shinka programs into SkyDiscover logs and a budget curve."""

    def __init__(
        self,
        tracker,
        output_dir: str,
        monitor_callback=None,
        budget: Optional[float] = None,
    ):
        from skydiscover.search.utils.budget_curve import BudgetCurve

        self.tracker = tracker
        self.output_dir = output_dir
        self.monitor_callback = monitor_callback
        self.seen_ids: set[str] = set()
        self.curve = BudgetCurve(budget=budget, unit="usd", cost_fn=self._spent)

    def _spent(self) -> float:
        return float(self.tracker.total_cost_usd or 0.0)

    def record(self, program) -> bool:
        """Record one unseen Shinka program; return whether it was new."""
        if program.id in self.seen_ids:
            return False
        self.seen_ids.add(program.id)

        from skydiscover.utils.metrics import format_metrics

        sky_program = _to_skydiscover_program(program)
        iteration = int(getattr(program, "generation", 0) or 0)
        score = float(getattr(program, "combined_score", 0.0) or 0.0)
        parent_id = getattr(program, "parent_id", None) or "none"
        previous_best = self.curve.incumbent

        logger.info(
            "Iteration %d: Program %s (parent: %s) completed",
            iteration,
            program.id,
            parent_id,
        )
        logger.info("Metrics: %s", format_metrics(sky_program.metrics))

        if iteration == 0 and not self.curve.points:
            # The supplied initial program is a free baseline. Shinka may have
            # embedded it before the first DB poll, but that algorithmic cost
            # must not move the initial score away from cost zero.
            self.curve.restore(
                {
                    "unit": "usd",
                    "spent": 0.0,
                    "points": [
                        {
                            "cost": 0.0,
                            "score": score,
                            "incumbent": score,
                            "iteration": iteration,
                            "candidates": 1,
                        }
                    ],
                }
            )
        else:
            self.curve.observe(score, iteration=iteration, candidates=1)

        if score > previous_best:
            logger.info("🌟 New best solution found at iteration %d", iteration)
        logger.info("ShinkaEvolve progress: %s", self.curve.summary_line())
        self.curve.write(self.output_dir)

        if self.monitor_callback:
            try:
                self.monitor_callback(sky_program, iteration)
            except Exception:
                logger.debug("Shinka monitor callback error", exc_info=True)
        return True

    def record_many(self, programs) -> int:
        """Record programs in Shinka completion order."""
        ordered = sorted(
            programs,
            key=lambda program: (
                float(getattr(program, "timestamp", 0.0) or 0.0),
                int(getattr(program, "generation", 0) or 0),
                str(program.id),
            ),
        )
        return sum(1 for program in ordered if self.record(program))

    def write(self) -> None:
        self.curve.write(self.output_dir)


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------


async def run(
    program_path: str,
    evaluator_path: str,
    config_obj: Config,
    iterations: int,
    output_dir: str,
    monitor_callback=None,
    feedback_reader=None,
) -> DiscoveryResult:
    """Run evolution using the ShinkaEvolve package."""
    try:
        # ShinkaEvolve >= 0.0.1 unified the sync/async implementations under
        # this name.
        from shinka.core import ShinkaEvolveRunner as EvolutionRunner
    except ImportError:
        # Backward compatibility with the API used when this adapter was
        # originally added.
        from shinka.core import AsyncEvolutionRunner as EvolutionRunner

    # ShinkaEvolve does not currently expose an explicit RNG seed in its
    # public config. Seed the process-level RNGs it uses so SkyDiscover's
    # database.random_seed still controls sampling as far as the backend
    # permits. Async completion order and remote LLM sampling may still vary.
    random_seed = getattr(config_obj.search.database, "random_seed", None)
    if random_seed is not None:
        import random

        random.seed(random_seed)
        try:
            import numpy as np

            np.random.seed(random_seed)
        except ImportError:
            pass

    from skydiscover.config import bridge_provider_env

    bridge_provider_env(config_obj)

    evo_config, job_config, db_config = _map_config(
        config_obj,
        iterations,
        evaluator_path,
        output_dir,
    )

    # Human feedback: set initial system prompt on feedback reader for dashboard visibility
    if feedback_reader and evo_config.task_sys_msg:
        feedback_reader.set_current_prompt(evo_config.task_sys_msg)

    # ShinkaEvolve supports passing code as strings directly
    with open(program_path, "r") as f:
        init_str = f.read()
    with open(evaluator_path, "r") as f:
        eval_str = f.read()

    # ShinkaEvolve runs the evaluator as a CLI subprocess:
    #   python evaluate.py --program_path X --results_dir Y
    # and expects results written to results_dir/metrics.json.  SkyDiscover
    # evaluators instead expose evaluate(path) -> dict.  Shinka copies this
    # string into its results directory, therefore add the original evaluator
    # directory to sys.path for sibling imports (for example txn_simulator)
    # and replace, rather than retain, a legacy __main__ entry point.
    #
    # Many ADRS evaluators already contain ``if __name__ == "__main__":``
    # that imports a Docker-only wrapper.py.  Keeping that block makes a
    # Shinka subprocess fail before this adapter can write metrics.json.
    # Some evaluators use __file__ to locate sibling modules and data. Shinka
    # executes a copy from its result directory, so retain the source path.
    eval_str = (
        _prepare_evaluator_source(evaluator_path, eval_str)
        + """

if __name__ == "__main__":
    import argparse, json, os
    from skydiscover.extras.external.shinkaevolve_backend import _evaluation_is_correct
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    result = evaluate(args.program_path)

    combined_score = float(result.get("combined_score", 0.0) or 0.0)
    is_correct = _evaluation_is_correct(result, combined_score)

    # A finite score (including zero) does not imply successful evaluation.
    # Failure signals veto even an explicit correct=True from the evaluator.
    runs_successfully = result.get("runs_successfully")
    is_correct = (
        is_correct
        and math.isfinite(combined_score)
        and not bool(result.get("error"))
        and (runs_successfully is None or bool(runs_successfully))
    )

    metrics = {
        "combined_score": combined_score,
        "public": result,
        "private": {},
        "text_feedback": str(result.get("text_feedback", "")),
    }
    with open(os.path.join(args.results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(args.results_dir, "correct.json"), "w") as f:
        json.dump({"correct": is_correct}, f)
"""
    )

    concurrency_kwargs = _runner_concurrency_kwargs(config_obj)
    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        init_program_str=init_str,
        evaluate_str=eval_str,
        **concurrency_kwargs,
    )

    # Shinka configures the root logger with ``force=True`` in its constructor
    # to create evolution_run.log. Install our handler afterwards so both the
    # upstream raw log and the SkyDiscover-format log remain active.
    from skydiscover.search.utils.logging_utils import setup_search_logging

    log_dir = config_obj.log_dir or os.path.join(output_dir, "logs")
    setup_search_logging(
        log_level=config_obj.log_level,
        log_dir=log_dir,
        name="shinkaevolve",
    )
    logger.info(
        "Starting ShinkaEvolve: iterations=%d, output_dir=%s",
        iterations,
        output_dir,
    )

    resolved_api_base = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or getattr(config_obj.llm, "api_base", None)
    )
    if resolved_api_base is None and config_obj.llm.models:
        resolved_api_base = config_obj.llm.models[0].api_base

    usage_tracker, close_usage_bridge = _install_usage_bridge(
        output_dir,
        api_base=resolved_api_base,
        request_extra_body=(
            dict(getattr(config_obj.llm.models[0], "extra_body", {}) or {})
            if config_obj.llm.models
            else None
        ),
    )
    reporter = _ShinkaProgressReporter(
        usage_tracker,
        output_dir,
        monitor_callback=monitor_callback,
        budget=getattr(config_obj.search.database, "budget_usd", None),
    )

    # Always poll so the normalized log is produced even when the dashboard
    # and human feedback are disabled.
    async def _poll_programs():
        last_feedback = ""
        while True:
            await asyncio.sleep(2)
            try:
                if runner.db is not None:
                    reporter.record_many(runner.db.get_all_programs())
            except Exception:
                logger.debug("Shinka program poll error", exc_info=True)

            # Human feedback: inject feedback into ShinkaEvolve's prompt sampler.
            if feedback_reader:
                try:
                    feedback = feedback_reader.read()
                    if feedback != last_feedback:
                        last_feedback = feedback
                        sampler = getattr(runner, "prompt_sampler", None)
                        original_prompt = evo_config.task_sys_msg or ""
                        if feedback and sampler:
                            if feedback_reader.mode == "replace":
                                sampler.task_sys_msg = feedback
                            else:
                                sampler.task_sys_msg = (
                                    original_prompt + "\n\n## Human Guidance\n" + feedback
                                )
                            feedback_reader.set_current_prompt(sampler.task_sys_msg)
                            logger.debug(
                                "Human feedback injected into ShinkaEvolve "
                                "(%d chars, mode=%s)",
                                len(feedback),
                                feedback_reader.mode,
                            )
                        elif sampler and not feedback:
                            sampler.task_sys_msg = original_prompt
                            feedback_reader.set_current_prompt(original_prompt)
                except Exception:
                    logger.debug("Human feedback injection error", exc_info=True)

    poll_task = asyncio.create_task(_poll_programs())
    try:
        if hasattr(runner, "run_async"):
            await runner.run_async()
        else:
            await runner.run()
    except Exception:
        logger.exception("ShinkaEvolve run failed")
        raise
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

        # Flush programs completed between the last two-second poll and exit.
        try:
            if runner.db is not None:
                reporter.record_many(runner.db.get_all_programs())
        except Exception:
            logger.debug("Final program flush error", exc_info=True)
        reporter.write()
        close_usage_bridge()

    # Extract results from the ShinkaEvolve database
    best_sp = runner.db.get_best_program()
    all_programs = runner.db.get_all_programs()

    # get_best_program() only returns "correct" programs. For continuous-score
    # problems (no pass/fail), fall back to the highest-scoring program overall.
    if best_sp is None and all_programs:
        best_sp = max(all_programs, key=lambda p: float(getattr(p, "combined_score", 0) or 0))

    initial_score = _get_initial_score(all_programs)

    best_skydiscover = _to_skydiscover_program(best_sp) if best_sp else None
    best_score = float(best_sp.combined_score or 0.0) if best_sp else 0.0
    logger.info(
        "ShinkaEvolve complete: best_score=%.12g, %s",
        best_score,
        reporter.curve.summary_line(),
    )

    return DiscoveryResult(
        best_program=best_skydiscover,
        best_score=best_score,
        best_solution=best_sp.code if best_sp else "",
        metrics=best_skydiscover.metrics if best_skydiscover else {},
        output_dir=output_dir,
        initial_score=initial_score,
    )
