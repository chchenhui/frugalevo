"""FrugalEvo controller: cost-aware two-tier evolutionary search.

The default starts with cheap direct search. After a stall, one guide response
returns a portfolio of strategies and an executable reference for its first
plan. If the reference improves, it is recorded immediately and Luna continues
that plan next; otherwise Luna runs in the current iteration. Productive plans
are retained while productive and retired when barren. The optional ordered
scheduler treats the guide program as portfolio-level evidence:
it can become the shared parent of comparable Luna pilots but earns no strategy
credit.

The default scheduler learns raw global combined-score gain per measured model
cost. A configured budget remains an optional stopping/admission guard and a
reporting window; it does not shape rewards or ordered-racing decisions. Guide calls also
propose one executable reference. The simple unbounded default uses fixed K=3
and M=6 batches.
"""

import ast
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from skydiscover.config import FrugalEvoDatabaseConfig
from skydiscover.llm.base import LLMResponse
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.frugalevo.evaluator_analysis import (
    generate_evaluator_package_guidance,
)
from skydiscover.search.frugalevo.strategy import (
    Strategy,
    StrategyLedger,
    implementation_instruction,
    parse_reference_implementation,
    parse_strategies,
    strategy_request,
)
from skydiscover.search.utils.budget_curve import BudgetCurve
from skydiscover.search.utils.discovery_utils import SerializableResult, load_evaluator_code
from skydiscover.utils.code_utils import parse_full_rewrite
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)

#: Objective values are bucketed at this resolution when deciding whether a
#: candidate is "the same solution again". Tighter than evaluator noise but
#: loose enough that float jitter in the last bits does not read as novel.
_SCORE_RESOLUTION = 9

#: A salvaged full rewrite must be at least this fraction of the parent's
#: length, so a stray snippet is not mistaken for a whole program.
_RECOVERY_MIN_RATIO = 0.5

_CHECKPOINT_STATE_FILE = "frugalevo_state.json"

#: Field defaults taken from the config dataclass, so the controller cannot
#: drift from it. A run configured with a plain ``DatabaseConfig`` -- which
#: happens when ``search.type`` is set after the config is built -- then
#: still gets FrugalEvo's documented defaults rather than a second,
#: silently different set hardcoded here.
_DB_DEFAULTS: Dict[str, Any] = {
    field.name: field.default for field in fields(FrugalEvoDatabaseConfig)
}


class FrugalEvoController(DiscoveryController):
    """Reuse a productive strategy across rounds of ``k`` implementations.

    The guide response contains a prose plan and, by default, one executable
    reference for that plan. ``k`` independent implementation-model calls then
    generate one candidate each; generation and evaluation run concurrently,
    and every usable child is inserted into the database. ``two_tier: false``
    remains as a single-tier ablation.
    """

    def _setting(self, name: str) -> Any:
        """Read a database setting, falling back to the config dataclass default."""
        return getattr(self.config.search.database, name, _DB_DEFAULTS[name])

    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)

        if not self.config.language:
            raise ValueError("FrugalEvo requires config.language to be set")
        self.language: str = self.config.language
        if self.language == "image":
            raise ValueError("FrugalEvo implementation generation supports text/code only")

        # --- implementation width ---
        self.schedule = str(self._setting("schedule"))
        if self.schedule not in ("frontload", "fixed"):
            raise ValueError(f"schedule must be 'frontload' or 'fixed', got {self.schedule!r}")
        self.implementations_per_strategy = max(
            1, int(self._setting("implementations_per_strategy"))
        )
        # ``candidates_per_iteration`` was the width knob in the old
        # one-response/k-candidate implementation.  Accept it as a deprecated
        # alias when explicitly configured, but keep one canonical runtime
        # value so the prompt, budget forecast and metadata cannot disagree.
        legacy_width = self._setting("candidates_per_iteration")
        if legacy_width is not None:
            legacy_width = int(legacy_width)
            if legacy_width < 1:
                raise ValueError("candidates_per_iteration must be at least 1")
            self.implementations_per_strategy = legacy_width
            logger.warning(
                "candidates_per_iteration is deprecated for FrugalEvo; "
                "use implementations_per_strategy"
            )
        self.candidates_first = max(1, int(self._setting("candidates_first")))
        self.candidates_last = max(1, int(self._setting("candidates_last")))
        # Compatibility attribute used by a few integrations.  It always
        # reflects the actual fixed width now.
        self.candidates_per_iteration = self.implementations_per_strategy
        self.stagnation_patience = max(1, int(self._setting("stagnation_patience")))

        # --- optional stopping/reporting budget ---
        self.budget = self._setting("budget_usd")
        self.curve = BudgetCurve(
            budget=self.budget,
            unit=str(self._setting("budget_unit")),
        )
        self.target_score = self._setting("target_score")
        if self.target_score is not None:
            self.target_score = float(self.target_score)
        self.target_score_tolerance = max(0.0, float(self._setting("target_score_tolerance")))
        self.stop_on_target = bool(self._setting("stop_on_target"))
        self.success_score = self._setting("success_score")
        if self.success_score is not None:
            self.success_score = float(self.success_score)
        self.success_score_margin = max(0.0, float(self._setting("success_score_margin")))

        # --- cost of a single call ---
        self.reasoning_effort = self._setting("reasoning_effort")
        self.ambitious_first_call = bool(self._setting("ambitious_first_call"))
        self.recover_unusable_diffs = bool(self._setting("recover_unusable_diffs"))

        # --- strategy tier (the expensive model) ---
        self.two_tier = bool(self._setting("two_tier"))
        self.opening_cascade = bool(self._setting("opening_cascade"))
        # ``0`` is an intentional Terra-first opening: skip the cheap probe
        # and let the direct guide hedge establish the initial architecture.
        # This is useful for benchmarks whose seed program is only an empty
        # interface skeleton.  Positive values retain the normal cheap-first
        # anytime cascade.
        self.opening_cheap_candidates = max(0, int(self._setting("opening_cheap_candidates")))
        self.opening_direct_guide_hedge = bool(self._setting("opening_direct_guide_hedge"))
        self.opening_post_guide_cheap_probe = bool(
            self._setting("opening_post_guide_cheap_probe")
        )
        self.opening_guide_reasoning_effort = self._setting("opening_guide_reasoning_effort")
        self.opening_failure_retries = max(0, int(self._setting("opening_failure_retries")))
        self.opening_guide_defer_target_ratio = float(
            self._setting("opening_guide_defer_target_ratio")
        )
        if not 0.0 <= self.opening_guide_defer_target_ratio <= 1.0:
            raise ValueError("opening_guide_defer_target_ratio must be between 0 and 1")
        self.strategy_reference_candidate = bool(self._setting("strategy_reference_candidate"))
        self.strategy_reference_failure_patience = max(
            1, int(self._setting("strategy_reference_failure_patience"))
        )
        self._pending_strategy_reference: Optional[str] = None
        self._pending_strategy_reference_prompt: Optional[Dict[str, str]] = None
        self._pending_strategy_reference_requested = False
        self._strategy_reference_failure_streak = 0
        self._strategy_reference_disabled = False
        self.strategy_when = str(self._setting("strategy_when"))
        if self.strategy_when not in ("reuse_on_improvement", "stalled", "always"):
            raise ValueError(
                "strategy_when must be 'reuse_on_improvement', 'stalled', or 'always', "
                f"got {self.strategy_when!r}"
            )
        self.strategy_patience = max(1, int(self._setting("strategy_patience")))
        self.improvement_epsilon = max(0.0, float(self._setting("improvement_epsilon")))
        self.initial_strategy_reasoning_effort = self._setting("initial_strategy_reasoning_effort")
        self.strategy_reasoning_effort = self._setting("strategy_reasoning_effort")
        self.strategy_escalation_reasoning_effort = self._setting(
            "strategy_escalation_reasoning_effort"
        )
        self.strategy_escalation_patience = max(
            1, int(self._setting("strategy_escalation_patience"))
        )
        self.strategy_escalation_interval = max(
            1, int(self._setting("strategy_escalation_interval"))
        )
        self.strategy_escalation_backoff = max(
            1.0, float(self._setting("strategy_escalation_backoff"))
        )
        self.guide_implementation_on_escalation = bool(
            self._setting("guide_implementation_on_escalation")
        )
        self.guide_implementation_reasoning_effort = self._setting(
            "guide_implementation_reasoning_effort"
        )
        self.guide_implementation_after_productive_stall = bool(
            self._setting("guide_implementation_after_productive_stall")
        )
        self.target_aware_guide_promotion_ratio = self._setting(
            "target_aware_guide_promotion_ratio"
        )
        if self.target_aware_guide_promotion_ratio is not None:
            self.target_aware_guide_promotion_ratio = float(self.target_aware_guide_promotion_ratio)
            if not 0.0 <= self.target_aware_guide_promotion_ratio <= 1.0:
                raise ValueError("target_aware_guide_promotion_ratio must be between 0 and 1")
        self.unguided_guide_hedge_interval = max(
            0, int(self._setting("unguided_guide_hedge_interval"))
        )
        self.unguided_guide_hedge_warmup_calls = max(
            1, int(self._setting("unguided_guide_hedge_warmup_calls"))
        )
        self.min_guide_amortization_iterations = max(
            1, int(self._setting("min_guide_amortization_iterations"))
        )
        self.strategies_per_guide_call = max(1, int(self._setting("strategies_per_guide_call")))
        self.initial_strategies_per_guide_call = max(
            1, int(self._setting("initial_strategies_per_guide_call"))
        )
        self.auto_generate_strategy_guidance = bool(
            self._setting("auto_generate_strategy_guidance")
        )
        self._strategy_objective_guidance = ""
        self._strategy_diverge_guidance = ""
        self._strategy_refine_guidance = ""
        self._strategy_guidance_attempted = False
        self._strategy_guidance_cost = 0.0
        self.ordered_strategy_racing = bool(self._setting("ordered_strategy_racing"))
        self.ordered_pilot_pruning = bool(self._setting("ordered_pilot_pruning"))
        self.ordered_pilot_prune_nonpositive = bool(
            self._setting("ordered_pilot_prune_nonpositive")
        )
        self.search_incumbent_enabled = bool(self._setting("search_incumbent_enabled"))
        self.search_incumbent_elite_ratio = float(
            self._setting("search_incumbent_elite_ratio")
        )
        if not 0.0 < self.search_incumbent_elite_ratio <= 1.0:
            raise ValueError("search_incumbent_elite_ratio must be in (0, 1]")
        self.search_incumbent_stagnation_patience = max(
            1, int(self._setting("search_incumbent_stagnation_patience"))
        )
        self.search_incumbent_cooldown = max(0, int(self._setting("search_incumbent_cooldown")))
        self.sequential_implementation_feedback = bool(
            self._setting("sequential_implementation_feedback")
        )
        self.lazy_strategy_guidance = bool(self._setting("lazy_strategy_guidance"))
        self.prompt_context_max_programs = max(0, int(self._setting("prompt_context_max_programs")))
        self.prompt_context_max_chars = max(0, int(self._setting("prompt_context_max_chars")))
        self.adaptive_implementation_racing = bool(self._setting("adaptive_implementation_racing"))
        self.adaptive_unguided_racing = bool(self._setting("adaptive_unguided_racing"))
        self.pilot_candidates = max(1, int(self._setting("pilot_candidates")))
        self.pilot_score_ratio = float(self._setting("pilot_score_ratio"))
        self.stop_racing_on_improvement = bool(self._setting("stop_racing_on_improvement"))
        self.adaptive_pilot_sizing = bool(self._setting("adaptive_pilot_sizing"))
        self.reliable_pilot_min_attempts = max(1, int(self._setting("reliable_pilot_min_attempts")))
        self.reliable_pilot_usable_ratio = float(self._setting("reliable_pilot_usable_ratio"))
        if not 0.0 <= self.pilot_score_ratio <= 1.0:
            raise ValueError("pilot_score_ratio must be between 0 and 1")
        if not 0.0 <= self.reliable_pilot_usable_ratio <= 1.0:
            raise ValueError("reliable_pilot_usable_ratio must be between 0 and 1")

        self.strategy_history_size = max(1, int(self._setting("strategy_history_size")))
        self.ledger = StrategyLedger()
        self._strategy: Optional[Strategy] = None
        self._strategy_queue: List[Strategy] = []
        self._ordered_portfolio_roots: List[int] = []
        self._ordered_exploitation_queue: List[int] = []
        self._ordered_active_root: Optional[int] = None
        self._ordered_pilot_scores: Dict[int, Optional[float]] = {}
        self._ordered_phase = "idle"
        self._ordered_pending_pilots: List[Strategy] = []
        self._ordered_reference_feedback: List[str] = []
        self._strategy_costs: List[float] = []
        self._impl_costs: List[float] = []
        self._guide_impl_costs: List[float] = []
        self._barren_rounds = 0
        self._last_strategy_call_escalated = False
        self._guide_implementation_due = False
        self._guide_implementation_due_reason: Optional[str] = None
        self._guide_implementation_unguided = False
        self._strategy_call_count = 0
        self._last_guide_implementation_spend = 0.0
        self._last_guide_implementation_count = 0
        self._last_unguided_guide_count = 0
        self._anchor_next_cheap_to_incumbent = False
        self._opening_cheap_probe_done = self.opening_cheap_candidates == 0
        self._opening_failure_retries_used = 0
        self._opening_guide_hedge_done = False
        self._opening_guide_hedge_active = False
        self._opening_strategy_handoff_active = False
        self._success_achieved = False

        # --- stagnation response and novelty feedback ---
        self.novelty_memory = max(0, int(self._setting("novelty_memory")))
        self.stalled_exploration_ratio = self._setting("stalled_exploration_ratio")
        self._base_exploration_ratio = getattr(self.database, "exploration_ratio", None)
        self._exploring = False

        # --- run state ---
        self._seen_solutions: Set[str] = set()
        self._score_ledger: "OrderedDict[float, int]" = OrderedDict()
        self._stagnation = 0
        self._next_strategy_escalation = self.strategy_escalation_patience
        self._strategy_escalation_gap = self.strategy_escalation_interval
        self._cost_samples: List[Tuple[int, float]] = []
        self._progress: Tuple[int, int] = (0, 0)

        if self.checkpoint_path:
            self._load_checkpoint_state(self.checkpoint_path)

        strategist = (
            ", ".join(m.name or "?" for m in self.config.llm.guide_models)
            or "same as weak model"
        )
        implementer = ", ".join(m.name or "?" for m in self.config.llm.models)
        guide_names = [model.name for model in self.config.llm.guide_models]
        implementation_names = [model.name for model in self.config.llm.models]
        if self.two_tier and guide_names == implementation_names:
            logger.warning(
                "FrugalEvo strong model matches weak model; configure a distinct "
                "strong model if the strategy tier should use a stronger model"
            )
        steady_strategy_effort = self.strategy_reasoning_effort or "model default"
        opening_strategy_effort = self.initial_strategy_reasoning_effort or steady_strategy_effort
        strategy_effort_label = steady_strategy_effort
        if opening_strategy_effort != steady_strategy_effort:
            strategy_effort_label = (
                f"opening:{opening_strategy_effort}/later:{steady_strategy_effort}"
            )
        logger.info(
            "FrugalEvo two-tier: strong model=%s (effort=%s, cadence=%s) -> "
            "weak model=%s (effort=%s) %s%d per strategy, "
            "opening=%s, portfolio=opening:%d plateau:%d, "
            "racing=%s, budget=%s",
            strategist,
            strategy_effort_label,
            self.strategy_when,
            implementer,
            self.reasoning_effort or "model default",
            "x",
            self.implementations_per_strategy,
            (
                f"cheap:{self.opening_cheap_candidates}/strong model:"
                + (
                    f"1@{self.opening_guide_reasoning_effort or 'model default'}"
                    f"/handoff:{'cheap' if self.opening_post_guide_cheap_probe else 'strategy'}"
                    if self.opening_direct_guide_hedge
                    else "disabled"
                )
                if self.opening_cascade
                else "disabled"
            ),
            self.initial_strategies_per_guide_call,
            self.strategies_per_guide_call,
            (
                "ordered-pilot-ranking"
                if self.ordered_strategy_racing
                else ("pilot-first" if self.adaptive_implementation_racing else "fixed-width")
            ),
            f"{self.budget:g} {self.curve.unit}" if self.budget else "unbounded",
        )
        if self.search_incumbent_enabled:
            logger.info(
                "Search incumbent diversification: use global best until %d consecutive "
                "stalled round(s), then sample from the top %.0f%% score band "
                "with %d incumbent-anchored cooldown round(s)",
                self.search_incumbent_stagnation_patience,
                100.0 * self.search_incumbent_elite_ratio,
                self.search_incumbent_cooldown,
            )
        if self.sequential_implementation_feedback:
            logger.info("Sequential implementation feedback enabled: stop on global gain")
        if self.strategy_reference_candidate:
            logger.info(
                "Strong-model reference generation remains enabled for every new "
                "strategy portfolio"
            )
        if self.ordered_strategy_racing:
            logger.info(
                "Ordered strategy racing enabled: one weak model pilot per strong model strategy, "
                "then score-ordered batches of M=%d until each retained strategy stalls%s",
                self.implementations_per_strategy,
                (
                    "; pilot pruning discards invalid and non-positive pilots"
                    if self.ordered_pilot_pruning
                    else ""
                ),
            )

    async def _call_llm(self, system_message: str, user_message: str, **kwargs: Any) -> LLMResponse:
        """Route generation through the configured reasoning effort.

        Effort is the largest single multiplier on the cost of a call, so it is
        part of the search policy rather than a static model setting.
        """
        if self.reasoning_effort and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return await super()._call_llm(system_message, user_message, **kwargs)

    async def _generate_strategy_guidance(self) -> None:
        """Run EE's EvoX-inspired evaluator/package analysis once.

        The EE-local generator returns only its parsed search examples. This
        preserves FrugalEvo's strategy-plan response contract instead of
        embedding EvoX's direct code-generation wrapper.
        """
        if (
            not self.two_tier
            or not self.auto_generate_strategy_guidance
            or self._strategy_guidance_attempted
        ):
            return

        self._strategy_guidance_attempted = True
        cost_before = self.curve.spent()
        try:
            evaluator_code = load_evaluator_code(self.evaluation_file)
            if not evaluator_code.strip():
                logger.warning(
                    "Skipping evaluator/package strategy analysis: evaluator source unavailable"
                )
                return

            evaluation_path = Path(self.evaluation_file)
            problem_dir = evaluation_path if evaluation_path.is_dir() else evaluation_path.parent
            model_names = ", ".join(model.name or "?" for model in self.config.llm.guide_models)
            logger.info(
                "Generating evaluator/package-aware strategy guidance with guide model(s): %s",
                model_names or "unknown",
            )
            objective, diverge, refine = await generate_evaluator_package_guidance(
                self.config.context_builder.system_message or "",
                evaluator_code,
                problem_dir=str(problem_dir),
                llm_pool=self.guide_llms,
            )
            self._strategy_objective_guidance = objective.strip()
            self._strategy_diverge_guidance = diverge.strip()
            self._strategy_refine_guidance = refine.strip()
            logger.info(
                "Generated evaluator/package-aware EE guidance "
                "(%d objective/%d exploration/%d refinement chars)",
                len(self._strategy_objective_guidance),
                len(self._strategy_diverge_guidance),
                len(self._strategy_refine_guidance),
            )
        except Exception as exc:
            self._strategy_objective_guidance = ""
            self._strategy_diverge_guidance = ""
            self._strategy_refine_guidance = ""
            logger.warning("Could not generate evaluator/package strategy guidance: %s", exc)
        finally:
            self._strategy_guidance_cost += max(0.0, self.curve.spent() - cost_before)

    # ==================================================================
    # Main loop
    # ==================================================================

    async def run_discovery(
        self,
        start_iteration: int,
        max_iterations: int,
        checkpoint_callback: Optional[Callable[[int], None]] = None,
        post_process_result: Optional[bool] = True,
        retry_times: Optional[int] = 3,
    ) -> Optional[Union[Program, SerializableResult]]:
        """Run cost-aware two-tier iterations until an external limit closes.

        In the default mode, one cheap direct probe and one low-effort direct
        guide implementation are recorded as separate opening rounds before a
        prose strategy portfolio is considered. A productive strategy is then
        reused; after it stalls, sparse guide-code insurance may be used before
        replacement. Normally every strategy round makes ``candidates`` calls
        to the cheap tier. ``retry_times`` is intentionally unused: retrying a
        failed candidate would buy another model call without new evidence.
        Transport retries inside the configured LLM client remain possible.
        """
        del retry_times
        total_iterations = start_iteration + max_iterations
        last_result: Optional[SerializableResult] = None

        self._seed_curve()
        if not self.lazy_strategy_guidance:
            await self._generate_strategy_guidance()

        for iteration in range(start_iteration, total_iterations):
            # Drives the width schedule when there is no budget to measure
            # progress against.
            self._progress = (iteration - start_iteration, max_iterations)
            if self.shutdown_event.is_set():
                logger.info("Shutdown requested, stopping FrugalEvo early")
                break
            if self._target_reached():
                logger.info(
                    "Stopping at iteration %d: target score %.12g reached (%s)",
                    iteration,
                    self.target_score,
                    self.curve.summary_line(),
                )
                break

            opening_probe = self._opening_cheap_probe_due()
            candidates = self._plan_iteration(iteration)
            if candidates is None:
                break
            if opening_probe:
                # Consume only after budget admission. A parse/evaluation
                # failure still counts as the probe: repeating it would turn
                # the bounded opening cost into an unbounded retry policy.
                self._opening_cheap_probe_done = True
                logger.info(
                    "Iteration %d: opening cascade records %d cheap direct probe(s) "
                    "before the guide-model hedge",
                    iteration,
                    candidates,
                )
            # Sample exactly once.  The strategist and every implementer in
            # this iteration must reason about the same program and context.
            wants_new_strategy = False if opening_probe else self._wants_new_strategy()
            guided_parent = self._should_anchor_strategy_parent(wants_new_strategy)
            use_search_incumbent = self._search_incumbent_due(guided_parent)
            parent_mode = "search_incumbent" if use_search_incumbent else None
            sample = self._sample_parent_and_context(parent_mode=parent_mode)
            consume_cheap_anchor = getattr(self, "_anchor_next_cheap_to_incumbent", False)
            if guided_parent and not use_search_incumbent:
                sample = self._anchor_sample_to_incumbent(sample)
            elif use_search_incumbent:
                selected_parent = sample[1]
                global_best = self.database.get_best_program() if self.database.programs else None
                logger.info(
                    "Selected search incumbent %s (score %.6f) after %d stalled round(s) "
                    "from top %.0f%% band; "
                    "global best remains %s (score %.6f)",
                    selected_parent.id[:8] if selected_parent is not None else "none",
                    get_score(selected_parent.metrics) if selected_parent is not None else 0.0,
                    self._stagnation,
                    100.0 * self.search_incumbent_elite_ratio,
                    global_best.id[:8] if global_best is not None else "none",
                    get_score(global_best.metrics) if global_best is not None else 0.0,
                )
            if consume_cheap_anchor:
                self._anchor_next_cheap_to_incumbent = False
            (
                raw_parent,
                parent,
                _parent_info,
                context_dict,
                _context_ids,
                _context_info,
            ) = sample
            round_cost_before = self.curve.spent()
            self._last_reference_candidate_count = 0
            outcomes: List[Tuple[int, SerializableResult]] = []
            ordered_pilot_batch: Optional[List[Tuple[Strategy, SerializableResult]]] = None
            ordered_reference_outcome: Optional[SerializableResult] = None
            pre_strategy_attempts = 0
            strategy_attempted = False
            implementation_cost_before = round_cost_before

            if wants_new_strategy:
                if self._strategy is not None:
                    logger.info(
                        "Iteration %d: completed strategy %s after %d round(s) — %s",
                        iteration,
                        self._strategy.label,
                        self._strategy.rounds,
                        self._strategy.outcome(),
                    )
                # A scheduled unguided Terra candidate does not consume or use
                # the next prose portfolio.  Run it first: if it improves, the
                # expensive strategy call would be pure deadweight; if it
                # fails, buy the portfolio and spend only the remaining width
                # on Luna, preserving the original per-iteration call count.
                if self._pre_strategy_hedge_due():
                    opening_hedge = self._opening_guide_hedge_due()
                    if opening_hedge:
                        self._opening_guide_hedge_done = True
                        self._opening_guide_hedge_active = True
                    self._strategy = None
                    self._guide_implementation_due = True
                    self._guide_implementation_due_reason = (
                        "opening-cascade direct guide hedge"
                        if opening_hedge
                        else (
                            "pre-portfolio unguided hedge before guide call "
                            f"{self._strategy_call_count + 1}"
                        )
                    )
                    self._guide_implementation_unguided = True
                    logger.info(
                        "Iteration %d: trying the %s unguided guide hedge before "
                        "buying a new strategy portfolio",
                        iteration,
                        "opening-cascade" if opening_hedge else "scheduled",
                    )
                    try:
                        outcomes = await self._implement_strategy(
                            iteration,
                            None,
                            1,
                            sample=sample,
                        )
                    finally:
                        self._opening_guide_hedge_active = False
                    pre_strategy_attempts = len(outcomes)
                    hedge_improved = any(
                        self._outcome_improves_incumbent(outcome) for _, outcome in outcomes
                    )
                    if hedge_improved:
                        # "The same strategy after an improvement" is "no
                        # strategy" for an unguided hedge. The historical
                        # policy gives the stronger incumbent one cheap pull;
                        # the one-shot opening ablation hands it directly to
                        # the normal strategy policy instead.
                        self._barren_rounds = 0
                        if opening_hedge and not self.opening_post_guide_cheap_probe:
                            self._opening_strategy_handoff_active = True
                            logger.info(
                                "Iteration %d: opening guide hedge raised the incumbent; "
                                "handing off directly to the strategy policy next iteration",
                                iteration,
                            )
                        else:
                            self._anchor_next_cheap_to_incumbent = True
                            logger.info(
                                "Iteration %d: pre-portfolio hedge raised the incumbent; "
                                "skipping the unused strategy call and weak model batch",
                                iteration,
                            )
                    elif opening_hedge:
                        # Keep each opening decision observable. The failed
                        # direct hedge is recorded at its own cost; the prose
                        # portfolio gets normal budget admission next round.
                        logger.info(
                            "Iteration %d: opening guide hedge did not improve; "
                            "deferring the prose portfolio to the next iteration",
                            iteration,
                        )
                    else:
                        # Never silently reuse an old strategy when this
                        # iteration is required to buy a new one.
                        self._strategy = await self._propose_strategy(
                            iteration,
                            raw_parent=raw_parent,
                            parent=parent,
                            context_dict=context_dict,
                        )
                        self._barren_rounds = 0
                        if self._strategy is not None:
                            remaining = max(0, candidates - pre_strategy_attempts)
                            if remaining:
                                self._strategy.rounds += 1
                                implementation_cost_before = self.curve.spent()
                                strategy_outcomes = await self._implement_strategy(
                                    iteration,
                                    self._strategy,
                                    remaining,
                                    sample=sample,
                                )
                                second_guide_count = self._last_guide_implementation_count
                                second_guide_spend = self._last_guide_implementation_spend
                                # The second batch resets these accounting
                                # markers. Restore the already-priced hedge;
                                # its cost is outside implementation_cost_before.
                                self._last_unguided_guide_count = pre_strategy_attempts
                                self._last_guide_implementation_count = (
                                    pre_strategy_attempts + second_guide_count
                                )
                                self._last_guide_implementation_spend = second_guide_spend
                                outcomes.extend(strategy_outcomes)
                                strategy_attempted = True
                else:
                    # Never silently reuse an old strategy when this iteration
                    # is required to buy a new one.
                    self._strategy = await self._propose_strategy(
                        iteration,
                        raw_parent=raw_parent,
                        parent=parent,
                        context_dict=context_dict,
                    )
                    self._barren_rounds = 0

                    if self._strategy is not None:
                        # The reference is bundled into the already-priced
                        # strategy response, so it has no implementation-tier
                        # spend of its own.
                        implementation_cost_before = self.curve.spent()
                        reference_outcome = await self._evaluate_strategy_reference(
                            iteration,
                            None if self.ordered_strategy_racing else self._strategy,
                            sample,
                            candidates + 1,
                        )
                        if reference_outcome is not None:
                            outcomes.append((0, reference_outcome))
                            ordered_reference_outcome = reference_outcome
                            self._last_reference_candidate_count = 1
                            if (
                                not self.ordered_strategy_racing
                                and self._outcome_improves_incumbent(reference_outcome)
                            ):
                                # End the iteration at the earliest paid
                                # improvement. Luna receives the stronger
                                # incumbent and the same strategy next round;
                                # charging its batch before recording this
                                # point would charge work before exposing a
                                # known gain, for no information benefit.
                                self._strategy.rounds += 1
                                strategy_attempted = True
                                logger.info(
                                    "Iteration %d: guide reference raised the incumbent; "
                                    "deferring the weak model batch to the next iteration",
                                    iteration,
                                )

                        pending_pilots = list(getattr(self, "_ordered_pending_pilots", []))
                        if self.ordered_strategy_racing and pending_pilots:
                            self._ordered_pending_pilots = []
                            implementation_cost_before = self.curve.spent()
                            pilot_sample = self._ordered_pilot_sample(
                                sample,
                                reference_outcome,
                            )
                            ordered_pilot_batch = await self._run_ordered_pilots(
                                iteration,
                                pending_pilots,
                                pilot_sample,
                            )
                            outcomes.extend(
                                (position, outcome)
                                for position, (_strategy, outcome) in enumerate(
                                    ordered_pilot_batch,
                                    start=1,
                                )
                            )
                            strategy_attempted = True

                if self._strategy is None and not outcomes:
                    logger.warning(
                        "Iteration %d: strategy generation failed; recording the spent call "
                        "and continuing to the next iteration",
                        iteration,
                    )

            if (
                self._strategy is not None
                and not strategy_attempted
                and pre_strategy_attempts < candidates
            ):
                self._strategy.rounds += 1
                implementation_cost_before = self.curve.spent()
                implementation_outcomes = await self._implement_strategy(
                    iteration,
                    self._strategy,
                    candidates - pre_strategy_attempts,
                    sample=sample,
                )
                outcomes.extend(implementation_outcomes)
                strategy_attempted = True
            elif self._strategy is None and not wants_new_strategy:
                implementation_cost_before = self.curve.spent()
                opening_options = {}
                if opening_probe and self._setting("opening_independent_sampling"):
                    opening_options["independent"] = True
                outcomes = await self._implement_strategy(
                    iteration,
                    None,
                    candidates,
                    sample=sample,
                    **opening_options,
                )

            attempted = len(outcomes)
            successful: List[Tuple[int, SerializableResult]] = []
            for position, (_, outcome) in enumerate(outcomes, start=1):
                last_result = outcome
                if outcome.error:
                    if (
                        ordered_pilot_batch is None
                        and self._strategy is not None
                        and position > pre_strategy_attempts
                    ):
                        self._strategy.record_failure(self._failure_feedback(outcome))
                    logger.warning(
                        "Iteration %d implementation %d/%d failed: %s",
                        iteration,
                        position,
                        attempted,
                        self._failure_feedback(outcome),
                    )
                    continue
                successful.append((position, outcome))

            if opening_probe:
                self._update_opening_probe_retry(
                    iteration,
                    usable=any(outcome.child_program_dict for _, outcome in successful),
                )

            if post_process_result:
                for candidate_index, outcome in successful:
                    self._process_iteration_result(
                        outcome,
                        iteration,
                        checkpoint_callback=None,
                        run_checkpoint=False,
                    )

            ordered_baseline = self.curve.incumbent
            self._record_iteration(
                iteration,
                attempted,
                successful,
                round_cost_before,
                implementation_cost_before=implementation_cost_before,
                strategy_attempted=strategy_attempted and ordered_pilot_batch is None,
            )
            if ordered_pilot_batch is not None:
                self._finish_ordered_pilots(
                    iteration,
                    ordered_pilot_batch,
                    reference_outcome=ordered_reference_outcome,
                    baseline=ordered_baseline,
                    implementation_spend=max(0.0, self.curve.spent() - implementation_cost_before),
                )
            if iteration > 0 and iteration % self.config.checkpoint_interval == 0:
                self.database.log_status()
                if checkpoint_callback:
                    checkpoint_callback(iteration)

        self.curve.write(self.output_dir)
        self._restore_exploration_ratio()

        if not post_process_result:
            return last_result
        return self._finalize_discovery()

    # ==================================================================
    # Budget accounting and adaptation
    # ==================================================================

    def _seed_curve(self) -> None:
        """Anchor the curve at (cost 0, score of the seed program).

        The initial program costs no LLM call, so whatever it scores is held
        for free from the origin -- and every algorithm compared on this task
        gets the same head start.
        """
        if self.curve.points:
            return
        best = self.database.get_best_program() if self.database.programs else None
        if best is not None:
            self.curve.observe(get_score(best.metrics), iteration=0, candidates=0)

    def _target_reached(self) -> bool:
        """Whether a caller-provided sufficient score has been attained."""
        if not getattr(self, "stop_on_target", True):
            return False
        target = getattr(self, "target_score", None)
        if target is None or not self.curve.points:
            return False
        tolerance = getattr(self, "target_score_tolerance", 0.0)
        return self.curve.incumbent + tolerance >= target

    def save_checkpoint_state(self, checkpoint_path: str) -> None:
        """Persist controller, strategy, cost, and reporting-curve state."""
        os.makedirs(checkpoint_path, exist_ok=True)
        state = {
            "version": 24,
            "curve": self.curve.to_dict(),
            "strategy_objective_guidance": getattr(self, "_strategy_objective_guidance", ""),
            "strategy_diverge_guidance": getattr(self, "_strategy_diverge_guidance", ""),
            "strategy_refine_guidance": getattr(self, "_strategy_refine_guidance", ""),
            "strategy_guidance_attempted": getattr(self, "_strategy_guidance_attempted", False),
            "strategy_guidance_cost": getattr(self, "_strategy_guidance_cost", 0.0),
            "strategies": [asdict(strategy) for strategy in self.ledger.entries],
            "active_strategy": self._strategy.index if self._strategy else None,
            "strategy_queue": [strategy.index for strategy in self._strategy_queue],
            "ordered_portfolio_roots": getattr(self, "_ordered_portfolio_roots", []),
            "ordered_exploitation_queue": getattr(self, "_ordered_exploitation_queue", []),
            "ordered_active_root": getattr(self, "_ordered_active_root", None),
            "ordered_pilot_scores": getattr(self, "_ordered_pilot_scores", {}),
            "ordered_phase": getattr(self, "_ordered_phase", "idle"),
            "ordered_reference_feedback": getattr(
                self, "_ordered_reference_feedback", []
            ),
            "strategy_reference_failure_streak": getattr(
                self, "_strategy_reference_failure_streak", 0
            ),
            "strategy_reference_disabled": getattr(
                self, "_strategy_reference_disabled", False
            ),
            "strategy_costs": self._strategy_costs,
            "implementation_costs": self._impl_costs,
            "guide_implementation_costs": self._guide_impl_costs,
            "guide_implementation_due": self._guide_implementation_due,
            "guide_implementation_due_reason": self._guide_implementation_due_reason,
            "guide_implementation_unguided": getattr(self, "_guide_implementation_unguided", False),
            "strategy_call_count": getattr(self, "_strategy_call_count", 0),
            "anchor_next_cheap_to_incumbent": getattr(
                self, "_anchor_next_cheap_to_incumbent", False
            ),
            "opening_cheap_probe_done": getattr(self, "_opening_cheap_probe_done", False),
            "opening_failure_retries_used": getattr(self, "_opening_failure_retries_used", 0),
            "opening_guide_hedge_done": getattr(self, "_opening_guide_hedge_done", False),
            "success_achieved": getattr(self, "_success_achieved", False),
            "barren_rounds": self._barren_rounds,
            "stagnation": self._stagnation,
            "next_strategy_escalation": self._next_strategy_escalation,
            "strategy_escalation_gap": self._strategy_escalation_gap,
            "seen_solutions": sorted(self._seen_solutions),
            "score_ledger": list(self._score_ledger.items()),
            "cost_samples": self._cost_samples,
            "exploring": self._exploring,
        }
        path = os.path.join(checkpoint_path, _CHECKPOINT_STATE_FILE)
        with open(path, "w") as handle:
            json.dump(state, handle, indent=2)

    def _load_checkpoint_state(self, checkpoint_path: str) -> None:
        """Restore controller state; old database-only checkpoints remain valid."""
        path = os.path.join(checkpoint_path, _CHECKPOINT_STATE_FILE)
        if not os.path.exists(path):
            logger.info(
                "Checkpoint has no FrugalEvo controller state; starting a fresh "
                "strategy ledger and cost segment"
            )
            return
        try:
            with open(path) as handle:
                state = json.load(handle)
            strategy_field_names = {field.name for field in fields(Strategy)}
            strategies = [
                Strategy(
                    **{
                        key: value
                        for key, value in record.items()
                        if key in strategy_field_names
                    }
                )
                for record in state.get("strategies", [])
            ]
            self.ledger = StrategyLedger(entries=strategies)
            by_index = {strategy.index: strategy for strategy in strategies}
            self._strategy = by_index.get(state.get("active_strategy"))
            self._strategy_queue = [
                by_index[index] for index in state.get("strategy_queue", []) if index in by_index
            ]
            valid_roots = {strategy.root_index for strategy in strategies}
            self._ordered_portfolio_roots = [
                int(index)
                for index in state.get("ordered_portfolio_roots", [])
                if int(index) in valid_roots
            ]
            self._ordered_exploitation_queue = [
                int(index)
                for index in state.get("ordered_exploitation_queue", [])
                if int(index) in valid_roots
            ]
            ordered_active = state.get("ordered_active_root")
            self._ordered_active_root = (
                int(ordered_active)
                if ordered_active is not None and int(ordered_active) in valid_roots
                else None
            )
            self._ordered_pilot_scores = {
                int(index): None if score is None else float(score)
                for index, score in state.get("ordered_pilot_scores", {}).items()
                if int(index) in valid_roots
            }
            ordered_phase = str(state.get("ordered_phase", "idle"))
            self._ordered_phase = (
                ordered_phase if ordered_phase in ("idle", "pilot", "exploit") else "idle"
            )
            self._ordered_pending_pilots = []
            self._ordered_reference_feedback = [
                str(item)
                for item in state.get("ordered_reference_feedback", [])
                if str(item).strip()
            ][-4:]
            if "strategy_reference_failure_streak" in state:
                self._strategy_reference_failure_streak = max(
                    0, int(state.get("strategy_reference_failure_streak", 0))
                )
            else:
                # Old ordered-racing checkpoints already contain enough
                # reference feedback to recover the trailing failure streak.
                trailing_failures = 0
                for item in reversed(self._ordered_reference_feedback):
                    if "promoted to shared pilot parent" in item:
                        break
                    trailing_failures += 1
                self._strategy_reference_failure_streak = trailing_failures
            # Version <=23 could permanently retire reference programs. Keep
            # loading its diagnostic streak, but explicitly reopen references.
            self._strategy_reference_disabled = False
            self._strategy_costs = [float(value) for value in state.get("strategy_costs", [])]
            self._impl_costs = [float(value) for value in state.get("implementation_costs", [])]
            self._guide_impl_costs = [
                float(value) for value in state.get("guide_implementation_costs", [])
            ]
            self._guide_implementation_due = bool(state.get("guide_implementation_due", False))
            self._guide_implementation_due_reason = state.get("guide_implementation_due_reason")
            self._guide_implementation_unguided = bool(
                state.get("guide_implementation_unguided", False)
            )
            self._strategy_call_count = max(0, int(state.get("strategy_call_count", 0)))
            self._anchor_next_cheap_to_incumbent = bool(
                state.get("anchor_next_cheap_to_incumbent", False)
            )
            self._success_achieved = bool(state.get("success_achieved", False))
            self._barren_rounds = max(0, int(state.get("barren_rounds", 0)))
            self._stagnation = max(0, int(state.get("stagnation", 0)))
            self._next_strategy_escalation = max(
                self.strategy_escalation_patience,
                int(state.get("next_strategy_escalation", self.strategy_escalation_patience)),
            )
            self._strategy_escalation_gap = max(
                self.strategy_escalation_interval,
                int(state.get("strategy_escalation_gap", self.strategy_escalation_interval)),
            )
            self._seen_solutions = set(state.get("seen_solutions", []))
            self._score_ledger = OrderedDict(
                (float(score), int(count)) for score, count in state.get("score_ledger", [])
            )
            self._cost_samples = [
                (int(width), float(cost)) for width, cost in state.get("cost_samples", [])
            ]
            self._strategy_objective_guidance = str(
                state.get("strategy_objective_guidance", "")
            ).strip()
            self._strategy_diverge_guidance = str(
                state.get("strategy_diverge_guidance", "")
            ).strip()
            self._strategy_refine_guidance = str(state.get("strategy_refine_guidance", "")).strip()
            self._strategy_guidance_attempted = bool(
                state.get(
                    "strategy_guidance_attempted",
                    self._strategy_objective_guidance
                    or self._strategy_diverge_guidance
                    or self._strategy_refine_guidance,
                )
            )
            self._strategy_guidance_cost = max(0.0, float(state.get("strategy_guidance_cost", 0.0)))
            self._exploring = bool(state.get("exploring", False))
            if self._exploring and self._base_exploration_ratio is not None:
                setattr(self.database, "exploration_ratio", float(self.stalled_exploration_ratio))
            self.curve.restore(state.get("curve") or {})
            # A checkpoint from before the opening-cascade state existed must
            # never restart the cascade in the middle of an evolved run.
            legacy_progress = bool(
                len(self.curve.points) > 1 or self.ledger.entries or self._strategy_call_count
            )
            self._opening_cheap_probe_done = bool(
                state.get("opening_cheap_probe_done", legacy_progress)
            )
            self._opening_failure_retries_used = max(
                0, int(state.get("opening_failure_retries_used", 0))
            )
            self._opening_guide_hedge_done = bool(
                state.get("opening_guide_hedge_done", legacy_progress)
            )
            logger.info(
                "Restored FrugalEvo state: %d curve points, %d strategies, "
                "spent=%.6f %s, stagnation=%d",
                len(self.curve.points),
                len(self.ledger.entries),
                self.curve.spent(),
                self.curve.unit,
                self._stagnation,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Could not restore FrugalEvo controller state: %s", exc)

    def _record_iteration(
        self,
        iteration: int,
        candidates: int,
        successful: List[Tuple[int, SerializableResult]],
        cost_before: float,
        *,
        implementation_cost_before: Optional[float] = None,
        strategy_attempted: bool = True,
    ) -> None:
        """Place this iteration on the budget curve and update adaptation state."""
        scores: List[float] = []
        strategy_scores: List[float] = []
        strategy_successes = 0
        strategy_reference_successes = 0
        strategy_guide_successes = 0
        unguided_count = getattr(self, "_last_unguided_guide_count", 0)
        fresh = 0
        for _, outcome in successful:
            program = outcome.child_program_dict or {}
            score = get_score(program.get("metrics") or {})
            scores.append(score)
            metadata = program.get("metadata") or {}
            strategy_index = metadata.get("strategy_index")
            is_unguided_guide = bool(
                unguided_count
                and metadata.get("implementation_model_tier") == "guide"
                and strategy_index is None
            )
            if (
                self._strategy is not None
                and not is_unguided_guide
                and (strategy_index is None or strategy_index == self._strategy.index)
            ):
                strategy_scores.append(score)
                strategy_successes += 1
                if metadata.get("implementation_model_tier") == "guide_strategy_reference":
                    strategy_reference_successes += 1
                elif metadata.get("implementation_model_tier") == "guide":
                    strategy_guide_successes += 1
            if self._register_solution(program.get("solution") or "", score):
                fresh += 1

        previous = self.curve.incumbent
        point = self.curve.observe(
            max(scores) if scores else None,
            iteration=iteration,
            candidates=candidates,
        )
        epsilon = getattr(self, "improvement_epsilon", 1e-9)
        improved = point.incumbent > previous + epsilon
        success_score = getattr(self, "success_score", None)
        success_margin = getattr(self, "success_score_margin", 0.0)
        if (
            not getattr(self, "_success_achieved", False)
            and success_score is not None
            and point.incumbent > success_score + success_margin
        ):
            self._success_achieved = True
            logger.info(
                "Iteration %d: robust success crossed %.12g + %.3g at score %.12g; "
                "recording the reporting milestone without changing search state",
                iteration,
                success_score,
                success_margin,
                point.incumbent,
            )
        # An unguided round has no active Strategy object to keep parent
        # selection anchored. Preserve the normal exploit-on-improvement rule
        # explicitly, except when the opening ablation asks to hand a
        # productive direct Terra result straight to the strategy policy.
        suppress_cheap_anchor = getattr(self, "_opening_strategy_handoff_active", False)
        self._opening_strategy_handoff_active = False
        # A cheap bootstrap must not override an explicit policy requesting
        # a strategy next. Preserve the separate post-guide cheap insurance.
        keep_cheap_climbing = (
            getattr(self, "strategy_when", "stalled") == "stalled" or unguided_count > 0
        )
        if improved and self._strategy is None and not suppress_cheap_anchor and keep_cheap_climbing:
            self._anchor_next_cheap_to_incumbent = True
        spend = point.cost - cost_before
        implementation_spend = point.cost - (
            cost_before if implementation_cost_before is None else implementation_cost_before
        )
        reference_count = getattr(self, "_last_reference_candidate_count", 0)
        if spend > 0:
            self._cost_samples.append((max(1, candidates - reference_count), spend))

        logger.info(
            "Iteration %d [%s]: %d/%d implementations usable (%d novel), Δcost=%.6f, %s",
            iteration,
            self._strategy.label if self._strategy else "no strategy",
            len(successful),
            candidates,
            fresh,
            spend,
            self.curve.summary_line(),
        )
        if self._strategy is not None and strategy_attempted:
            strategy_attempts = max(0, candidates - unguided_count)
            self._strategy.attempts += strategy_attempts
            self._strategy.implementations += strategy_successes
            self._strategy.reference_attempts += reference_count
            self._strategy.reference_implementations += strategy_reference_successes
            self._strategy.guide_implementations += strategy_guide_successes
            strategy_improved = bool(strategy_scores and max(strategy_scores) > previous + epsilon)
            hedge_preempted_strategy = bool(improved and unguided_count and not strategy_scores)
            self._strategy.last_used_iteration = iteration
            strategy_spend = implementation_spend
            if unguided_count:
                strategy_spend = max(
                    0.0,
                    strategy_spend - getattr(self, "_last_guide_implementation_spend", 0.0),
                )
            self._strategy.implementation_spend += max(0.0, strategy_spend)
            if strategy_scores:
                best = max(strategy_scores)
                if self._strategy.best_score is None or best > self._strategy.best_score:
                    self._strategy.best_score = best
            if strategy_improved:
                gain = max(0.0, point.incumbent - previous)
                scale = max(abs(previous), abs(point.incumbent), 1e-12)
                self._strategy.cumulative_gain += gain
                self._strategy.cumulative_relative_gain += gain / scale
                self._strategy.productive_rounds += 1
            elif not hedge_preempted_strategy:
                self._strategy.barren_rounds += 1
            # The low-density unguided hedge is deliberately not credited to
            # the prose strategy. If it ends the batch with a real global gain,
            # however, retain the current strategy and run its Luna
            # implementations next round. It received no implementation in
            # this round, so rebase it on the new incumbent and undo the round
            # count reserved by the main loop before the hedge ran.
            if hedge_preempted_strategy:
                self._strategy.baseline_score = point.incumbent
                self._strategy.rounds = max(0, self._strategy.rounds - 1)
                self._barren_rounds = 0
                logger.info(
                    "Iteration %d: unguided guide hedge improved the incumbent; "
                    "retaining strategy %s for the next weak model round",
                    iteration,
                    self._strategy.label,
                )
            else:
                self._barren_rounds = 0 if strategy_improved else self._barren_rounds + 1
            if (
                getattr(self, "ordered_strategy_racing", False)
                and getattr(self, "_ordered_phase", "idle") == "exploit"
                and self._strategy.root_index == getattr(self, "_ordered_active_root", None)
            ):
                if strategy_improved:
                    logger.info(
                        "Iteration %d: ordered strategy root #%d raised the incumbent; "
                        "keeping it for another M=%d weak model batch",
                        iteration,
                        self._strategy.root_index,
                        self.implementations_per_strategy,
                    )
                else:
                    logger.info(
                        "Iteration %d: ordered strategy root #%d produced no new global "
                        "best; advancing to the next pilot-ranked strategy",
                        iteration,
                        self._strategy.root_index,
                    )
                    self._ordered_active_root = None
            if (
                not getattr(self, "ordered_strategy_racing", False)
                and not improved
                and self.strategy_when != "always"
                and (
                    getattr(self, "guide_implementation_after_productive_stall", False)
                    or self._target_aware_guide_promotion_due()
                )
                and self._strategy.improved
                and self._strategy.guide_implementation_attempts == 0
            ):
                self._guide_implementation_due = True
                self._guide_implementation_due_reason = (
                    "productive strategy stalled under the weak model"
                )
                logger.info(
                    "Iteration %d: strategy %s improved before stalling; scheduling its "
                    "one guide-model implementation before retirement",
                    iteration,
                    self._strategy.label,
                )
        guide_count = getattr(self, "_last_guide_implementation_count", 0)
        guide_spend = getattr(self, "_last_guide_implementation_spend", 0.0)
        cheap_candidates = max(0, candidates - guide_count - reference_count)
        cheap_spend = max(0.0, implementation_spend - guide_spend)
        if cheap_spend > 0 and cheap_candidates > 0:
            self._impl_costs.append(cheap_spend / cheap_candidates)

        # Only a raised incumbent counts as productive. Implementations that
        # land on a score already reached bought nothing -- which is precisely
        # the plateau the novelty ledger then reports back.
        self._update_adaptation(iteration, improved)

    def _target_aware_guide_promotion_due(self) -> bool:
        """Whether a productive operator deserves one strong last-mile implementation."""
        target = getattr(self, "target_score", None)
        ratio = getattr(self, "target_aware_guide_promotion_ratio", None)
        if target is None or target <= 0 or ratio is None:
            return False
        tolerance = getattr(self, "target_score_tolerance", 0.0)
        incumbent = self.curve.incumbent + tolerance
        return target * ratio <= incumbent < target

    def _update_adaptation(self, iteration: int, productive: bool) -> None:
        """Respond to whether the last round paid off.

        Parent selection may move away from the incumbent after the configured
        plateau, the novelty ledger reports what has already been reached, and
        -- under ``strategy_when: stalled`` -- this eventually triggers a guide
        plan. A productive round resets both ordinary stagnation and the
        schedule for the next periodic high-effort guide call.
        """
        if productive:
            self._stagnation = 0
            patience = getattr(self, "strategy_escalation_patience", None)
            if patience is not None:
                self._next_strategy_escalation = patience
                self._strategy_escalation_gap = self.strategy_escalation_interval
            self._restore_exploration_ratio()
            return

        self._stagnation += 1
        if self._stagnation >= self.stagnation_patience:
            self._raise_exploration_ratio(iteration)

    def _spent_fraction(self) -> float:
        """How much of the run is consumed, in ``[0, 1]``.

        Against the budget when there is one -- that is literally the ``c/B``
        in the metric -- and against the iteration count otherwise.
        """
        if self.budget:
            return min(1.0, float(self.curve.spent()) / float(self.budget))
        done, total = self._progress
        return min(1.0, done / total) if total > 0 else 0.0

    def _next_candidate_count(self) -> int:
        """How many independent implementation calls the next round should make.

        In fixed mode this is the number of cheap implementations fed by the
        round's one expensive strategy.  Frontload mode remains available for
        experiments and interpolates from ``candidates_first`` to
        ``candidates_last`` over the run.

        These are separate LLM calls, not multiple candidates embedded in one
        response.  That isolates parse failures and lets each implementation
        make an independent draw from the implementation-model pool.
        """
        if getattr(self, "ordered_strategy_racing", False):
            return (
                self.implementations_per_strategy
                if getattr(self, "_ordered_phase", "idle") == "exploit"
                else 1
            )
        if self.schedule == "fixed":
            return self.implementations_per_strategy
        fraction = self._spent_fraction()
        width = self.candidates_first + (self.candidates_last - self.candidates_first) * fraction
        low, high = sorted((self.candidates_first, self.candidates_last))
        return max(low, min(high, int(round(width))))

    def _opening_cheap_probe_due(self) -> bool:
        """Whether the bounded cheap first observation has not run yet."""
        return bool(
            getattr(self, "two_tier", False)
            and getattr(self, "opening_cascade", False)
            and getattr(self, "opening_cheap_candidates", 1) > 0
            and not getattr(self, "_opening_cheap_probe_done", False)
        )

    def _update_opening_probe_retry(self, iteration: int, *, usable: bool) -> None:
        """Retry a broken cheap opening once, but never retry a valid result.

        The retry protects startup from an unlucky parse or evaluator failure.
        A usable low-scoring program is still information,
        so it completes the bounded opening probe and cannot start an
        open-ended cheap retry loop.
        """
        if usable:
            return
        used = getattr(self, "_opening_failure_retries_used", 0)
        limit = getattr(self, "opening_failure_retries", 0)
        if used >= limit:
            return
        self._opening_failure_retries_used = used + 1
        self._opening_cheap_probe_done = False
        logger.info(
            "Iteration %d: cheap opening produced no usable program; scheduling "
            "failure-only retry %d/%d",
            iteration,
            self._opening_failure_retries_used,
            limit,
        )

    def _opening_guide_hedge_due(self) -> bool:
        """Whether the cascade owes its direct strong-model implementation."""
        cheap_stage_complete = bool(
            getattr(self, "_opening_cheap_probe_done", False)
            or getattr(self, "opening_cheap_candidates", 1) == 0
        )
        return bool(
            getattr(self, "two_tier", False)
            and getattr(self, "opening_cascade", False)
            and getattr(self, "opening_direct_guide_hedge", True)
            and cheap_stage_complete
            and not getattr(self, "_opening_guide_hedge_done", False)
            and not self._defer_opening_guide_hedge()
        )

    def _defer_opening_guide_hedge(self) -> bool:
        """Let a near-target cheap trajectory climb until its first stall."""
        target = getattr(self, "target_score", None)
        ratio = getattr(self, "opening_guide_defer_target_ratio", 0.0)
        if target is None or target <= 0 or ratio <= 0:
            return False
        if getattr(self, "_stagnation", 0) > 0:
            return False
        tolerance = getattr(self, "target_score_tolerance", 0.0)
        return self.curve.incumbent + tolerance >= target * ratio

    def _plan_iteration(self, iteration: int) -> Optional[int]:
        """Admitted implementation width for the next round, or None to stop.

        A configured budget is an external resource guard, not part of the
        reward. A narrower round that fits may still be admitted, so desired
        width is shrunk toward the floor before the run is stopped.
        """
        if self._opening_cheap_probe_due():
            wanted = getattr(self, "opening_cheap_candidates", 1)
        elif (
            getattr(self, "ordered_strategy_racing", False)
            and not self._opening_guide_hedge_due()
            and self._strategy_call_due()
        ):
            # A portfolio scout is one logical observation containing K Luna
            # calls, so budget admission must price all pilots up front.
            wanted = self._guide_portfolio_size()
        else:
            wanted = self._next_candidate_count()
        if self.budget is None:
            return wanted

        remaining = self.curve.remaining()
        if remaining <= 0:
            logger.info(
                "Stopping at iteration %d: budget of %g %s is spent (%s)",
                iteration,
                self.budget,
                self.curve.unit,
                self.curve.summary_line(),
            )
            return None

        # Ordered racing compares a complete K-pilot scout or M-wide batch.
        # A narrowed batch would silently change both its ordering and its
        # stall rule, so finite-budget runs admit the whole observation or
        # stop before it. Other modes may still narrow toward one candidate.
        floor = (
            wanted
            if getattr(self, "ordered_strategy_racing", False)
            and not self._opening_cheap_probe_due()
            and not self._opening_guide_hedge_due()
            else 1
        )
        for candidates in range(wanted, floor - 1, -1):
            forecast = self._forecast_cost(candidates)
            if forecast is None or forecast <= remaining:
                if candidates < wanted:
                    logger.info(
                        "Iteration %d narrowed from %d to %d candidate(s): "
                        "only %.6f %s of budget remains",
                        iteration,
                        wanted,
                        candidates,
                        remaining,
                        self.curve.unit,
                    )
                return candidates

        logger.info(
            "Stopping at iteration %d: forecast %.6f %s for the narrowest round exceeds "
            "the remaining %.6f (%s)",
            iteration,
            self._forecast_cost(floor) or 0.0,
            self.curve.unit,
            remaining,
            self.curve.summary_line(),
        )
        return None

    def _forecast_cost(self, candidates: int) -> Optional[float]:
        """Expected cost of a ``candidates``-implementation round.

        The forecast is the historical mean strategy cost when a strategy is
        due, plus the historical mean cost of each requested implementation.
        A rare guide-model implementation is forecast separately from cheap
        implementations. ``None`` means at least one required tier has no
        history yet; its first call is never blocked by a made-up zero estimate.
        """
        strategy_due = self._strategy_call_due()
        guide_implementation_due = self._guide_implementation_forecast_due(strategy_due)
        cheap_candidates = candidates - int(guide_implementation_due)
        guide_impl_costs = getattr(self, "_guide_impl_costs", [])
        # Do not pretend an unknown component is free.  The first complete
        # round is deliberately admitted so both tiers can be priced.
        if (
            (cheap_candidates > 0 and not self._impl_costs)
            or (strategy_due and not self._strategy_costs)
            or (guide_implementation_due and not guide_impl_costs)
        ):
            return None
        forecast = 0.0
        if cheap_candidates > 0:
            forecast += (sum(self._impl_costs) / len(self._impl_costs)) * cheap_candidates
        if guide_implementation_due:
            forecast += sum(guide_impl_costs) / len(guide_impl_costs)
        if strategy_due:
            forecast += sum(self._strategy_costs) / len(self._strategy_costs)
        return forecast

    def _guide_implementation_forecast_due(self, strategy_due: Optional[bool] = None) -> bool:
        """Whether the next round contains one strong-model code candidate."""
        if getattr(self, "_guide_implementation_due", False):
            return True
        if self._opening_guide_hedge_due():
            return True
        if strategy_due is None:
            strategy_due = self._strategy_call_due()
        if strategy_due and self._unguided_guide_hedge_due(
            getattr(self, "_strategy_call_count", 0) + 1
        ):
            return True
        if not getattr(self, "guide_implementation_on_escalation", False):
            return False
        if not getattr(self, "strategy_escalation_reasoning_effort", None):
            return False
        next_escalation = getattr(
            self, "_next_strategy_escalation", self.strategy_escalation_patience
        )
        return bool(strategy_due and self._stagnation >= next_escalation)

    def _unguided_guide_hedge_due(self, call_number: Optional[int] = None) -> bool:
        """Schedule a low-density direct Terra code attempt on long plateaus."""
        interval = getattr(self, "unguided_guide_hedge_interval", 0)
        if interval <= 0:
            return False
        warmup = getattr(self, "unguided_guide_hedge_warmup_calls", 2)
        number = getattr(self, "_strategy_call_count", 0) if call_number is None else call_number
        return number >= warmup and (number - warmup) % interval == 0

    def _strategy_call_due(self) -> bool:
        """Whether switching strategy requires a new expensive guide call."""
        if not self._wants_new_strategy():
            return False
        if getattr(self, "ordered_strategy_racing", False):
            return not getattr(self, "_ordered_exploitation_queue", [])
        if self.strategy_when == "always":
            return True
        if getattr(self, "_strategy_queue", []):
            return False
        return True

    def _pre_strategy_hedge_due(self) -> bool:
        """Whether to try unguided guide code before buying the next portfolio.

        The hedge prompt contains no prose strategy, so generating the next
        portfolio first cannot help it.  Moving the same scheduled call ahead
        of the portfolio lets a successful hedge skip that unused expense;
        after failure, the normal portfolio path still runs with the remaining
        implementation width.
        """
        if not self.two_tier:
            return False
        if self._opening_guide_hedge_due():
            return True
        if self.strategy_when == "always":
            return False
        if getattr(self, "_strategy_queue", []):
            return False
        return self._unguided_guide_hedge_due(getattr(self, "_strategy_call_count", 0) + 1)

    # ==================================================================
    # Novelty ledger
    # ==================================================================

    def _register_solution(self, solution: str, score: float) -> bool:
        """Record an evaluated candidate; return True if it was not seen before."""
        rounded = round(float(score), _SCORE_RESOLUTION)
        self._score_ledger[rounded] = self._score_ledger.get(rounded, 0) + 1
        self._score_ledger.move_to_end(rounded)

        if not solution:
            return False
        digest = hashlib.sha1(re.sub(r"\s+", " ", solution).strip().encode()).hexdigest()
        if digest in self._seen_solutions:
            return False
        self._seen_solutions.add(digest)
        return True

    def _opening_move_instruction(self) -> str:
        """Ask for a finished solution, not an increment, until the seed is beaten.

        While the incumbent is still the seed program, the whole gap above it
        is on the table at full weight, and an incremental edit spends a call
        to capture a fraction of it. This holds past a wasted first call: if
        call 1 produced nothing, call 2 faces exactly the same situation.
        """
        if not self.ambitious_first_call:
            return ""
        points = self.curve.points
        if points and self.curve.incumbent > points[0].incumbent:
            return ""
        return """

# This is the opening attempt

The current solution is a placeholder, so there is nothing worth preserving in
it. Do not make an incremental edit: work out the best complete solution you
can and give it in full. Later iterations will refine whatever you produce, so
aim for the strongest result you can reach now rather than a safe small step.
"""

    def _novelty_ledger_text(self) -> str:
        """Objective values already reached, as a 'do not re-derive' list.

        Repeat counts are the actionable part: telling the model that one
        score has been produced thirty-four times is what makes it stop
        proposing the thirty-fifth variant of the same packing.

        The ledger is input tokens, i.e. cost, so it is withheld while the
        search is still improving -- there is nothing to warn about yet.
        """
        if not self.novelty_memory or not self._score_ledger or not self._stagnation:
            return ""

        recent = list(self._score_ledger.items())[-self.novelty_memory :]
        lines = []
        for score, count in sorted(recent, key=lambda item: item[0], reverse=True):
            repeats = f" (reached {count} times)" if count > 1 else ""
            lines.append(f"- score {score:.6f}{repeats}")

        return f"""

# Already explored -- do not re-derive

These objective values have already been produced by earlier candidates:
{chr(10).join(lines)}

A candidate that lands on one of these values again is worth nothing, however
well written it is. Change the structure of the solution, not its constants.
"""

    # ==================================================================
    # Exploration shift
    # ==================================================================

    def _raise_exploration_ratio(self, iteration: int) -> None:
        """Bias parent selection away from the incumbent while stalled.

        A plateau is attached to the incumbent, so continuing to exploit it
        buys duplicates. ``exploration_ratio`` is not on the database ABC, so
        it is read and written by name; only the live attribute is touched and
        the configured value is restored on the next improvement.
        """
        if (
            self._exploring
            or self.stalled_exploration_ratio is None
            or self._base_exploration_ratio is None
            or self.stalled_exploration_ratio <= self._base_exploration_ratio
        ):
            return
        setattr(self.database, "exploration_ratio", float(self.stalled_exploration_ratio))
        self._exploring = True
        logger.info(
            "Iteration %d: %d non-improving iterations, exploration_ratio %.2f -> %.2f",
            iteration,
            self._stagnation,
            self._base_exploration_ratio,
            self.stalled_exploration_ratio,
        )

    def _restore_exploration_ratio(self) -> None:
        if self._exploring and self._base_exploration_ratio is not None:
            setattr(self.database, "exploration_ratio", self._base_exploration_ratio)
            self._exploring = False

    # ==================================================================
    # Two-tier iteration
    # ==================================================================

    def _wants_new_strategy(self) -> bool:
        """Whether this round should buy a plan from the expensive tier.

        Under the default ``stalled`` policy, the cheap tier runs alone until it
        stops raising the incumbent. Buying the plan on stagnation also
        aims the expensive model at the case it is actually needed for -- the
        cheap tier has run out of ideas, not out of implementation skill. Once
        strategies are active, an improving plan is reused. After its first
        non-improving round, a proven plan gets one guide-model implementation;
        an unproductive plan is replaced immediately.

        ``reuse_on_improvement`` buys the first plan before the cheap bootstrap;
        ``always`` buys a fresh plan every round.
        """
        if not self.two_tier:
            return False
        # The opening stages are deliberately sequential so the cheap point
        # is placed on the cost curve before the direct guide attempt. The
        # second stage uses the existing pre-portfolio hedge path, hence it
        # presents as a strategy transition even though no prose is bought.
        if self._opening_cheap_probe_due():
            return False
        if self._opening_guide_hedge_due():
            return True
        # A successful unguided guide hedge intentionally hands the stronger
        # incumbent to one cheap batch before any new portfolio is bought.
        # This must override ``reuse_on_improvement`` too: there is no active
        # prose strategy in that case, but immediately interpreting that as a
        # request for another guide call repeats the same hedge forever.
        if getattr(self, "_anchor_next_cheap_to_incumbent", False):
            return False
        if getattr(self, "ordered_strategy_racing", False):
            phase = getattr(self, "_ordered_phase", "idle")
            if phase == "pilot":
                # A newly purchased portfolio's pilots normally finish in the
                # same iteration; this state is only transient.
                return False
            if phase == "exploit":
                return (
                    getattr(self, "_ordered_active_root", None) is None or self._barren_rounds > 0
                )
            if self.strategy_when == "always":
                return True
            if self.strategy_when == "reuse_on_improvement":
                return self._strategy is None
            return self._stagnation >= self.strategy_patience
        # A strategy that already proved useful gets one strong-model coding
        # attempt after Luna stalls; do that before retiring or replacing it.
        if self._strategy is not None and getattr(self, "_guide_implementation_due", False):
            return False
        if self.strategy_when == "always":
            return True
        if self.strategy_when == "reuse_on_improvement":
            return self._strategy is None or self._barren_rounds > 0
        if self._strategy is None:
            return self._stagnation >= self.strategy_patience
        return self._barren_rounds >= self.strategy_patience

    def _should_anchor_strategy_parent(self, wants_new_strategy: bool) -> bool:
        """Use the incumbent for guided work and post-hedge cheap exploitation."""
        return self.two_tier and (
            wants_new_strategy
            or self._strategy is not None
            or getattr(self, "_anchor_next_cheap_to_incumbent", False)
        )

    def _search_incumbent_due(self, guided_parent: bool) -> bool:
        """Diversify a guided parent only after consecutive barren rounds."""
        patience = self.search_incumbent_stagnation_patience
        cooldown = getattr(self, "search_incumbent_cooldown", 0)
        return bool(
            guided_parent
            and self.search_incumbent_enabled
            and self._stagnation >= patience
            and (self._stagnation - patience) % (cooldown + 1) == 0
        )

    async def _propose_strategy(
        self,
        iteration: int,
        *,
        raw_parent: Any = None,
        parent: Optional[Program] = None,
        context_dict: Optional[Dict[str, List[Program]]] = None,
    ) -> Optional[Strategy]:
        """Activate the next plan, buying a portfolio only when needed.

        This is the call that buys the next prose portfolio. When
        ``strategy_reference_candidate`` is enabled, each already-paid
        response also carries one executable reference.
        """
        self._pending_strategy_reference = None
        self._pending_strategy_reference_prompt = None
        if self.ordered_strategy_racing:
            ordered = self._activate_next_ordered_strategy(iteration)
            if ordered is not None:
                return ordered
        queued = getattr(self, "_strategy_queue", [])
        if not self.ordered_strategy_racing and queued and self.strategy_when != "always":
            strategy = queued.pop(0)
            strategy.baseline_score = self.curve.incumbent
            logger.info(
                "Iteration %d: activated queued strategy %s (%d remain) — %s",
                iteration,
                strategy.label,
                len(queued),
                strategy.plan.replace("\n", " ")[:160],
            )
            return strategy

        if not self.ordered_strategy_racing:
            replay = self._strategy_replay_when_guide_cannot_amortize(iteration)
            if replay is not None:
                return replay

        if getattr(self, "lazy_strategy_guidance", False):
            await self._generate_strategy_guidance()

        # Direct unit/integration callers may omit the sampled values.  The
        # normal run loop always supplies them so the plan and implementations
        # are anchored to the same parent.
        if parent is None and self.database.programs:
            parent = self.database.get_best_program()
            raw_parent = parent
        program = parent.solution if parent else ""
        score = get_score(parent.metrics) if parent else None

        if parent is None:
            prompt = self.context_builder.build_prompt(current_program=None, context={})
            prompt_context: Dict[str, List[Program]] = {}
        else:
            prompt_context = self._compact_prompt_context(context_dict or {})
            prompt = self._build_prompt(
                current_program=raw_parent or parent,
                context_programs=prompt_context,
                failed_attempts=[],
            )
        portfolio_size = self._guide_portfolio_size()
        include_reference = self._strategy_reference_due()
        target_hint = self._strategy_context_hint(prompt_context)
        target_hint += self._strategy_guidance_hint()
        if self.ordered_strategy_racing and self._strategy_reference_due():
            target_hint += self._ordered_reference_feedback_hint()
        request = strategy_request(
            program,
            score,
            self.ledger,
            metrics=parent.metrics if parent else None,
            target_hint=target_hint,
            count=portfolio_size,
            include_reference=include_reference,
            shared_reference=self.ordered_strategy_racing,
            history_limit=getattr(self, "strategy_history_size", 8),
        )

        cost_before = self.curve.spent()
        try:
            response = await self._call_strategy_llm(prompt["system"], request)
        except Exception as exc:
            spend = self.curve.spent() - cost_before
            if spend > 0:
                self._strategy_costs.append(spend)
            logger.warning("Iteration %d: strategy call failed: %s", iteration, exc)
            return None
        spend = self.curve.spent() - cost_before
        if spend > 0:
            self._strategy_costs.append(spend)

        strategies = parse_strategies(
            response.text or "",
            len(self.ledger.entries) + 1,
            portfolio_size,
        )
        if not strategies:
            logger.warning("Iteration %d: strategy response was unusable", iteration)
            return None
        if include_reference:
            self._pending_strategy_reference = parse_reference_implementation(response.text or "")
            self._pending_strategy_reference_prompt = {
                "system": prompt["system"],
                "user": request,
            }
        self._pending_strategy_reference_requested = include_reference

        self._strategy_call_count += 1

        cost_share = spend / len(strategies)
        for strategy in strategies:
            strategy.cost = cost_share
            strategy.improvement_epsilon = getattr(self, "improvement_epsilon", 1e-9)
            self.ledger.add(strategy)
        strategy, *rest = strategies
        strategy.baseline_score = self.curve.incumbent
        if self.ordered_strategy_racing:
            for item in rest:
                item.baseline_score = self.curve.incumbent
            self._strategy_queue = []
            self._ordered_portfolio_roots = [item.root_index for item in strategies]
            self._ordered_exploitation_queue = []
            self._ordered_active_root = None
            self._ordered_pilot_scores = {}
            self._ordered_phase = "pilot"
            self._ordered_pending_pilots = list(strategies)
        else:
            self._strategy_queue.extend(rest)
        if (
            not self.ordered_strategy_racing
            and getattr(self, "_last_strategy_call_escalated", False)
            and getattr(self, "guide_implementation_on_escalation", False)
        ):
            self._guide_implementation_due = True
            self._guide_implementation_due_reason = "exponentially backed-off long plateau"
            self._guide_implementation_unguided = False
        logger.info(
            "Iteration %d: guide proposed %d strategies for %.6f %s; activating %s — %s",
            iteration,
            len(strategies),
            spend,
            self.curve.unit,
            strategy.label,
            strategy.plan.replace("\n", " ")[:160],
        )
        return strategy

    def _ordered_root_strategy(self, root: int) -> Optional[Strategy]:
        """Return the original Terra strategy for an ordered-racing root."""
        return next(
            (
                strategy
                for strategy in self.ledger.entries
                if strategy.index == root and strategy.root_index == root
            ),
            None,
        )

    def _strategy_guidance_hint(self) -> str:
        """Render the cached EE-local meta-analysis for the strategist."""
        sections = []
        if self._strategy_objective_guidance:
            # First, so the weight/tradeoff decomposition frames the method
            # lists rather than being read as an afterthought.
            sections.append(
                "## Scored objective: weights and tradeoffs\n\n"
                + self._strategy_objective_guidance
            )
        if self._strategy_diverge_guidance:
            sections.append(
                "## Evaluator-aware exploration examples\n\n" + self._strategy_diverge_guidance
            )
        if self._strategy_refine_guidance:
            sections.append(
                "## Evaluator-aware refinement examples\n\n" + self._strategy_refine_guidance
            )
        if not sections:
            return ""
        return (
            "\n\n# Evaluator- and package-aware search guidance\n\n"
            "A guide model generated the following task-specific evidence once after "
            "reading the evaluator and available packages. Use relevant library functions "
            "and evaluator-visible mechanisms when they improve the submitted score. Treat "
            "these as search guidance, not as permission to violate the problem's hard "
            "constraints or the strategy-only response format.\n\n"
            "Target the scored objective as decomposed below, not the nominal domain "
            "goal: pick strategies that move the terms with the most remaining weight, "
            "and do not keep proposing variants of one method family once its term is "
            "near its maximum.\n\n" + "\n\n".join(sections)
        )

    def _ordered_reference_feedback_hint(self) -> str:
        """Return recent portfolio-reference evidence for the next Terra call."""
        feedback = getattr(self, "_ordered_reference_feedback", [])
        if not feedback:
            return ""
        return (
            "\n\n# Recent portfolio reference evaluations\n\n"
            + "\n".join(f"- {item}" for item in feedback[-4:])
            + "\nUse these evaluator results to avoid repeating a broken or regressive "
            "portfolio reference."
        )

    def _strategy_reference_due(self) -> bool:
        """Whether every newly generated strategy portfolio needs a reference."""
        return bool(self.strategy_reference_candidate)

    def _record_strategy_reference_result(
        self,
        iteration: int,
        outcome: Optional[SerializableResult],
        *,
        baseline: float,
        requested: bool,
    ) -> None:
        """Record reference evidence without disabling future references."""
        if not requested:
            return

        improved = False
        if outcome is None:
            summary = "portfolio reference missing from the guide response"
        elif outcome.error or not outcome.child_program_dict:
            summary = f"portfolio reference failed: {self._failure_feedback(outcome)}"
        else:
            reference_score = get_score(outcome.child_program_dict.get("metrics") or {})
            improved = reference_score > baseline + self.improvement_epsilon
            relation = (
                "promoted to shared pilot parent"
                if improved
                else "did not beat the incumbent"
            )
            summary = f"portfolio reference scored {reference_score:.12g} and {relation}"

        feedback = list(getattr(self, "_ordered_reference_feedback", []))
        feedback.append(summary)
        self._ordered_reference_feedback = feedback[-4:]

        if improved:
            self._strategy_reference_failure_streak = 0
            return

        self._strategy_reference_failure_streak = (
            getattr(self, "_strategy_reference_failure_streak", 0) + 1
        )
        self._strategy_reference_disabled = False
        logger.info(
            "Iteration %d: strong model reference did not raise the global incumbent "
            "(%d consecutive miss(es)); reference generation remains enabled",
            iteration,
            self._strategy_reference_failure_streak,
        )

    def _ordered_pilot_sample(
        self,
        sample: Tuple[
            Any,
            Optional[Program],
            Optional[Tuple[str, str]],
            Dict[str, List[Program]],
            List[str],
            List[Tuple[str, str]],
        ],
        reference_outcome: Optional[SerializableResult],
    ) -> Tuple[
        Any,
        Optional[Program],
        Optional[Tuple[str, str]],
        Dict[str, List[Program]],
        List[str],
        List[Tuple[str, str]],
    ]:
        """Promote a better portfolio reference to the shared Luna pilot parent.

        The reference remains an ordinary evaluated candidate and is processed
        later with the rest of the logical iteration. Building the temporary
        parent here lets all K pilots see the same stronger program without
        assigning its gain or source code to any individual strategy.
        """
        if reference_outcome is None or not self._outcome_improves_incumbent(
            reference_outcome
        ):
            return sample

        program_dict = reference_outcome.child_program_dict
        if not program_dict:
            return sample
        program_class = getattr(self.database, "_program_class", Program)
        reference_parent = program_class(**program_dict)
        _raw_parent, parent, _parent_info, context_dict, _context_ids, _context_info = sample

        shared_context = {
            key: [program for program in programs if program.id != reference_parent.id]
            for key, programs in context_dict.items()
        }
        if parent is not None and parent.id != reference_parent.id and not any(
            parent.id == program.id
            for programs in shared_context.values()
            for program in programs
        ):
            shared_context.setdefault("portfolio_origin", []).append(parent)
        shared_context = {
            key: programs for key, programs in shared_context.items() if programs
        }
        context_ids = [
            program.id for programs in shared_context.values() for program in programs
        ]
        context_info = [
            (key, program.id)
            for key, programs in shared_context.items()
            for program in programs
        ]
        logger.info(
            "Ordered racing promotes strong model portfolio reference %s (score %.12g) "
            "as the shared parent for all weak model pilots",
            reference_parent.id[:8],
            get_score(reference_parent.metrics),
        )
        return (
            reference_parent,
            reference_parent,
            ("portfolio_reference", reference_parent.id),
            shared_context,
            context_ids,
            context_info,
        )

    def _activate_next_ordered_strategy(self, iteration: int) -> Optional[Strategy]:
        """Activate the next pilot-ranked strategy, or close the portfolio."""
        if self._ordered_phase != "exploit" or self._ordered_active_root is not None:
            return None
        if not self._ordered_exploitation_queue:
            if self._ordered_portfolio_roots:
                logger.info(
                    "Iteration %d: ordered portfolio %s has no remaining retained "
                    "strategies; buying a new strong model portfolio",
                    iteration,
                    self._ordered_portfolio_roots,
                )
            self._ordered_portfolio_roots = []
            self._ordered_pilot_scores = {}
            self._ordered_phase = "idle"
            return None

        root = self._ordered_exploitation_queue.pop(0)
        source = self._ordered_root_strategy(root)
        if source is None:
            logger.warning("Iteration %d: ordered strategy root #%d is missing", iteration, root)
            return self._activate_next_ordered_strategy(iteration)
        replay = Strategy(
            index=len(self.ledger.entries) + 1,
            title=f"ordered attempt {source.title}",
            plan=source.plan,
            baseline_score=self.curve.incumbent,
            improvement_epsilon=getattr(self, "improvement_epsilon", 1e-9),
            source_strategy_index=root,
            reference_solution=source.reference_solution,
            reference_score=source.reference_score,
            failure_feedback=list(source.failure_feedback),
        )
        self.ledger.add(replay)
        self._ordered_active_root = root
        self._barren_rounds = 0
        logger.info(
            "Iteration %d: ordered racing activates root #%d after weak model pilot score %s "
            "(%d strategy/strategies remain)",
            iteration,
            root,
            (
                "unusable"
                if self._ordered_pilot_scores.get(root) is None
                else f"{self._ordered_pilot_scores[root]:.12g}"
            ),
            len(self._ordered_exploitation_queue),
        )
        return replay

    async def _run_ordered_pilots(
        self,
        iteration: int,
        strategies: List[Strategy],
        sample: Tuple[
            Any,
            Optional[Program],
            Optional[Tuple[str, str]],
            Dict[str, List[Program]],
            List[str],
            List[Tuple[str, str]],
        ],
    ) -> List[Tuple[Strategy, SerializableResult]]:
        """Generate one comparable Luna pilot per strategy from one parent."""
        raw_parent, parent, parent_info, context_dict, context_ids, context_info = sample

        async def pilot(strategy: Strategy) -> SerializableResult:
            return await self._generate_one(
                variant=1,
                variants=1,
                strategy=strategy,
                raw_parent=raw_parent,
                parent=parent,
                parent_info=parent_info,
                context_dict=context_dict,
                context_ids=context_ids,
                context_info=context_info,
                iteration=iteration,
                implementation_model_tier="cheap",
                scouting=True,
            )

        logger.info(
            "Iteration %d: ordered racing launches one weak model pilot for each of %d "
            "strong model strategies from the same parent",
            iteration,
            len(strategies),
        )
        raw_results = await asyncio.gather(
            *(pilot(strategy) for strategy in strategies),
            return_exceptions=True,
        )
        results: List[Tuple[Strategy, SerializableResult]] = []
        for strategy, result in zip(strategies, raw_results):
            if isinstance(result, BaseException):
                outcome = SerializableResult(error=str(result), iteration=iteration)
            else:
                outcome = result
            results.append((strategy, outcome))

        # Keep scoreless children out of the database as well as out of the
        # exploitation queue. Otherwise get_score() could fall back to an
        # average of unrelated metrics and accidentally treat an invalid pilot
        # as objective evidence.
        for _strategy, outcome in results:
            if self._ordered_pilot_score(outcome) is None and not outcome.error:
                outcome.error = "Pilot returned no finite combined_score"
        return results

    @staticmethod
    def _ordered_pilot_score(outcome: SerializableResult) -> Optional[float]:
        """Return a finite combined score, or None for an unusable pilot."""
        if outcome.error or not outcome.child_program_dict:
            return None
        metrics = outcome.child_program_dict.get("metrics") or {}
        if "combined_score" not in metrics:
            return None
        try:
            score = float(metrics["combined_score"])
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) else None

    def _finish_ordered_pilots(
        self,
        iteration: int,
        pilots: List[Tuple[Strategy, SerializableResult]],
        *,
        reference_outcome: Optional[SerializableResult],
        baseline: float,
        implementation_spend: float,
    ) -> None:
        """Record pilot evidence and build the deterministic attempt order."""
        if not pilots:
            self._ordered_phase = "idle"
            self._strategy = None
            return

        per_pilot_spend = max(0.0, implementation_spend) / len(pilots)
        pilot_scores: Dict[int, Optional[float]] = {}
        candidate_rows: List[Tuple[float, Strategy]] = []
        for strategy, outcome in pilots:
            strategy.rounds += 1
            strategy.attempts += 1
            strategy.implementation_spend += per_pilot_spend
            strategy.last_used_iteration = iteration
            score = self._ordered_pilot_score(outcome)
            if score is None:
                strategy.record_failure(self._failure_feedback(outcome))
                pilot_scores[strategy.root_index] = None
                continue
            strategy.implementations += 1
            strategy.best_score = (
                score if strategy.best_score is None else max(strategy.best_score, score)
            )
            pilot_scores[strategy.root_index] = score
            candidate_rows.append((score, strategy))

        if candidate_rows:
            best_score, best_strategy = max(
                candidate_rows,
                key=lambda item: (item[0], -item[1].root_index),
            )
            if best_score > baseline + self.improvement_epsilon:
                gain = max(0.0, self.curve.incumbent - baseline)
                scale = max(abs(baseline), abs(self.curve.incumbent), 1e-12)
                best_strategy.cumulative_gain += gain
                best_strategy.cumulative_relative_gain += gain / scale
                best_strategy.productive_rounds += 1

        original_position = {
            strategy.root_index: position for position, (strategy, _outcome) in enumerate(pilots)
        }
        ordered = sorted(
            (strategy for strategy, _outcome in pilots),
            key=lambda strategy: (
                pilot_scores.get(strategy.root_index) is not None,
                (
                    pilot_scores[strategy.root_index]
                    if pilot_scores.get(strategy.root_index) is not None
                    else float("-inf")
                ),
                -original_position[strategy.root_index],
            ),
            reverse=True,
        )
        retained = ordered
        pruned: List[Strategy] = []
        if self.ordered_pilot_pruning:
            # Do not threshold positive scores by an absolute or relative
            # margin: score scales differ by task, and a weak first
            # implementation can still identify a productive strategy. Keep
            # every positive pilot and let ordered racing use its score only
            # to decide execution order.
            retained = [
                strategy
                for strategy in ordered
                if pilot_scores.get(strategy.root_index) is not None
                and (
                    not self.ordered_pilot_prune_nonpositive
                    or pilot_scores[strategy.root_index] > 0.0
                )
            ]
            retained_roots = {strategy.root_index for strategy in retained}
            pruned = [
                strategy for strategy in ordered if strategy.root_index not in retained_roots
            ]
        self._ordered_pilot_scores = pilot_scores
        self._ordered_exploitation_queue = [strategy.root_index for strategy in retained]
        self._ordered_active_root = None
        self._ordered_phase = "exploit"
        self._strategy = None
        self._barren_rounds = 0
        logger.info(
            "Iteration %d: weak model pilot ranking is %s; strong model reference scores are "
            "recorded but do not affect this order",
            iteration,
            [
                (
                    strategy.root_index,
                    pilot_scores.get(strategy.root_index),
                )
                for strategy in ordered
            ],
        )
        if self.ordered_pilot_pruning:
            logger.info(
                "Iteration %d: pilot pruning keeps roots %s and discards %s because their "
                "pilot was unusable or non-positive",
                iteration,
                [strategy.root_index for strategy in retained],
                [
                    (strategy.root_index, pilot_scores.get(strategy.root_index))
                    for strategy in pruned
                ],
            )

    async def _evaluate_strategy_reference(
        self,
        iteration: int,
        strategy: Optional[Strategy],
        sample: Tuple[
            Any,
            Optional[Program],
            Optional[Tuple[str, str]],
            Dict[str, List[Program]],
            List[str],
            List[Tuple[str, str]],
        ],
        candidate_count: int,
    ) -> Optional[SerializableResult]:
        """Evaluate code included in the already-paid strategy response.

        ``strategy`` is ``None`` for ordered racing, where the executable is a
        portfolio-level candidate rather than an implementation of strategy 1.
        """
        reference = self._pending_strategy_reference
        prompt = self._pending_strategy_reference_prompt or {"system": "", "user": ""}
        requested = bool(getattr(self, "_pending_strategy_reference_requested", False))
        baseline = self.curve.incumbent
        self._pending_strategy_reference = None
        self._pending_strategy_reference_prompt = None
        self._pending_strategy_reference_requested = False

        def finish(
            outcome: Optional[SerializableResult],
        ) -> Optional[SerializableResult]:
            self._record_strategy_reference_result(
                iteration,
                outcome,
                baseline=baseline,
                requested=requested,
            )
            return outcome

        if not reference:
            return finish(None)

        raw_parent, parent, parent_info, context_dict, context_ids, context_info = sample
        child_solution = parse_full_rewrite(reference, self.language)
        if not child_solution:
            return finish(SerializableResult(
                error="Guide strategy reference contained no complete program",
                prompt=prompt,
                llm_response=reference,
                iteration=iteration,
            ))
        child_solution = child_solution.strip()
        if parent is not None and (
            child_solution == parent.solution.strip()
            or len(child_solution) < _RECOVERY_MIN_RATIO * len(parent.solution)
        ):
            return finish(SerializableResult(
                error="Guide strategy reference was unchanged or only a code fragment",
                prompt=prompt,
                llm_response=reference,
                iteration=iteration,
            ))

        if strategy is not None:
            strategy.reference_solution = child_solution
        reference_label = strategy.label if strategy is not None else "ordered portfolio"
        outcome = await self._evaluate_candidate(
            candidate_index=0,
            candidate_count=candidate_count,
            candidate_response=reference,
            child_solution=child_solution,
            changes_summary=f"Guide reference for {reference_label}",
            parent=parent,
            parent_info=parent_info,
            context_ids=context_ids,
            context_info=context_info,
            prompt=prompt,
            iteration=iteration,
            iteration_start=time.time(),
            llm_generation_time=0.0,
            strategy=strategy,
            implementation_model_tier=(
                "guide_strategy_reference"
                if strategy is not None
                else "guide_portfolio_reference"
            ),
        )
        if strategy is not None and not outcome.error and outcome.child_program_dict:
            strategy.reference_score = get_score(outcome.child_program_dict.get("metrics") or {})
        return finish(outcome)

    def _guide_portfolio_size(self) -> int:
        """Choose how many plans to buy without delaying or wasting them.

        A genuinely opening call buys one plan so the performance curve can
        move early. A call made after the cheap bootstrap, and later refreshes,
        amortize the guide over a portfolio capped by the iterations left.
        ``always`` cannot consume a queue, so it also buys exactly one plan.
        """
        if getattr(self, "ordered_strategy_racing", False):
            # Scouting is one atomic K-pilot observation. Narrowing K to the
            # number of outer-loop iterations would change the experiment
            # merely because its pilots run concurrently inside one round.
            # The first portfolio may intentionally be broader than refreshes;
            # the ledger, not curve length, identifies that first purchase.
            return (
                self.initial_strategies_per_guide_call
                if not self.ledger.entries
                else self.strategies_per_guide_call
            )
        if self.strategy_when == "always":
            return 1
        # A genuinely opening guide call stays narrow so it can move the curve
        # quickly. In ``stalled`` mode Luna may already have produced several
        # observed rounds before Terra is needed; that is a plateau call, not
        # an opening call, so buy the full amortized portfolio immediately.
        curve_points = getattr(getattr(self, "curve", None), "points", [])
        opening_call = not self.ledger.entries and len(curve_points) <= 1
        if opening_call:
            wanted = self.initial_strategies_per_guide_call
        else:
            wanted = self.strategies_per_guide_call
        done, total = self._progress
        remaining = max(1, total - done) if total > 0 else wanted
        return min(wanted, remaining)

    def _strategy_replay_when_guide_cannot_amortize(self, iteration: int) -> Optional[Strategy]:
        """Replay the best tested plan when a finite run has only one round left.

        The replay is a new ledger entry because its baseline and outcome
        belong to the current incumbent. No guide cost is assigned to it.
        """
        if self.strategy_when == "always":
            return None
        done, total = self._progress
        remaining = total - done if total > 0 else self.min_guide_amortization_iterations
        tried = [strategy for strategy in self.ledger.entries if strategy.rounds > 0]
        if remaining >= self.min_guide_amortization_iterations or not tried:
            return None

        source = max(
            tried,
            key=lambda strategy: (
                float("-inf") if strategy.best_score is None else strategy.best_score,
                strategy.implementations / max(1, strategy.attempts),
                strategy.index,
            ),
        )
        replay = Strategy(
            index=len(self.ledger.entries) + 1,
            title=f"replay {source.title}",
            plan=source.plan,
            baseline_score=self.curve.incumbent,
            improvement_epsilon=getattr(self, "improvement_epsilon", 1e-9),
            source_strategy_index=source.root_index,
        )
        self.ledger.add(replay)
        logger.info(
            "Iteration %d: %d iteration(s) remain, so the guide call cannot be "
            "amortized; replaying %s",
            iteration,
            remaining,
            source.label,
        )
        return replay

    def _entry_scheduler_gain(self, strategy: Strategy) -> float:
        """Raw global combined-score gain credited to this operator."""
        if strategy.cumulative_gain > 0:
            return strategy.cumulative_gain
        # Old checkpoints may only contain the earlier relative-gain field.
        if strategy.cumulative_relative_gain > 0:
            return strategy.cumulative_relative_gain
        return 0.0

    def _scheduler_attempts(self, entries: List[Strategy]) -> int:
        return sum(max(0, int(entry.attempts)) for entry in entries)

    def _scheduler_implementations(self, entries: List[Strategy]) -> int:
        return sum(max(0, int(entry.implementations)) for entry in entries)

    def _compact_prompt_context(
        self, context_dict: Dict[str, List[Program]]
    ) -> Dict[str, List[Program]]:
        """Bound archive code while preserving sampled diversity and metadata.

        The incumbent is sent in full elsewhere. Context programs are evidence
        for alternative structure, so a labelled head/signature/tail excerpt
        retains their useful role without making every later Luna call scale
        with the accumulated size of several evolved programs.
        """
        max_programs = getattr(self, "prompt_context_max_programs", 0)
        if max_programs <= 0 or not context_dict:
            return {}
        max_chars = getattr(self, "prompt_context_max_chars", 0)
        compact: Dict[str, List[Program]] = {}
        used = 0
        depth = 0
        groups = [(label, programs) for label, programs in context_dict.items() if programs]
        while used < max_programs:
            added = False
            for label, programs in groups:
                if used >= max_programs:
                    break
                if depth >= len(programs):
                    continue
                program = programs[depth]
                if max_chars > 0 and len(program.solution) > max_chars:
                    record = program.to_dict()
                    record["solution"] = self._structural_program_excerpt(
                        program.solution, max_chars
                    )
                    rendered = Program.from_dict(record)
                else:
                    rendered = program
                compact.setdefault(label, []).append(rendered)
                used += 1
                added = True
            if not added:
                break
            depth += 1
        return compact

    @staticmethod
    def _structural_program_excerpt(solution: str, max_chars: int) -> str:
        """Return a clearly non-executable structural excerpt within a hard cap."""
        if max_chars <= 0 or len(solution) <= max_chars:
            return solution
        marker = (
            "# STRUCTURAL CONTEXT EXCERPT ONLY; omitted code is not executable.\n"
            "# Use its mechanisms as evidence, but implement against the full current parent.\n"
        )
        separator = "\n# ... omitted ...\n"
        allowance = max(0, max_chars - len(marker) - len(separator))
        if allowance <= 32:
            return marker[:max_chars]

        signature_lines = []
        for line in solution.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("def ", "async def ", "class ", "@")):
                signature_lines.append(line)
        signatures = "\n".join(signature_lines)
        middle = f"\n# ... omitted; signatures ...\n{signatures}\n" if signatures else ""
        middle = middle[: allowance // 3]
        remaining = max(0, allowance - len(middle))
        head_size = (remaining * 2) // 3
        tail_size = remaining - head_size
        head = solution[:head_size]
        tail = solution[-tail_size:] if tail_size else ""
        return marker + head + middle + separator + tail

    @staticmethod
    def _strategy_context_hint(context_dict: Dict[str, List[Program]]) -> str:
        """Render the sampled implementation context for the strategist.

        The normal implementation prompt already contains these programs.  A
        compact, explicit rendering here prevents the strategy tier from
        designing in isolation while still ending its prompt with the
        strategy-only response contract.
        """
        programs = [program for group in context_dict.values() for program in group]
        if not programs:
            return ""

        sections = [
            "\n\n# Other sampled programs\n\n"
            "These are the same alternatives the implementers will see. Use them as "
            "evidence for the strategy portfolio for the current parent above."
        ]
        for index, context_program in enumerate(programs, start=1):
            sections.append(
                f"\n\n## Context {index} (score "
                f"{get_score(context_program.metrics):.6f})\n\n"
                f"```\n{context_program.solution}\n```"
            )
        return "".join(sections)

    async def _call_strategy_llm(self, system_message: str, user_message: str) -> LLMResponse:
        """Route the strategy call to the guide model pool (the expensive tier)."""
        kwargs: Dict[str, Any] = {}
        effort = self._current_strategy_reasoning_effort()
        if effort:
            kwargs["reasoning_effort"] = effort
        return await self.guide_llms.generate(
            system_message, [{"role": "user", "content": user_message}], **kwargs
        )

    async def _call_guide_implementation_llm(
        self, system_message: str, user_message: str
    ) -> LLMResponse:
        """Use the guide pool for one rare long-plateau implementation."""
        kwargs: Dict[str, Any] = {}
        effort = (
            getattr(self, "opening_guide_reasoning_effort", None)
            if getattr(self, "_opening_guide_hedge_active", False)
            else self.guide_implementation_reasoning_effort
        )
        if effort:
            kwargs["reasoning_effort"] = effort
        return await self.guide_llms.generate(
            system_message, [{"role": "user", "content": user_message}], **kwargs
        )

    def _current_strategy_reasoning_effort(self) -> Optional[str]:
        """Use stronger guide reasoning with backoff during a long plateau."""
        self._last_strategy_call_escalated = False
        if not self.ledger.entries and self.initial_strategy_reasoning_effort:
            return self.initial_strategy_reasoning_effort
        next_escalation = getattr(
            self, "_next_strategy_escalation", self.strategy_escalation_patience
        )
        if self.strategy_escalation_reasoning_effort and self._stagnation >= next_escalation:
            gap = getattr(self, "_strategy_escalation_gap", self.strategy_escalation_interval)
            while next_escalation <= self._stagnation:
                next_escalation += gap
                gap = max(gap + 1, int(round(gap * self.strategy_escalation_backoff)))
            self._next_strategy_escalation = next_escalation
            self._strategy_escalation_gap = gap
            self._last_strategy_call_escalated = True
            logger.info(
                "Strategy guide effort escalated %s -> %s after %d stagnant iteration(s); "
                "next escalation at %d (gap %d)",
                self.strategy_reasoning_effort or "model default",
                self.strategy_escalation_reasoning_effort,
                self._stagnation,
                next_escalation,
                gap,
            )
            return self.strategy_escalation_reasoning_effort
        return self.strategy_reasoning_effort

    async def _implement_strategy(
        self,
        iteration: int,
        strategy: Optional[Strategy],
        variants: int,
        *,
        independent: bool = False,
        sample: Optional[
            Tuple[
                Any,
                Optional[Program],
                Optional[Tuple[str, str]],
                Dict[str, List[Program]],
                List[str],
                List[Tuple[str, str]],
            ]
        ] = None,
    ) -> List[Tuple[int, SerializableResult]]:
        """Turn one plan into ``variants`` independently evaluated programs.

        ``strategy`` is None in single-tier mode (``two_tier: false``), in which
        case the cheap model generates straight from the parent -- the ablation
        that separates a plan's contribution from simply drawing more samples.

        With adaptive racing, a small pilot group is evaluated first and the
        remaining calls run concurrently only if at least one pilot is usable
        and near the incumbent. Fixed-width mode runs every call concurrently.
        """
        sample = sample or self._sample_parent_and_context()
        raw_parent, parent, parent_info, context_dict, context_ids, context_info = sample
        self._last_guide_implementation_spend = 0.0
        self._last_guide_implementation_count = 0
        self._last_unguided_guide_count = 0
        evaluation_feedback: List[str] = []

        async def generate(
            variant: int,
            *,
            model_tier: str = "cheap",
            generation_strategy: Optional[Strategy] = strategy,
        ) -> SerializableResult:
            return await self._generate_one(
                variant=variant,
                variants=variants,
                strategy=generation_strategy,
                raw_parent=raw_parent,
                parent=parent,
                parent_info=parent_info,
                context_dict=context_dict,
                context_ids=context_ids,
                context_info=context_info,
                iteration=iteration,
                implementation_model_tier=model_tier,
                evaluation_feedback=evaluation_feedback,
            )

        outcomes: List[Tuple[int, SerializableResult]] = []
        first_cheap_variant = 1
        guide_due = getattr(self, "_guide_implementation_due", False)
        unguided_due = getattr(self, "_guide_implementation_unguided", False)
        if guide_due and (strategy is not None or unguided_due):
            # Run this call alone so its spend can be learned separately for
            # later budget admission. All remaining implementations can still
            # run concurrently through the cheap pool.
            due_reason = self._guide_implementation_due_reason or "scheduled insurance"
            unguided = unguided_due
            self._guide_implementation_due = False
            self._guide_implementation_due_reason = None
            self._guide_implementation_unguided = False
            if not unguided:
                strategy.guide_implementation_attempts += 1
            logger.info(
                "Iteration %d: guide implementation insurance (%s) routes candidate 1/%d "
                "through the guide model%s",
                iteration,
                due_reason,
                variants,
                " without a strategy constraint" if unguided else f" for strategy {strategy.label}",
            )
            strong_cost_before = self.curve.spent()
            try:
                strong_result: Any = await generate(
                    1,
                    model_tier="guide",
                    generation_strategy=None if unguided else strategy,
                )
            except BaseException as exc:
                strong_result = exc
            strong_spend = max(0.0, self.curve.spent() - strong_cost_before)
            self._last_guide_implementation_spend = strong_spend
            self._last_guide_implementation_count = 1
            self._last_unguided_guide_count = int(unguided)
            if strong_spend > 0:
                self._guide_impl_costs.append(strong_spend)
            outcomes.extend(self._number_outcomes([strong_result], iteration, start=1))
            if self._outcome_improves_incumbent(outcomes[-1][1]):
                logger.info(
                    "Iteration %d: guide implementation raised the incumbent; "
                    "ending the batch before buying weak model calls",
                    iteration,
                )
                return outcomes
            first_cheap_variant = 2

        if getattr(self, "sequential_implementation_feedback", False) and not independent:
            consecutive_failures = 0
            for variant in range(first_cheap_variant, variants + 1):
                results = await asyncio.gather(generate(variant), return_exceptions=True)
                numbered = self._number_outcomes(results, iteration, start=variant)
                outcomes.extend(numbered)
                result = numbered[0][1]
                consecutive_failures = consecutive_failures + 1 if result.error else 0
                if consecutive_failures >= max(1, int(self._setting("sequential_failure_limit"))):
                    logger.info(
                        "Iteration %d: stopping sequential batch after %d consecutive "
                        "invalid implementations; saving %d call(s)",
                        iteration, consecutive_failures, variants - variant,
                    )
                    break
                if self._outcome_improves_incumbent(result):
                    logger.info(
                        "Iteration %d: sequential candidate %d raised the incumbent; "
                        "saving %d stale-parent call(s)", iteration, variant, variants - variant,
                    )
                    break
                if (
                    not result.error
                    and self._setting("sequential_parallel_after_valid_miss")
                    and variant < variants
                ):
                    remainder_start = variant + 1
                    logger.info(
                        "Iteration %d: candidate %d was valid but did not improve; "
                        "releasing the remaining %d candidate(s) concurrently",
                        iteration, variant, variants - variant,
                    )
                    remainder = await asyncio.gather(
                        *(generate(index) for index in range(remainder_start, variants + 1)),
                        return_exceptions=True,
                    )
                    outcomes.extend(
                        self._number_outcomes(remainder, iteration, start=remainder_start)
                    )
                    return outcomes
                if result.error:
                    feedback = self._failure_feedback(result)
                else:
                    child = result.child_program_dict or {}
                    feedback = (
                        f"Evaluated metrics: {child.get('metrics', {})}. "
                        f"Global score to beat: {self.curve.incumbent:.12g}. "
                        "This attempt did not improve the global best."
                    )
                # Include the actual patch so the next call can diagnose it,
                # while explicitly keeping its edit target at the original parent.
                failed_solution = (result.child_program_dict or {}).get("solution", "")
                if result.error and failed_solution and len(failed_solution) <= 24000:
                    # A patch is an instruction, not the program that actually
                    # failed. Expose the evaluated code for coherent repair of
                    # definitions AND call sites, without inserting invalid
                    # candidates into the population or changing their parent.
                    evaluation_feedback.append(
                        f"Attempt {variant}: {feedback}\nActual evaluated failing program:\n"
                        f"```\n{failed_solution}\n```\n"
                        "Repair this program coherently: check every affected definition "
                        "and call site. Return the COMPLETE corrected program, not a diff "
                        "against this failing program. The original parent remains the "
                        "only valid SEARCH/REPLACE target."
                    )
                else:
                    evaluation_feedback.append(
                        f"Attempt {variant}: {feedback}\nPrevious response:\n"
                        + (result.llm_response or "")[:12000]
                    )
            return outcomes

        if (
            independent
            or getattr(self, "ordered_strategy_racing", False)
            or not self.adaptive_implementation_racing
            or variants <= 1
        ):
            results = await asyncio.gather(
                *(generate(variant) for variant in range(first_cheap_variant, variants + 1)),
                return_exceptions=True,
            )
            outcomes.extend(self._number_outcomes(results, iteration, start=first_cheap_variant))
            return outcomes

        if strategy is None:
            if not getattr(self, "adaptive_unguided_racing", False):
                results = await asyncio.gather(
                    *(generate(variant) for variant in range(first_cheap_variant, variants + 1)),
                    return_exceptions=True,
                )
                outcomes.extend(
                    self._number_outcomes(results, iteration, start=first_cheap_variant)
                )
                return outcomes

            # A direct cheap climb has no paid strategy to protect, so its
            # admission rule can be lossless: try one candidate first and
            # stop only on a real global gain. Failure or a non-improving
            # result launches the entire remainder, retaining fixed-width
            # exploration exactly when it is needed.
            pilot_result = await asyncio.gather(
                generate(first_cheap_variant),
                return_exceptions=True,
            )
            pilot_outcomes = self._number_outcomes(
                pilot_result, iteration, start=first_cheap_variant
            )
            outcomes.extend(pilot_outcomes)
            if any(self._outcome_improves_incumbent(outcome) for _, outcome in pilot_outcomes):
                saved = variants - first_cheap_variant
                logger.info(
                    "Iteration %d: direct cheap pilot raised the incumbent; ending "
                    "before buying %d stale-parent candidate(s)",
                    iteration,
                    saved,
                )
                return outcomes

            remainder_start = first_cheap_variant + 1
            results = await asyncio.gather(
                *(generate(variant) for variant in range(remainder_start, variants + 1)),
                return_exceptions=True,
            )
            outcomes.extend(self._number_outcomes(results, iteration, start=remainder_start))
            return outcomes

        cheap_slots = variants - first_cheap_variant + 1
        pilot_slots = self._adaptive_pilot_count(strategy, cheap_slots)
        pilot_end = first_cheap_variant + pilot_slots - 1
        pilot_result = await asyncio.gather(
            *(generate(variant) for variant in range(first_cheap_variant, pilot_end + 1)),
            return_exceptions=True,
        )
        pilot_outcomes = self._number_outcomes(pilot_result, iteration, start=first_cheap_variant)
        outcomes.extend(pilot_outcomes)
        if getattr(self, "stop_racing_on_improvement", False) and any(
            self._outcome_improves_incumbent(outcome) for _, outcome in pilot_outcomes
        ):
            logger.info(
                "Iteration %d: strategy %s pilot raised the incumbent; ending the "
                "batch before buying %d stale-parent implementation(s)",
                iteration,
                strategy.label,
                max(0, variants - pilot_end),
            )
            return outcomes
        if not any(self._pilot_is_promising(outcome, strategy) for _, outcome in pilot_outcomes):
            logger.info(
                "Iteration %d: strategy %s stopped after %d pilot(s); "
                "no expansion to %d candidates",
                iteration,
                strategy.label,
                pilot_slots,
                variants,
            )
            return outcomes

        remaining = await asyncio.gather(
            *(generate(variant) for variant in range(pilot_end + 1, variants + 1)),
            return_exceptions=True,
        )
        outcomes.extend(self._number_outcomes(remaining, iteration, start=pilot_end + 1))
        return outcomes

    @staticmethod
    def _failure_feedback(outcome: SerializableResult) -> str:
        """Compact an implementation failure for strategy and guide prompts."""
        error = outcome.error or "Implementation failed"
        if "diff" in error.lower() and outcome.llm_response:
            explanation = re.sub(r"\s+", " ", outcome.llm_response).strip()[:320]
            if explanation:
                return f"{error}: {explanation}"
        return error

    @staticmethod
    def _number_outcomes(
        results: List[Any], iteration: int, *, start: int
    ) -> List[Tuple[int, SerializableResult]]:
        outcomes: List[Tuple[int, SerializableResult]] = []
        for variant, result in enumerate(results, start=start):
            if isinstance(result, BaseException):
                outcomes.append(
                    (variant, SerializableResult(error=str(result), iteration=iteration))
                )
            else:
                outcomes.append((variant, result))
        return outcomes

    def _adaptive_pilot_count(self, strategy: Strategy, available: int) -> int:
        """Use one pilot only for a historically productive, reliable operator."""
        configured = min(max(1, self.pilot_candidates), max(1, available))
        if configured <= 1 or not getattr(self, "adaptive_pilot_sizing", False):
            return configured

        entries = [
            entry
            for entry in self.ledger.entries
            if entry.root_index == strategy.root_index and entry.rounds > 0
        ]
        attempts = self._scheduler_attempts(entries)
        usable = self._scheduler_implementations(entries)
        gain = sum(self._entry_scheduler_gain(entry) for entry in entries)
        if attempts < getattr(self, "reliable_pilot_min_attempts", 1) or gain <= 0:
            return configured
        usable_ratio = usable / attempts if attempts else 0.0
        if usable_ratio < getattr(self, "reliable_pilot_usable_ratio", 1.0):
            return configured
        return 1

    def _pilot_is_promising(self, outcome: SerializableResult, strategy: Strategy) -> bool:
        """Gate the rest of a cheap batch on one usable near-incumbent pilot."""
        if outcome.error or not outcome.child_program_dict:
            return False
        score = get_score(outcome.child_program_dict.get("metrics") or {})
        baseline = self.curve.incumbent
        if baseline <= 0:
            return score >= baseline
        # Evaluators routinely return the same mathematical score with a few
        # last-bit differences. Use the controller's score resolution so an
        # equal candidate is not rejected as worse by ~1e-16.
        return round(score, _SCORE_RESOLUTION) >= round(
            baseline * self.pilot_score_ratio, _SCORE_RESOLUTION
        )

    def _outcome_improves_incumbent(self, outcome: SerializableResult) -> bool:
        """Whether an already evaluated candidate bought a real global gain."""
        if outcome.error or not outcome.child_program_dict:
            return False
        score = get_score(outcome.child_program_dict.get("metrics") or {})
        return score > self.curve.incumbent + self.improvement_epsilon

    async def _generate_one(
        self,
        *,
        variant: int,
        variants: int,
        strategy: Optional[Strategy],
        raw_parent: Any,
        parent: Optional[Program],
        parent_info: Optional[Tuple[str, str]],
        context_dict: Dict[str, List[Program]],
        context_ids: List[str],
        context_info: List[Tuple[str, str]],
        iteration: int,
        failed_attempts: Optional[List[Dict[str, Any]]] = None,
        implementation_model_tier: str = "cheap",
        scouting: bool = False,
        evaluation_feedback: Optional[List[str]] = None,
    ) -> SerializableResult:
        """One model call: implement the plan, then evaluate the result."""
        iteration_start = time.time()

        if parent is None:
            prompt = self.context_builder.build_prompt(current_program=None, context={})
        else:
            prompt = self._build_prompt(
                current_program=raw_parent,
                context_programs=self._compact_prompt_context(context_dict),
                failed_attempts=failed_attempts or [],
            )
        if strategy is not None:
            prompt["user"] += implementation_instruction(
                strategy,
                variant,
                variants,
                scouting=scouting,
                sequential=getattr(self, "sequential_implementation_feedback", False),
            )
        prompt["user"] += self._opening_move_instruction()
        prompt["user"] += self._novelty_ledger_text()
        if evaluation_feedback:
            prompt["user"] += (
                "\n\n# Evaluated attempts from this same parent\n\n"
                + "\n\n".join(evaluation_feedback[-2:])
                + "\nDiagnose the evidence and correct the implementation, rather than "
                "repeating an unsuccessful patch. For a timeout, reduce work or replace "
                "the costly stage; do not append another optimizer. All SEARCH edits "
                "still target the ORIGINAL current parent shown above, not a previous "
                "attempt. Preserve the strategy's actual gain mechanism."
            )

        if self.feedback_reader:
            self.feedback_reader.set_current_prompt(prompt["system"])
            feedback = self.feedback_reader.read()
            if feedback:
                prompt = self.feedback_reader.apply_feedback(prompt)
                self.feedback_reader.log_usage(iteration, feedback, self.feedback_reader.mode)

        llm_start = time.time()
        try:
            if implementation_model_tier == "guide":
                if strategy is None:
                    prompt["user"] += (
                        "\n\n# Strong-model baseline hedge\n\n"
                        "Directly produce and internally verify the strongest complete, "
                        "executable improvement you can. You are not constrained by the prose "
                        "strategies used by the cheap-model candidates in this iteration.\n"
                    )
                else:
                    prompt["user"] += (
                        "\n\n# Strong-model checkpoint\n\n"
                        "Implement and internally verify this strategy as a complete, executable "
                        "candidate. Do not return another prose strategy.\n"
                    )
                response = await self._call_guide_implementation_llm(
                    prompt["system"], prompt["user"]
                )
            else:
                response = await self._call_llm(prompt["system"], prompt["user"])
        except Exception as exc:
            return SerializableResult(
                error=f"Implementation call failed: {exc}", iteration=iteration
            )
        llm_generation_time = time.time() - llm_start

        if not response.text:
            return SerializableResult(
                error="Empty implementation response", prompt=prompt, iteration=iteration
            )

        child_solution: Optional[str]
        changes_summary: Optional[str]
        parse_error: Optional[str]
        if parent is None:
            child_solution = parse_full_rewrite(response.text, self.language)
            changes_summary = (
                f"From scratch under strategy {strategy.label}"
                if strategy is not None
                else "Generated from scratch"
            )
            parse_error = None if child_solution else "No valid solution in response"
        else:
            child_solution, changes_summary, parse_error = self._parse_llm_response(
                response.text, parent.solution, iteration, attempt=1, retry_times=1
            )
            if parse_error or not child_solution:
                recovered = self._recover_full_rewrite(response.text, parent.solution)
                if recovered:
                    child_solution, parse_error = recovered, None
                    changes_summary = "Full rewrite recovered from an unusable diff"

        if child_solution and len(child_solution) > self.config.max_solution_length:
            parse_error = (
                "Generated solution exceeds maximum length "
                f"({len(child_solution)} > {self.config.max_solution_length})"
            )
            child_solution = None

        if parse_error or not child_solution:
            return SerializableResult(
                error=parse_error or "Implementation parsing failed",
                prompt=prompt,
                llm_response=response.text,
                iteration=iteration,
            )

        return await self._evaluate_candidate(
            candidate_index=variant,
            candidate_count=variants,
            candidate_response=response.text,
            child_solution=child_solution,
            changes_summary=changes_summary,
            parent=parent,
            parent_info=parent_info,
            context_ids=context_ids,
            context_info=context_info,
            prompt=prompt,
            iteration=iteration,
            iteration_start=iteration_start,
            llm_generation_time=llm_generation_time,
            strategy=strategy,
            implementation_model_tier=implementation_model_tier,
        )

    def _sample_parent_and_context(
        self,
        *,
        parent_mode: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> Tuple[
        Any,
        Optional[Program],
        Optional[Tuple[str, str]],
        Dict[str, List[Program]],
        List[str],
        List[Tuple[str, str]],
    ]:
        """Sample once and normalize the database's plain/dict return formats."""
        if not self.database.programs:
            return None, None, None, {}, [], []

        sample_kwargs: Dict[str, Any] = {
            "num_context_programs": self.num_context_programs,
        }
        if parent_mode is not None:
            sample_kwargs["parent_mode"] = parent_mode
        if parent_id is not None:
            sample_kwargs["parent_id"] = parent_id
        raw_parent, raw_context = self.database.sample(**sample_kwargs)
        if isinstance(raw_parent, dict):
            if len(raw_parent) != 1:
                raise ValueError(f"sample() must return one parent, got {len(raw_parent)}")
            parent_key, parent = next(iter(raw_parent.items()))
        else:
            parent_key, parent = "", raw_parent

        context_dict = raw_context if isinstance(raw_context, dict) else {"": raw_context}
        context_ids = [p.id for programs in context_dict.values() for p in programs]
        context_info = [(key, p.id) for key, programs in context_dict.items() for p in programs]
        return (
            raw_parent,
            parent,
            (parent_key, parent.id),
            context_dict,
            context_ids,
            context_info,
        )

    def _anchor_sample_to_incumbent(
        self,
        sample: Tuple[
            Any,
            Optional[Program],
            Optional[Tuple[str, str]],
            Dict[str, List[Program]],
            List[str],
            List[Tuple[str, str]],
        ],
    ) -> Tuple[
        Any,
        Optional[Program],
        Optional[Tuple[str, str]],
        Dict[str, List[Program]],
        List[str],
        List[Tuple[str, str]],
    ]:
        """Aim every guided implementation round at the global incumbent.

        MAP-Elites still supplies diverse context and still rotates islands,
        but neither a newly purchased plan nor a productive reused plan should
        accidentally optimize a weak exploratory parent. The displaced sampled
        parent is retained as context so its structure remains evidence.
        """
        raw_parent, parent, _parent_info, context_dict, _context_ids, _context_info = sample
        incumbent = self.database.get_best_program() if self.database.programs else None
        if incumbent is None or (parent is not None and parent.id == incumbent.id):
            return sample

        anchored_context = {
            key: [program for program in programs if program.id != incumbent.id]
            for key, programs in context_dict.items()
        }
        if parent is not None and not any(
            parent.id == program.id
            for programs in anchored_context.values()
            for program in programs
        ):
            anchored_context.setdefault("exploration_parent", []).append(parent)
        anchored_context = {key: programs for key, programs in anchored_context.items() if programs}
        context_ids = [program.id for programs in anchored_context.values() for program in programs]
        context_info = [
            (key, program.id) for key, programs in anchored_context.items() for program in programs
        ]
        logger.info(
            "Anchored guided iteration to global incumbent %s (score %.6f) "
            "instead of sampled %s",
            incumbent.id[:8],
            get_score(incumbent.metrics),
            parent.id[:8] if parent is not None else "none",
        )
        return (
            incumbent,
            incumbent,
            ("", incumbent.id),
            anchored_context,
            context_ids,
            context_info,
        )

    async def _evaluate_candidate(
        self,
        *,
        candidate_index: int,
        candidate_count: int,
        candidate_response: str,
        child_solution: str,
        changes_summary: Optional[str],
        parent: Optional[Program],
        parent_info: Optional[Tuple[str, str]],
        context_ids: List[str],
        context_info: List[Tuple[str, str]],
        prompt: Dict[str, str],
        iteration: int,
        iteration_start: float,
        llm_generation_time: float,
        strategy: Optional[Strategy] = None,
        implementation_model_tier: str = "cheap",
    ) -> SerializableResult:
        """Evaluate one parsed candidate; this stage runs concurrently."""
        child_id = str(uuid.uuid4())
        mutation_error = (
            self._process_global_mutation_error(child_solution)
            if self._setting("reject_process_global_mutations")
            else None
        )
        if mutation_error:
            failed_child = Program(
                id=child_id,
                solution=child_solution,
                language=self.language,
                parent_id=parent.id if parent else None,
                metrics={"combined_score": 0.0, "error": mutation_error},
                iteration_found=iteration,
                metadata={
                    "changes": changes_summary,
                    "failed_evaluation": True,
                    "implementation_model_tier": implementation_model_tier,
                },
            )
            return SerializableResult(
                error=mutation_error,
                child_program_dict=failed_child.to_dict(),
                prompt=prompt,
                llm_response=candidate_response,
                iteration=iteration,
                llm_generation_time=llm_generation_time,
                eval_time=0.0,
            )
        eval_start = time.time()
        try:
            eval_result = await self.evaluator.evaluate_program(child_solution, child_id)
        except Exception as exc:
            return SerializableResult(
                error=f"Candidate evaluation raised: {exc}",
                prompt=prompt,
                llm_response=candidate_response,
                iteration=iteration,
            )
        eval_time = time.time() - eval_start
        metrics = dict(eval_result.metrics)
        artifacts = eval_result.artifacts or {}

        evaluation_error = self._evaluation_error(metrics, artifacts)
        if evaluation_error:
            failed_child = Program(
                id=child_id,
                solution=child_solution,
                language=self.language,
                parent_id=parent.id if parent else None,
                metrics=metrics,
                iteration_found=iteration,
                metadata={
                    "changes": changes_summary,
                    "failed_evaluation": True,
                    "implementation_model_tier": implementation_model_tier,
                },
                artifacts=artifacts,
            )
            return SerializableResult(
                error=evaluation_error,
                child_program_dict=failed_child.to_dict(),
                prompt=prompt,
                llm_response=candidate_response,
                iteration=iteration,
                llm_generation_time=llm_generation_time,
                eval_time=eval_time,
            )

        metadata: Dict[str, Any] = {
            "candidate_index": candidate_index,
            "candidate_count": candidate_count,
            "implementations_per_strategy": candidate_count,
            "two_tier_generation": strategy is not None,
            "implementation_model_tier": implementation_model_tier,
        }
        if strategy is not None:
            metadata["strategy_index"] = strategy.index
            metadata["strategy_title"] = strategy.title
        if parent is None:
            child = Program(
                id=child_id,
                solution=child_solution,
                language=self.language,
                parent_id=None,
                metrics=metrics,
                iteration_found=iteration,
                metadata={"changes": changes_summary, **metadata},
                artifacts=artifacts,
            )
        else:
            child = self._create_child_program(
                child_id=child_id,
                child_solution=child_solution,
                parent=parent,
                context_program_ids=context_ids,
                parent_info=parent_info or ("", parent.id),
                context_info=context_info,
                child_metrics=metrics,
                iteration=iteration,
                changes_summary=changes_summary,
                extra_metadata=metadata,
                artifacts=artifacts,
            )

        return SerializableResult(
            child_program_dict=child.to_dict(),
            parent_id=parent.id if parent else None,
            other_context_ids=context_ids,
            iteration_time=time.time() - iteration_start,
            llm_generation_time=llm_generation_time,
            eval_time=eval_time,
            prompt=prompt,
            llm_response=candidate_response,
            iteration=iteration,
        )

    @staticmethod
    def _process_global_mutation_error(solution: str) -> Optional[str]:
        """Reject generated JAX configuration changes before they poison peers."""
        try:
            tree = ast.parse(solution)
        except SyntaxError:
            return None

        jax_names = {"jax"}
        config_names: Set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "jax":
                        jax_names.add(alias.asname or "jax")
            elif isinstance(node, ast.ImportFrom) and node.module == "jax":
                for alias in node.names:
                    if alias.name == "config":
                        config_names.add(alias.asname or "config")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"update", "parse_flags_with_absl"}:
                continue
            owner = node.func.value
            direct_config = isinstance(owner, ast.Name) and owner.id in config_names
            jax_config = (
                isinstance(owner, ast.Attribute)
                and owner.attr == "config"
                and isinstance(owner.value, ast.Name)
                and owner.value.id in jax_names
            )
            if direct_config or jax_config:
                return (
                    "Generated program changes process-wide JAX configuration; configure "
                    "precision outside candidate code so later evaluations remain comparable"
                )
        return None

    @staticmethod
    def _evaluation_error(metrics: Dict[str, Any], artifacts: Dict[str, Any]) -> Optional[str]:
        """Return an evaluator failure message using the default controller rules."""
        if "combined_score" in metrics:
            try:
                finite = math.isfinite(float(metrics["combined_score"]))
            except (ValueError, TypeError):
                finite = False
            if not finite:
                return "Evaluator returned a non-finite or non-numeric combined_score"
        failed = (
            metrics.get("validity") in (0, -1)
            or (metrics.get("timeout") is True and metrics.get("validity") is None)
            or (
                metrics.get("combined_score") == 0
                and (metrics.get("error") is not None or "error" in artifacts)
            )
        )
        if not failed:
            return None
        return (
            (metrics.get("error") if isinstance(metrics.get("error"), str) else None)
            or artifacts.get("error")
            or artifacts.get("stderr")
            or metrics.get("error_message")
            or "Evaluation failed"
        )

    def _recover_full_rewrite(self, response: str, parent_solution: str) -> Optional[str]:
        """Salvage a whole program from a response whose diff would not apply.

        More than half of all wasted candidates are diffs whose SEARCH block
        never matched the parent -- usually because the model emitted the
        finished file in a fence instead of edits. The call is already paid
        for, so re-reading its own response costs nothing, while asking again
        would cost a whole extra call at a strictly later point on the cost
        axis.

        Guarded against accepting a fragment as a program: the salvage must be
        substantial relative to the parent, and must actually differ from it.

        Measured at B=$0.02 over 3 replicates this did not pay (0.790 with vs
        0.837 without, within noise, one run derailed after a salvaged program
        became a parent), so it is off unless ``recover_unusable_diffs`` is
        set. A failed diff usually means a confused response, not a whole
        program offered in place of edits.
        """
        if not self.recover_unusable_diffs:
            return None
        candidate = parse_full_rewrite(response, self.language)
        if not candidate:
            return None
        candidate = candidate.strip()
        if not candidate or candidate == parent_solution.strip():
            return None
        if len(candidate) < _RECOVERY_MIN_RATIO * len(parent_solution):
            return None
        return candidate
