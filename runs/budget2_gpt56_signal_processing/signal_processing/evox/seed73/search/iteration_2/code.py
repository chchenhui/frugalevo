# EVOLVE-BLOCK-START
import logging
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search balancing strong frontier programs and productive lineages."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_observed_score = -float("inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.label_targets: Dict[str, int] = {}

    def _score(self, program: Optional[EvolvedProgram]) -> Optional[float]:
        if program is None:
            return None
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        score = self._score(program)

        # Learn lineage usefulness from actual completed evaluations.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent_score = self._score(self.get(program.parent_id))
            if score is not None and parent_score is not None:
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + score - parent_score
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1
            target_id = program.parent_info[1] or program.parent_id
            if target_id:
                key = label + ":" + target_id
                self.label_targets[key] = self.label_targets.get(key, 0) + 1

        if score is not None:
            threshold = max(0.01, 0.01 * abs(self.best_observed_score)) if math.isfinite(
                self.best_observed_score
            ) else 0.0
            if score > self.best_observed_score + threshold:
                self.best_observed_score = score
                self.stagnation_steps = 0
            else:
                self.best_observed_score = max(self.best_observed_score, score)
                self.stagnation_steps += 1

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        programs = list(self.programs.values())
        valid = [(p, self._score(p)) for p in programs]
        valid = [(p, s) for p, s in valid if s is not None]

        if not valid:
            if not programs:
                raise ValueError("No candidates available for sampling")
            return {"": self.random_state.choice(programs)}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]
        worst = valid[-1][1]
        span = max(best - worst, 1e-9)
        median = valid[len(valid) // 2][1]

        # Keep most selections near the frontier, but retain historically
        # productive parents even when their own score is not currently elite.
        productive_ids = sorted(
            self.parent_gain_sum,
            key=lambda pid: self.parent_gain_sum[pid] / max(1, self.parent_uses.get(pid, 1)),
            reverse=True,
        )[:4]

        pool: List[Tuple[EvolvedProgram, float]] = [
            (p, s) for p, s in valid if s >= median
        ]
        pool_ids = {p.id for p, _ in pool}
        for p, s in valid:
            if p.id in productive_ids and p.id not in pool_ids:
                pool.append((p, s))
                pool_ids.add(p.id)

        weights: List[float] = []
        for program, score in pool:
            quality = (score - worst) / span
            uses = self.parent_uses.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / max(1, uses)
            # Successful lineages matter, especially in this population where
            # a modest-scoring ancestor has produced frontier solutions.
            weight = 0.35 + 2.2 * quality + max(-0.2, min(1.0, mean_gain * 18.0))
            weight += 0.8 / (1 + uses)
            weights.append(max(0.05, weight))

        parent = self.random_state.choices(
            [p for p, _ in pool], weights=weights, k=1
        )[0]
        label = ""

        # Explicit instructions are reserved for genuine plateaus and are not
        # repeatedly aimed at the same candidate.
        if self.stagnation_steps >= 8 and self.stagnation_steps % 5 == 0:
            alternatives = [
                (p, s) for p, s in valid
                if s >= median and p.id != valid[0][0].id
                and self.label_targets.get(self.DIVERGE_LABEL + ":" + p.id, 0) == 0
            ]
            if alternatives:
                parent = self.random_state.choice(alternatives)[0]
                label = self.DIVERGE_LABEL
        elif self.stagnation_steps >= 6 and self.stagnation_steps % 5 == 3:
            top = valid[:min(4, len(valid))]
            refinable = [
                (p, s) for p, s in top
                if self.label_targets.get(self.REFINE_LABEL + ":" + p.id, 0) == 0
            ]
            if refinable:
                parent = self.random_state.choice(refinable)[0]
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"": []}

        context_limit = max(0, num_context_programs or 0)
        remaining = [(p, s) for p, s in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Use complementary score regions: another frontier reference, an
        # upper-mid alternative, and occasionally a productive non-elite idea.
        targets = [0.95, 0.72, 0.50, 0.30]
        for target in targets:
            if len(contexts) >= context_limit or not remaining:
                break
            choices = []
            choice_weights = []
            for program, score in remaining:
                normalized = (score - worst) / span
                distance = abs(normalized - target)
                novelty = 1.0 / (1 + self.context_uses.get(program.id, 0))
                choices.append(program)
                choice_weights.append(max(0.05, 1.2 - distance + 0.6 * novelty))
            chosen = self.random_state.choices(choices, weights=choice_weights, k=1)[0]
            contexts.append(chosen)
            remaining = [(p, s) for p, s in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END