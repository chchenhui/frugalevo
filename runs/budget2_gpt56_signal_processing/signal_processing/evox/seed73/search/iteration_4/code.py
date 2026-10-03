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
    """Score-guided search that exploits strong solutions while avoiding reuse."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_observed_score = -float("inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.recent_labels: List[str] = []
        self.recent_gains: List[float] = []

    def _score(self, program: Optional[EvolvedProgram]) -> Optional[float]:
        if program is None:
            return None
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def _meaningful(self, score: float) -> bool:
        if not math.isfinite(self.best_observed_score):
            return True
        return score > self.best_observed_score + max(
            0.01, 0.01 * abs(self.best_observed_score)
        )

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        parent = self.get(program.parent_id) if program.parent_id else None
        parent_score = self._score(parent)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + gain
                )
                self.recent_gains.append(gain)
                self.recent_gains = self.recent_gains[-8:]

        context_ids = program.other_context_ids or []
        for context_id in context_ids:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        label = ""
        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
        self.recent_labels.append(label)
        self.recent_labels = self.recent_labels[-6:]

        if score is not None:
            if self._meaningful(score):
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
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        valid = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in valid if s is not None]
        if not valid:
            return {"": self.random_state.choice(candidates)}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]
        floor = valid[-1][1]
        span = max(best - floor, 1e-9)

        # Use a compact elite pool: recent failures suggest concentrating on the
        # best known signal-processing ideas, but not repeatedly mutating one.
        pool_size = min(len(valid), max(6, min(12, len(valid) // 3 + 1)))
        pool = valid[:pool_size]
        programs: List[EvolvedProgram] = []
        weights: List[float] = []

        for rank, (program, score) in enumerate(pool):
            quality = (score - floor) / span
            uses = self.parent_uses.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / max(1, uses)
            gain_bonus = max(-0.35, min(0.6, mean_gain * 25.0))
            weight = (0.6 + 2.2 * quality + gain_bonus) / (1.0 + 0.35 * uses)
            weight *= 1.0 / (1.0 + 0.08 * rank)
            programs.append(program)
            weights.append(max(0.05, weight))

        parent = self.random_state.choices(programs, weights=weights, k=1)[0]
        label = ""

        # Labels are sparse interventions after a real plateau. A deep plateau
        # first asks for a different high-quality approach; later it asks the
        # best candidate for focused tuning.
        no_recent_label = not any(self.recent_labels[-3:])
        if self.stagnation_steps >= 14 and no_recent_label and len(pool) > 2:
            parent = self.random_state.choice([p for p, _ in pool[1:]])
            label = self.DIVERGE_LABEL
        elif self.stagnation_steps >= 8 and no_recent_label:
            parent = valid[0][0]
            label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"": []}

        context_count = max(0, num_context_programs or 0)
        context_pool = [p for p, _ in valid[:max(12, len(valid) // 2)] if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        while context_pool and len(contexts) < context_count:
            choices: List[EvolvedProgram] = []
            weights = []
            for program in context_pool:
                score = self._score(program)
                quality = ((score or floor) - floor) / span
                novelty = 1.0 / (1.0 + self.context_uses.get(program.id, 0))
                separation = 1.0
                if contexts:
                    separation += min(
                        abs((score or floor) - (self._score(old) or floor)) / span
                        for old in contexts
                    )
                choices.append(program)
                weights.append(max(0.05, (0.4 + quality + novelty) * separation))

            chosen = self.random_state.choices(choices, weights=weights, k=1)[0]
            contexts.append(chosen)
            context_pool = [p for p in context_pool if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END