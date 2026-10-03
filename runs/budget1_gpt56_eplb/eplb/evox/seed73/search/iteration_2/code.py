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
    """Adaptive scalar-search database with score and lineage-aware sampling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen_score = float("-inf")
        self.last_meaningful_improvement_iteration = -1
        self.parent_use_count: Dict[str, int] = {}
        self.parent_reward_sum: Dict[str, float] = {}
        self.parent_reward_count: Dict[str, int] = {}
        self.context_pair_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", -1), iteration)

        score = self._score(program)
        effective_iteration = (
            iteration if iteration is not None else program.iteration_found
        )

        # Rebuild useful lineage statistics through add(), including on resume.
        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                delta = score - parent_score
                self.parent_reward_sum[program.parent_id] = (
                    self.parent_reward_sum.get(program.parent_id, 0.0) + delta
                )
                self.parent_reward_count[program.parent_id] = (
                    self.parent_reward_count.get(program.parent_id, 0) + 1
                )

            for context_id in program.other_context_ids:
                key = program.parent_id + "|" + context_id
                self.context_pair_count[key] = self.context_pair_count.get(key, 0) + 1

        if score is not None:
            threshold = max(0.01, abs(self.best_seen_score) * 0.01) if (
                self.best_seen_score != float("-inf")
            ) else 0.0
            if score > self.best_seen_score + threshold:
                self.last_meaningful_improvement_iteration = effective_iteration
            self.best_seen_score = max(self.best_seen_score, score)

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

        scored = [(program, self._score(program)) for program in candidates]
        numeric = [(program, score) for program, score in scored if score is not None]

        # If evaluation is incomplete, retain safe random behavior.
        if not numeric:
            parent = self.random_state.choice(candidates)
            others = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(others)
            return {"": parent}, {"": others[: (num_context_programs or 0)]}

        numeric.sort(key=lambda item: item[1], reverse=True)
        scores = [score for _, score in numeric]
        low, high = min(scores), max(scores)
        spread = max(high - low, 1e-9)

        # Favor strong programs, but reward parents whose children improved and
        # penalize repeatedly mutating the same lineage.
        parent_weights: List[float] = []
        for program, score in numeric:
            quality = (score - low) / spread
            uses = self.parent_use_count.get(program.id, 0)
            reward_count = self.parent_reward_count.get(program.id, 0)
            reward = (
                self.parent_reward_sum.get(program.id, 0.0) / reward_count
                if reward_count
                else 0.0
            )
            reward_bonus = max(-0.25, min(0.35, reward * 80.0))
            novelty = 1.0 / math.sqrt(1.0 + uses)
            parent_weights.append(
                max(0.05, (0.20 + 0.80 * quality + reward_bonus) * novelty)
            )

        parent = self.random_state.choices(
            [program for program, _ in numeric], weights=parent_weights, k=1
        )[0]

        context_count = max(0, num_context_programs or 0)
        pool = [(p, s) for p, s in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Context is deliberately drawn from several good alternatives rather
        # than repeatedly reusing weak examples or the same parent/context pair.
        while pool and len(contexts) < context_count:
            weights = []
            for program, score in pool:
                quality = (score - low) / spread
                pair_key = parent.id + "|" + program.id
                pair_reuse = self.context_pair_count.get(pair_key, 0)
                weights.append((0.25 + 0.75 * quality) / (1.0 + pair_reuse))

            chosen = self.random_state.choices(
                [program for program, _ in pool], weights=weights, k=1
            )[0]
            contexts.append(chosen)
            pool = [(p, s) for p, s in pool if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END