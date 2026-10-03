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
    """Balanced elite exploration with lineage-aware parent selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program: Optional[EvolvedProgram] = None
        self.best_score_seen: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.last_iteration: int = 0
        self.last_meaningful_improvement_iteration: int = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.parent_gain_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        if program.iteration_found == 0 or iteration == 0:
            self.initial_program = program

        current_iteration = (
            iteration if isinstance(iteration, int) else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_usage[program.parent_id] = (
                self.parent_usage.get(program.parent_id, 0) + 1
            )

            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + gain
                )
                self.parent_gain_count[program.parent_id] = (
                    self.parent_gain_count.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.best_program_id = program.id
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                previous_best = self.best_score_seen
                delta = score - previous_best
                relative = delta / max(abs(previous_best), 1e-12)
                self.best_score_seen = score
                self.best_program_id = program.id
                if delta > 0.01 or relative > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

        if getattr(self.config, "db_path", None):
            self._save_program(program)
        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        all_programs = list(self.programs.values())
        if not all_programs:
            raise ValueError("No candidates available for sampling")

        ranked = [(self._score(p), p) for p in all_programs]
        ranked = [(s, p) for s, p in ranked if s is not None]
        ranked.sort(key=lambda x: x[0], reverse=True)

        if not ranked:
            return {"": self.random_state.choice(all_programs)}, {"": []}

        # During a long plateau, broaden beyond only the near-identical elite.
        stagnation = self.last_iteration - self.last_meaningful_improvement_iteration
        pool_fraction = 0.65 if stagnation >= 10 else 0.45
        pool_size = max(4, int(math.ceil(len(ranked) * pool_fraction)))
        parent_pool = ranked[:pool_size]

        low_score = parent_pool[-1][0]
        high_score = parent_pool[0][0]
        average_gains: List[float] = []
        for _, program in parent_pool:
            count = self.parent_gain_count.get(program.id, 0)
            if count:
                average_gains.append(self.parent_gain_sum[program.id] / count)
        gain_scale = max(max([abs(x) for x in average_gains] or [0.0]), 1e-5)

        weights: List[float] = []
        for score, program in parent_pool:
            quality = 1.0 + (score - low_score) / max(high_score - low_score, 1e-9)
            uses = self.parent_usage.get(program.id, 0)
            reuse_bonus = 1.0 / math.sqrt(1.0 + uses)

            count = self.parent_gain_count.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / max(count, 1)
            lineage_bonus = max(0.6, min(1.5, 1.0 + mean_gain / gain_scale))

            weights.append(quality * reuse_bonus * lineage_bonus)

        parent = self.random_state.choices(
            [p for _, p in parent_pool], weights=weights, k=1
        )[0]

        context_count = max(0, num_context_programs or 0)
        if context_count == 0:
            return {"": parent}, {"": []}

        available = [p for _, p in ranked if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Supply complementary strong examples instead of repeatedly showing
        # the same best solution plus very weak early failures.
        bands = [
            available[:max(1, len(available) // 4)],
            available[max(1, len(available) // 4):max(2, len(available) // 2)],
            available[max(2, len(available) // 2):max(3, 3 * len(available) // 4)],
        ]

        for band in bands:
            if len(contexts) >= context_count or not band:
                continue
            choices = [p for p in band if p.id not in {c.id for c in contexts}]
            if choices:
                weights = [
                    1.0 / (1.0 + self.context_usage.get(p.id, 0))
                    for p in choices
                ]
                contexts.append(self.random_state.choices(choices, weights=weights, k=1)[0])

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        while remaining and len(contexts) < context_count:
            weights = [
                1.0 / (1.0 + self.context_usage.get(p.id, 0))
                for p in remaining
            ]
            chosen = self.random_state.choices(remaining, weights=weights, k=1)[0]
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END