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
    """Adaptive elite search with controlled diversification."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # Rebuilt through add(), including when a database is restored.
        self.best_score_seen: Optional[float] = None
        self.last_meaningful_improvement_iteration = 0
        self.observed_iteration = 0
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.parent_gain_count: Dict[str, int] = {}
        self.last_label_iteration = -100

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        current_iteration = iteration
        if current_iteration is None:
            current_iteration = program.iteration_found
        if isinstance(current_iteration, int):
            self.observed_iteration = max(self.observed_iteration, current_iteration)
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if (
                child_score is not None
                and parent_score is not None
                and child_score > parent_score
            ):
                self.parent_gain_count[program.parent_id] = (
                    self.parent_gain_count.get(program.parent_id, 0) + 1
                )

        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

        parent_info = program.parent_info
        if (
            isinstance(parent_info, tuple)
            and len(parent_info) >= 1
            and parent_info[0] in (self.DIVERGE_LABEL, self.REFINE_LABEL)
        ):
            self.last_label_iteration = self.observed_iteration

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.last_meaningful_improvement_iteration = self.observed_iteration
            elif score > self.best_score_seen:
                improvement = score - self.best_score_seen
                threshold = max(0.01, 0.01 * abs(self.best_score_seen))
                self.best_score_seen = score
                if improvement > threshold:
                    self.last_meaningful_improvement_iteration = self.observed_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def _weighted_choice(
        self, programs: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        total = sum(weights)
        if total <= 0:
            return self.random_state.choice(programs)
        point = self.random_state.random() * total
        running = 0.0
        for program, weight in zip(programs, weights):
            running += weight
            if running >= point:
                return program
        return programs[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_count = 4 if num_context_programs is None else max(0, num_context_programs)
        scored = [(self._score(p), p) for p in candidates]
        valid = [p for score, p in scored if score is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            contexts = [
                p for p in candidates if p.id != parent.id
            ][:context_count]
            return {"": parent}, {"": contexts}

        ranked = sorted(valid, key=lambda p: self._score(p) or float("-inf"), reverse=True)

        # Search among a compact elite set, but penalize parents repeatedly used
        # without producing improvements.
        elite_size = min(len(ranked), max(4, int(math.ceil(len(ranked) * 0.4))))
        elite = ranked[:elite_size]
        weights: List[float] = []
        for rank, program in enumerate(elite):
            uses = self.parent_use_count.get(program.id, 0)
            gains = self.parent_gain_count.get(program.id, 0)
            weights.append((1.0 / (rank + 1)) * (1.0 + 0.35 * gains) / (1.0 + 0.4 * uses))
        parent = self._weighted_choice(elite, weights)

        stagnation = self.observed_iteration - self.last_meaningful_improvement_iteration
        label = ""
        if stagnation >= 10 and (
            self.observed_iteration - self.last_label_iteration >= 4
        ):
            # A tightly clustered elite indicates local tuning has saturated.
            # Prefer divergence, but occasionally refine a parent that has
            # historically produced gains.
            if self.random_state.random() < 0.65:
                label = self.DIVERGE_LABEL
            elif self.parent_gain_count.get(parent.id, 0) > 0:
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        available = [p for p in ranked if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # One strong alternative anchors the candidate, while the remaining
        # examples are selected with low context reuse for varied perspectives.
        if available and context_count:
            contexts.append(available[0])

        while available and len(contexts) < context_count:
            pool = [p for p in available if p.id not in {x.id for x in contexts}]
            if not pool:
                break
            pool_weights = [
                1.0 / (1.0 + self.context_use_count.get(p.id, 0))
                * (1.0 + min(i, 6) * 0.08)
                for i, p in enumerate(pool)
            ]
            contexts.append(self._weighted_choice(pool, pool_weights))

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END