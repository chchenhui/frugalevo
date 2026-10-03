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
    """Searches for productive parent lineages while retaining diverse context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.best_child_score: Dict[str, float] = {}
        self.best_child_id: Dict[str, str] = {}
        self.best_score_seen: Optional[float] = None
        self.last_meaningful_improvement_iteration = 0
        self.last_iteration = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = (
            iteration if iteration is not None else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_usage[program.parent_id] = (
                self.parent_usage.get(program.parent_id, 0) + 1
            )
            child_score = self._score(program)
            if child_score is not None:
                previous = self.best_child_score.get(program.parent_id)
                if previous is None or child_score > previous:
                    self.best_child_score[program.parent_id] = child_score
                    self.best_child_id[program.parent_id] = program.id

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = (
                self.context_usage.get(context_id, 0) + 1
            )

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                delta = score - self.best_score_seen
                relative = delta / max(abs(self.best_score_seen), 1e-12)
                self.best_score_seen = score
                if delta > 0.01 or relative > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        numeric = [
            (score, program)
            for program in self.programs.values()
            for score in [self._score(program)]
            if score is not None
        ]

        if not numeric:
            candidates = list(self.programs.values())
            if not candidates:
                raise ValueError("No candidates available for sampling")
            return {"": self.random_state.choice(candidates)}, {"": []}

        numeric.sort(key=lambda item: item[0], reverse=True)
        scores = [score for score, _ in numeric]
        low, high = scores[-1], scores[0]
        spread = max(high - low, 1e-6)

        # Credit parents that previously produced a substantially better child.
        # This is especially useful here because a mid/low-score parent produced
        # one of the best observed candidates.
        parent_weights: List[float] = []
        programs: List[EvolvedProgram] = []
        for score, program in numeric:
            own_quality = (score - low) / spread
            child_score = self.best_child_score.get(program.id, score)
            innovation = max(0.0, child_score - score) / spread
            reuse = self.parent_usage.get(program.id, 0)

            weight = (0.25 + own_quality + 2.5 * innovation) / math.sqrt(1 + reuse)
            programs.append(program)
            parent_weights.append(max(weight, 0.01))

        # Preserve some discovery pressure for unexplored lower/middle approaches.
        if self.random_state.random() < 0.18 and len(numeric) >= 6:
            middle = [p for _, p in numeric[len(numeric) // 3 :]]
            unused = [p for p in middle if self.parent_usage.get(p.id, 0) == 0]
            parent = self.random_state.choice(unused or middle)
        else:
            parent = self.random_state.choices(programs, weights=parent_weights, k=1)[0]

        count = max(0, num_context_programs or 0)
        if count == 0:
            return {"": parent}, {"": []}

        contexts: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def add_context(candidate: Optional[EvolvedProgram]) -> None:
            if (
                candidate is not None
                and candidate.id not in used_ids
                and len(contexts) < count
            ):
                contexts.append(candidate)
                used_ids.add(candidate.id)

        # Pair a productive ancestor with its strongest known descendant, then
        # provide elite alternatives and one contrasting non-elite example.
        child_id = self.best_child_id.get(parent.id)
        add_context(self.get(child_id) if child_id else None)
        add_context(numeric[0][1])

        if len(numeric) > 4:
            add_context(numeric[len(numeric) // 2][1])

        remaining = [p for _, p in numeric if p.id not in used_ids]
        while len(contexts) < count and remaining:
            weights = [
                1.0 / math.sqrt(1 + self.context_usage.get(p.id, 0))
                for p in remaining
            ]
            chosen = self.random_state.choices(remaining, weights=weights, k=1)[0]
            add_context(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END