# EVOLVE-BLOCK-START
import logging
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
    """Adaptive elite search with low-reuse contextual examples."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use: Dict[str, int] = {}
        self.context_use: Dict[str, int] = {}
        self.seen_ids = set()
        self.best_score: Optional[float] = None
        self.last_meaningful_gain_iteration = 0
        self.latest_iteration = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store programs and rebuild useful progress/reuse statistics."""
        self.programs[program.id] = program

        current_iteration = iteration
        if not isinstance(current_iteration, int):
            current_iteration = program.iteration_found
        if not isinstance(current_iteration, int):
            current_iteration = self.latest_iteration
        self.latest_iteration = max(self.latest_iteration, current_iteration)

        if program.id not in self.seen_ids:
            self.seen_ids.add(program.id)

            if program.parent_id:
                self.parent_use[program.parent_id] = (
                    self.parent_use.get(program.parent_id, 0) + 1
                )
            for context_id in program.other_context_ids or []:
                self.context_use[context_id] = self.context_use.get(context_id, 0) + 1

            score = self._score(program)
            if score is not None:
                if self.best_score is None:
                    self.best_score = score
                    self.last_meaningful_gain_iteration = current_iteration
                elif score > self.best_score:
                    improvement = score - self.best_score
                    meaningful = improvement > 0.01 or (
                        self.best_score != 0
                        and improvement > abs(self.best_score) * 0.01
                    )
                    self.best_score = score
                    if meaningful:
                        self.last_meaningful_gain_iteration = current_iteration

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

        count = max(0, num_context_programs or 0)
        scored = [(self._score(p), p) for p in candidates]
        valid = [(s, p) for s, p in scored if s is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            pool = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(pool)
            return {"": parent}, {"": pool[:count]}

        valid.sort(key=lambda item: item[0], reverse=True)
        elite_size = min(len(valid), max(3, min(8, len(valid) // 2 + 1)))
        elite = [p for _, p in valid[:elite_size]]

        # Elite exploitation, tempered by past parent reuse. Rank weights keep
        # the current best important without repeatedly selecting it exclusively.
        weights = []
        for rank, parent in enumerate(elite):
            reuse = self.parent_use.get(parent.id, 0)
            weights.append((elite_size - rank + 1) / (1.0 + 0.45 * reuse))
        parent = self.random_state.choices(elite, weights=weights, k=1)[0]

        stagnant = self.latest_iteration - self.last_meaningful_gain_iteration
        if stagnant >= 6 and self.latest_iteration % 5 == 0:
            # A plateau near the optimum usually benefits from focused polishing.
            return {self.REFINE_LABEL: parent}, {}
        if stagnant >= 10 and self.latest_iteration % 7 == 0 and len(elite) > 1:
            # Occasionally ask for a distinct direction from a strong alternative.
            alternatives = [p for p in elite if p.id != parent.id]
            return {self.DIVERGE_LABEL: self.random_state.choice(alternatives)}, {}

        pool = [p for _, p in valid if p.id != parent.id]
        chosen: List[EvolvedProgram] = []
        signatures = set()

        # Context favors strong, underused programs and avoids showing the LLM
        # many near-identical solutions.
        while pool and len(chosen) < count:
            best_index = 0
            best_value = None
            for index, candidate in enumerate(pool):
                score = self._score(candidate) or 0.0
                signature = candidate.solution[:180]
                novelty = 0.35 if signature not in signatures else 0.0
                reuse_penalty = 0.08 * self.context_use.get(candidate.id, 0)
                value = score + novelty - reuse_penalty + self.random_state.random() * 0.03
                if best_value is None or value > best_value:
                    best_value = value
                    best_index = index
            selected = pool.pop(best_index)
            chosen.append(selected)
            signatures.add(selected.solution[:180])

        return {"": parent}, {"": chosen}


# EVOLVE-BLOCK-END