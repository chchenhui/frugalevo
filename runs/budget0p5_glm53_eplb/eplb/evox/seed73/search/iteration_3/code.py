# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Stagnation-escape search database.

    When the population is stuck (no meaningful improvement), sampling leans
    heavily on DIVERGE_LABEL with a fresh (under-used, non-best) parent and
    empty context — the mechanism that historically produced breakthroughs.
    Otherwise it samples unlabeled with diverse multi-tier context.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = 0.0
        self.stagnation = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program, tracking stagnation and parent usage."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        score = self._score(program)
        if score > 0:
            # Meaningful improvement: >1% relative or >0.01 absolute.
            if score > self.best_score + 0.01 or score > self.best_score * 1.01:
                self.stagnation = 0
            elif score > self.best_score:
                pass  # tiny gain: still counts as stagnation
            self.best_score = max(self.best_score, score)

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def _pick_diverge_parent(self, candidates: List[EvolvedProgram]) -> EvolvedProgram:
        """Pick a non-best, under-used parent (weighted by inverse usage)."""
        best_id = None
        best_score = -1.0
        for p in candidates:
            s = self._score(p)
            if s > best_score:
                best_score, best_id = s, p.id

        pool = [p for p in candidates if p.id != best_id]
        if not pool:
            pool = candidates

        # Weight by inverse usage so no parent is over-exploited.
        weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p in pool]
        return self.random_state.choices(pool, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        """Select parent + context based on stagnation state."""
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4

        # Stagnant population: mostly DIVERGE with fresh parent, empty context.
        if self.stagnation >= 3 and self.random_state.random() < 0.65:
            parent = self._pick_diverge_parent(candidates)
            return {self.DIVERGE_LABEL: parent}, {self.DIVERGE_LABEL: []}

        # Otherwise: unlabeled sample with diverse multi-tier context.
        scored = sorted(candidates, key=self._score, reverse=True)
        parent = None
        if scored and self.random_state.random() < 0.5:
            # Top-tier parent (but rotate among top few to avoid overuse).
            top = scored[: min(3, len(scored))]
            parent = self.random_state.choice(top)
        else:
            # Mid-tier parent for exploration.
            mid = scored[len(scored) // 4: 3 * len(scored) // 4] or scored
            parent = self.random_state.choice(mid)

        context: List[EvolvedProgram] = []
        seen = {parent.id}
        # Best program for direction.
        if scored and scored[0].id not in seen:
            context.append(scored[0])
            seen.add(scored[0].id)
        # Worst program as a contrasting example (what to avoid).
        if scored and scored[-1].id not in seen:
            context.append(scored[-1])
            seen.add(scored[-1].id)
        # Random others for diversity.
        others = [p for p in candidates if p.id not in seen]
        self.random_state.shuffle(others)
        context.extend(others[: n_ctx - len(context)])

        return {"": parent}, {"": context[:n_ctx]}


# EVOLVE-BLOCK-END