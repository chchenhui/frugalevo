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
    """Adaptive parent/context selection.

    Since the population is saturated at the top score, the strategy is:
    - Rotate among the best-scoring programs as parents (avoid overuse).
    - Provide the best programs as context so the LLM can combine strengths.
    - Occasionally apply REFINE_LABEL on the current best to push past the
      plateau, and rarely DIVERGE_LABEL for fresh directions.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.sample_calls = 0
        self.best_score: float = -1.0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.parent_use_count.setdefault(program.id, 0)

        s = self._score(program)
        if s > self.best_score:
            self.best_score = s

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        # Rank by score; ties broken by recency and low usage for rotation.
        scored = sorted(candidates, key=lambda p: -self._score(p))
        top = scored[: max(5, len(scored) // 3)]

        # Weighted pick among top scorers, penalizing overused parents.
        weights = []
        for p in top:
            w = 1.0 / (1.0 + 3 * self.parent_use_count.get(p.id, 0))
            weights.append(w + 0.05)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Context: mix of other top programs + a couple of diverse/lower-score
        # programs for complementary perspectives.
        others = [p for p in scored if p.id != parent.id]
        ctx = [p for p in others[: n_ctx]]
        if len(ctx) < n_ctx and len(others) > len(ctx):
            rest = others[len(ctx):]
            self.random_state.shuffle(rest)
            ctx.extend(rest[: n_ctx - len(ctx)])

        # Labels: mostly empty; occasionally REFINE the best, rarely DIVERGE.
        label = ""
        r = self.random_state.random()
        if r < 0.25 and self._score(parent) >= self.best_score - 1e-9:
            label = self.REFINE_LABEL
            ctx = []  # focused refinement on the parent
        elif r < 0.35 and self.sample_calls % 7 == 0:
            label = self.DIVERGE_LABEL
            ctx = []

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END