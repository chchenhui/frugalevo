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
    """Diversity-aware search database.

    Key ideas:
    1. Anti-reuse: track how often each program has been used as parent or
       context; prefer under-used programs to break the convergence loop.
    2. Stagnation-driven labels: when the best score hasn't meaningfully
       improved, alternate between DIVERGE (new direction from a mid-tier
       parent) and REFINE (on the current best, but rarely).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.best_id: Optional[str] = None
        self.stagnation: int = 0
        self.sample_calls: int = 0

    @staticmethod
    def _score(p: Program) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        s = self._score(program)
        # Only count meaningful improvements (>1% relative or >0.01 absolute)
        meaningful = s > self.best_score + 0.01 or s > self.best_score * 1.01
        if s > self.best_score:
            if meaningful:
                self.stagnation = 0
            self.best_score = s
            self.best_id = program.id
        else:
            self.stagnation += 1

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

        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(1, len(scored) // 4)]
        mid = scored[len(scored) // 4: len(scored) // 2] or scored

        label = ""
        parent = None

        # Stagnation-driven label use (rarely, and rotate targets)
        if self.stagnation >= 3 and self.sample_calls % 3 == 0:
            label = self.DIVERGE_LABEL
            # Diverge from a mid-tier, under-used parent (fresh direction)
            pool = [p for p in mid if p.id != self.best_id] or mid
            parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
        elif self.stagnation >= 3 and self.sample_calls % 5 == 0 and self.best_id:
            label = self.REFINE_LABEL
            parent = self.programs[self.best_id]
        else:
            # Default: exploit top tier but rotate via anti-reusage weighting
            pool = top
            weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p in pool]
            parent = self.random_state.choices(pool, weights=weights, k=1)[0]

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: mix of top-tier and lower-tier programs, anti-reuse weighted
        others = [p for p in candidates if p.id != parent.id]
        others.sort(key=lambda p: self.context_usage.get(p.id, 0))
        top_ctx = [p for p in others if self._score(p) >= self.best_score - 0.005]
        low_ctx = [p for p in others if p not in top_ctx]
        ctx = []
        for p in (top_ctx + low_ctx):
            if p not in ctx:
                ctx.append(p)
            if len(ctx) >= n_ctx:
                break
        # Fallback fill
        for p in others:
            if len(ctx) >= n_ctx:
                break
            if p not in ctx:
                ctx.append(p)

        for p in ctx:
            self.context_usage[p.id] = self.context_usage.get(p.id, 0) + 1

        if label:
            return {label: parent}, {"": []}

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END