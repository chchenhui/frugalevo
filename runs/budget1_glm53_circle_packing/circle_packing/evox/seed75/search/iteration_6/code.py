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
    """Search strategy focused on escaping stagnation.

    Principle: when progress stalls, refining the current best is saturated
    (many duplicate 0.9984 children). Breakthroughs historically came from
    DIVERGE on fresh, non-best parents. So we rotate parents by usage count,
    and on stagnation we diverge from the least-used mid/high-tier parent.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        score = self._score(program)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation = 0
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

        n_ctx = num_context_programs or 4
        scored = sorted(candidates, key=self._score, reverse=True)

        # Stagnation: diverge from a fresh, non-best parent (top 40% but not top 8).
        if self.stagnation >= 3 and len(scored) > 10:
            pool = scored[8:max(12, len(scored) // 2)]
            pool = [p for p in pool if self.parent_usage.get(p.id, 0) < 2]
            if pool:
                parent = min(pool, key=lambda p: self.parent_usage.get(p.id, 0))
                self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
                return {self.DIVERGE_LABEL: parent}, {self.DIVERGE_LABEL: []}

        # Default: rotate among top-tier parents by least usage.
        top = scored[:max(4, len(scored) // 4)]
        top = [p for p in top if self._score(p) > 0.5] or top
        parent = min(top, key=lambda p: (self.parent_usage.get(p.id, 0), self.random_state.random()))
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: one other strong program + random diverse programs (incl. low scorers).
        ctx = []
        strong = [p for p in top if p.id != parent.id and self._score(p) != self._score(parent)]
        if strong:
            ctx.append(self.random_state.choice(strong))
        rest = [p for p in candidates if p.id != parent.id and p.id not in {c.id for c in ctx}]
        self.random_state.shuffle(rest)
        ctx.extend(rest[: n_ctx - len(ctx)])

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END