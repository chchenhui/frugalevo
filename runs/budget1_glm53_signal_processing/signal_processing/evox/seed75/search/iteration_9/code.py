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
    """Adaptive parent/context selection for a converged population.

    Population is stuck near the best score. Strategy:
    - Rotate parents among top-tier programs with a usage penalty (avoid overuse).
    - Provide diverse context: best, a mid-tier program, a low scorer, and a random one.
    - Occasionally try DIVERGE on the best program (never tried before) when deeply stuck.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.diverge_attempts: int = 0

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.usage_count[program.id] = self.usage_count.get(program.id, 0)

        s = self._score(program)
        if s > self.best_score + 0.0001:
            if s > self.best_score * 1.01 or s > self.best_score + 0.01:
                self.stagnation = 0
            self.best_score = max(self.best_score, s)
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        progs = list(self.programs.values())
        scored = sorted(progs, key=self._score, reverse=True)
        n_ctx = num_context_programs or 4

        # Every ~6th call while deeply stuck: try DIVERGE on the best program
        # (best program was never used as a labeled target before).
        if self.stagnation > 5 and self.diverge_attempts < 2 and len(scored) % 6 == 0:
            self.diverge_attempts += 1
            parent = scored[0]
            self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}

        # Otherwise: rotate among top-tier parents with usage penalty.
        top = scored[: max(4, len(scored) // 4)]
        top.sort(key=lambda p: (self.usage_count.get(p.id, 0), -self._score(p)))
        parent = top[0] if self.random_state.random() < 0.7 else self.random_state.choice(top)
        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        # Diverse context: best (if not parent), a mid-tier, a low scorer, random.
        ctx: List[EvolvedProgram] = []
        seen = {parent.id}
        for cand in [scored[0], scored[len(scored) // 2], scored[-1]]:
            if cand.id not in seen:
                ctx.append(cand)
                seen.add(cand.id)
        while len(ctx) < n_ctx:
            c = self.random_state.choice(scored)
            if c.id not in seen:
                ctx.append(c)
                seen.add(c.id)
            elif len(seen) >= len(scored):
                break
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END