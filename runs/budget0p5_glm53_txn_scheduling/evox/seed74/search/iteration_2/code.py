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
    """Simple adaptive search: usage-capped parent selection from top tiers,
    mixed-score context, occasional divergence on stagnation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.diverge_count: int = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # Score-sorted population
        scored = sorted(candidates, key=self._score, reverse=True)

        # Stagnation-driven divergence: pick an underused mid-tier parent, fresh direction
        if self.stagnation >= 6 and self.diverge_count < 3:
            self.diverge_count += 1
            self.stagnation = 0
            pool = scored[len(scored) // 4: len(scored) * 3 // 4] or scored
            pool = [p for p in pool if self.parent_usage.get(p.id, 0) < 2] or pool
            parent = self.random_state.choice(pool)
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {"": []}

        # Normal mode: parent from top third, usage-capped (max 3 uses)
        top = scored[: max(1, len(scored) // 3)]
        fresh = [p for p in top if self.parent_usage.get(p.id, 0) < 3] or top
        parent = self.random_state.choice(fresh)
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: best program + diverse tiers (mid + one low), excluding parent
        ctx: List[EvolvedProgram] = []
        best = scored[0]
        if best.id != parent.id:
            ctx.append(best)
        rest = [p for p in scored if p.id != parent.id and p.id != best.id]
        if rest:
            mid = rest[len(rest) // 3: 2 * len(rest) // 3]
            low = rest[2 * len(rest) // 3:]
            for pool in (mid, low):
                if pool and len(ctx) < n_ctx:
                    pick = self.random_state.choice(pool)
                    if pick.id not in {c.id for c in ctx}:
                        ctx.append(pick)
            while len(ctx) < n_ctx and rest:
                pick = self.random_state.choice(rest)
                if pick.id not in {c.id for c in ctx}:
                    ctx.append(pick)
                else:
                    break
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END