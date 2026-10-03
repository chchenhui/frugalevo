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
    """Score-aware sampling: exploit top programs while rotating parents
    and providing diverse high-quality context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4

        # Tier 1: programs within 1% of best; tier 2: within 10%.
        best = max(self._score(p) for p in candidates)
        top = [p for p in candidates if self._score(p) >= best * 0.99]
        mid = [p for p in candidates if best * 0.9 <= self._score(p) < best * 0.99]
        pool = top if top else candidates

        # Rotate parents: prefer least-used top programs.
        pool_sorted = sorted(pool, key=lambda p: (self.usage_count.get(p.id, 0), self.random_state.random()))
        parent = pool_sorted[0] if pool_sorted else self.random_state.choice(pool)
        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        # Context: mix of other top programs and a couple of mid-tier for diversity.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if p in pool]
        mid_others = [p for p in others if p in mid]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(mid_others)
        ctx = top_others[: max(1, n_ctx - 2)] + mid_others[:2]
        if len(ctx) < n_ctx:
            rest = [p for p in others if p not in ctx]
            self.random_state.shuffle(rest)
            ctx.extend(rest[: n_ctx - len(ctx)])
        ctx = ctx[:n_ctx]

        # If deeply stagnant, occasionally ask for divergence from a top parent.
        label = ""
        if self.stagnation >= 5 and self.random_state.random() < 0.3:
            label = self.DIVERGE_LABEL
            ctx = []
            self.stagnation = 0

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END