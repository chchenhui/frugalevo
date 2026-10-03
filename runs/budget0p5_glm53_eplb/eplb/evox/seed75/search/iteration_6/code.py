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


def _score(p: Program) -> float:
    v = p.metrics.get("combined_score")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Simple adaptive strategy for a converged population.

    - Parent: rotated among top-tier programs (anti-reuse weighting).
    - Context: mixed tiers (top, mid, low outlier) for diverse perspectives.
    - Labels: mostly unused (they historically regress); occasional REFINE
      on the best program when deeply stuck.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_uses: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iters_since_improvement: int = 0
        self.sample_calls: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        s = _score(program)
        if s > self.best_score + 0.01:
            self.best_score = max(self.best_score, s)
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        pool = sorted(self.programs.values(), key=_score, reverse=True)

        # Top tier: top 25% of programs
        top_n = max(2, len(pool) // 4)
        top = pool[:top_n]

        # Parent: weighted by inverse usage among top tier
        weights = [1.0 / (1 + self.parent_uses.get(p.id, 0)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_uses[parent.id] = self.parent_uses.get(parent.id, 0) + 1

        # Context: mix of tiers, excluding parent
        others = [p for p in pool if p.id != parent.id]
        ctx: List[EvolvedProgram] = []
        if others:
            ctx.append(others[0])  # best non-parent
        mid = others[len(others) // 3: 2 * len(others) // 3]
        if mid:
            ctx.append(self.random_state.choice(mid))
        if len(others) >= 2:
            ctx.append(others[-1])  # low outlier for contrast
        rest = [p for p in others if p not in ctx]
        self.random_state.shuffle(rest)
        while len(ctx) < n_ctx and rest:
            ctx.append(rest.pop())
        ctx = ctx[:n_ctx]

        # Occasional REFINE on best program when deeply stuck (rare)
        label = ""
        if self.iters_since_improvement >= 4 and self.sample_calls % 5 == 0:
            best_p = pool[0]
            if self.parent_uses.get(best_p.id, 0) < 3:
                parent = best_p
                label = self.REFINE_LABEL
                ctx = []
                self.parent_uses[parent.id] = self.parent_uses.get(parent.id, 0) + 1

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END