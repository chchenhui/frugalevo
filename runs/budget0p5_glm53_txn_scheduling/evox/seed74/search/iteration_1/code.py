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


def _score(p) -> float:
    v = p.metrics.get("combined_score") if p.metrics else None
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Exploit-first sampling with stagnation-driven divergence."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iters_since_improve: int = 0
        self.sample_calls: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        s = _score(program)
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.iters_since_improve = 0
        else:
            self.iters_since_improve += 1

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
        k = num_context_programs or 4

        # Rank by score
        ranked = sorted(candidates, key=_score, reverse=True)
        top = ranked[: max(3, len(ranked) // 3)]

        # Stagnation: alternate DIVERGE (fresh direction from best) and REFINE (best program)
        label = ""
        if self.iters_since_improve >= 3:
            if self.iters_since_improve % 2 == 0:
                parent = ranked[0]
                label = self.REFINE_LABEL
            else:
                # Diverge from a good but not overused program
                pool = [p for p in top if self.usage_counts.get(p.id, 0) < 3] or top
                parent = self.random_state.choice(pool)
                label = self.DIVERGE_LABEL
            self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1
            if label:
                return {label: parent}, {}

        # Normal mode: exploit top tier but avoid overuse; occasionally explore mid tier
        if self.random_state.random() < 0.25 and len(ranked) > len(top):
            pool = ranked[len(top): 2 * len(top)]
        else:
            pool = [p for p in top if self.usage_counts.get(p.id, 0) < 4] or top

        parent = self.random_state.choice(pool)
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Context: mix of best programs (not parent) + one diverse lower-tier program
        others = [p for p in ranked if p.id != parent.id]
        ctx = others[: max(1, k - 1)]
        if len(ranked) > k + 2 and self.random_state.random() < 0.5:
            ctx = ctx + [self.random_state.choice(ranked[-5:])]
        ctx = ctx[:k]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END