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
    """Adaptive search database.

    Simple strategy:
    - Normally: pick a good (top-tier) but under-used parent, give it diverse
      context (best program + a mid/low scorer + a random program).
    - On stagnation: alternate DIVERGE (on a fresh, under-used, decent-scoring
      program with empty context) and REFINE (on the current best).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation = 0
        self.parent_usage: Dict[str, int] = {}
        self.diverge_count = 0
        self.refine_count = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score", 0.0)
        return float(v) if isinstance(v, (int, float)) else 0.0

    def _update_best_program(self, program: EvolvedProgram) -> None:
        score = self._score(program)
        # meaningful improvement: >1% relative or >0.01 absolute
        if score > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = score
            self.stagnation = 0
        else:
            self.stagnation += 1

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1
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
        n_ctx = num_context_programs or 0

        scored = sorted(candidates, key=self._score, reverse=True)
        best = scored[0]

        # --- Stagnation-triggered labeled moves ---
        if self.stagnation >= 3:
            if self.diverge_count <= self.refine_count:
                # DIVERGE from a fresh, under-used, decent program
                pool = [p for p in scored[: len(scored) // 2]
                        if self.parent_usage.get(p.id, 0) <= 1]
                if not pool:
                    pool = scored[: max(1, len(scored) // 2)]
                parent = self.random_state.choice(pool)
                self.diverge_count += 1
                return {self.DIVERGE_LABEL: parent}, {"": []}
            else:
                # REFINE the current best (but not repeatedly if it fails)
                self.refine_count += 1
                return {self.REFINE_LABEL: best}, {"": []}

        # --- Default: exploit top tier with under-use preference, diverse context ---
        top = scored[: max(1, len(scored) // 4)]
        usage = [self.parent_usage.get(p.id, 0) for p in top]
        weights = [1.0 / (1 + u) for u in usage]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]

        ctx: List[EvolvedProgram] = []
        if best.id != parent.id:
            ctx.append(best)
        # a mid/low scorer for a different perspective
        rest = [p for p in candidates if p.id != parent.id and p.id != best.id]
        if rest:
            low = sorted(rest, key=self._score)[: max(1, len(rest) // 3)]
            ctx.append(self.random_state.choice(low))
            others = [p for p in rest if p.id != ctx[-1].id]
            self.random_state.shuffle(others)
            ctx.extend(others[: max(0, n_ctx - len(ctx))])
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END