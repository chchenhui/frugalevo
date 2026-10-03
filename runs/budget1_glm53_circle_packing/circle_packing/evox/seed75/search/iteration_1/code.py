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
    """Best-first search with stagnation-triggered refinement/divergence.

    Population is near-optimal (0.9916 / ratio ~1.0). Strategy: exploit the
    best programs, provide diverse top-tier context, and use REFINE/DIVERGE
    labels when no meaningful improvement happens for several iterations.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.stagnation = 0
        self.best_score = -1.0
        self.parent_use_count: Dict[str, int] = {}

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        s = self._score(program)
        if s > self.best_score + 0.01:  # meaningful improvement
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id} (score={s}, stagnation={self.stagnation})")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(3, len(scored) // 3)]

        # Stagnating: alternate refine best / diverge from underused parent
        if self.stagnation >= 3:
            if self.stagnation % 3 == 2:
                # Diverge from a mid/low-tier parent that hasn't been overused
                pool = scored[len(scored) // 3:] or scored
                pool = sorted(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
                parent = pool[self.random_state.randrange(min(3, len(pool)))]
                self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
                return {self.DIVERGE_LABEL: parent}, {}
            # Refine the best program
            parent = scored[self.random_state.randrange(min(2, len(scored)))]
            self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
            return {self.REFINE_LABEL: parent}, {}

        # Default: parent from top tier, least-used first with randomness
        top_sorted = sorted(top, key=lambda p: self.parent_use_count.get(p.id, 0))
        parent = top_sorted[self.random_state.randrange(min(3, len(top_sorted)))]

        # Context: mix of other top-tier programs + one diverse mid-tier program
        ctx = [p for p in top if p.id != parent.id]
        self.random_state.shuffle(ctx)
        if len(ctx) < n_ctx:
            rest = [p for p in scored if p.id != parent.id and p not in ctx]
            self.random_state.shuffle(rest)
            ctx.extend(rest)
        examples = ctx[:n_ctx]

        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
        return {"": parent}, {"": examples}


# EVOLVE-BLOCK-END