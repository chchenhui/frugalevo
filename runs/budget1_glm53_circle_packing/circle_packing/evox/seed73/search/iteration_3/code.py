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
    return float(v) if isinstance(v, (int, float)) else -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Stagnation-aware search: refine the best, occasionally diverge from
    under-explored high-tier programs, with diverse context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.use_counts: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnant: int = 0
        self.last_parent_id: Optional[str] = None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        s = _score(program)
        if s > self.best_score + 0.01:  # meaningful improvement only
            self.best_score = s
            self.stagnant = 0
        else:
            self.stagnant += 1
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        cands = [p for p in self.programs.values() if _score(p) >= 0]
        if not cands:
            cands = list(self.programs.values())
        cands.sort(key=_score, reverse=True)
        top = cands[: max(3, len(cands) // 4)]
        best = top[0]

        # avoid repeating the same parent back-to-back when possible
        def least_used(pool):
            return min(pool, key=lambda p: self.use_counts.get(p.id, 0))

        if self.stagnant >= 3 and self.random_state.random() < 0.4:
            # diverge from an under-used, decent program (not the overused best)
            pool = [p for p in cands[: len(cands) // 2] if p.id != self.last_parent_id]
            parent = least_used(pool) if pool else best
            label = self.DIVERGE_LABEL
            ctx = []
        elif self.stagnant >= 2:
            # refine the best (or a top-tier program if best overused)
            pool = [p for p in top if p.id != self.last_parent_id]
            parent = least_used(pool) if pool else best
            label = self.REFINE_LABEL
            ctx = []
        else:
            # default: weighted pick among top tier, prefer under-used
            parent = self.random_state.choice(top)
            label = ""
            rest = [p for p in cands if p.id != parent.id]
            self.random_state.shuffle(rest)
            # diverse context: a couple of top + a couple from elsewhere
            ctx = rest[:2] + [p for p in rest[2:] if _score(p) < 0.9][:2]
            ctx = ctx[:n_ctx]

        self.use_counts[parent.id] = self.use_counts.get(parent.id, 0) + 1
        self.last_parent_id = parent.id
        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END