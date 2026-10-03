# EVOLVE-BLOCK-START
import logging
import math
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
    """Refine-first search: exploit the top tier with varied context,
    occasionally diverge from under-used good parents when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.sample_count: int = 0

    # ---------- helpers ----------
    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def _is_meaningful(self, new: float, old: float) -> bool:
        if old <= 0:
            return new > old + 0.01
        return (new - old) > max(0.01, 0.01 * abs(old))

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score and self.best_score > -math.inf:
            if self._is_meaningful(s, self.best_score):
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = max(self.best_score, s)
        elif self.best_score == -math.inf:
            self.best_score = s
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

        self.sample_count += 1
        n_ctx = num_context_programs or 4
        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(3, len(scored) // 4)]  # top 25% (exploit tier)

        # Stuck for a while: mostly refine the best (reliable), occasionally
        # diverge from a different good-but-underused parent (breakthroughs).
        if self.stagnation >= 3:
            if self.stagnation % 4 == 3:
                # Diverge from a good parent that hasn't been over-used.
                pool = [p for p in top if self.parent_usage.get(p.id, 0) < 3]
                parent = self.random_state.choice(pool) if pool else scored[0]
                self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
                return {self.DIVERGE_LABEL: parent}, {"": []}
            # Refine the current best (or a near-best to vary lineage).
            parent = scored[0] if self.stagnation % 2 == 0 else scored[min(1, len(scored) - 1)]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.REFINE_LABEL: parent}, {"": []}

        # Normal mode: weighted pick among top tier, penalizing over-used parents.
        weights = [max(self._score(p), 0.01) / (1.0 + self.parent_usage.get(p.id, 0)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: mostly other top-tier programs (what works), plus one
        # diverse mid/low-tier program for a contrasting perspective.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if p in top]
        rest = [p for p in others if p not in top]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(rest)
        ctx = top_others[: max(1, n_ctx - 1)] + rest[:1]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END