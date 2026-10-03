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
    """Diverse rotation strategy: avoid reusing parents/contexts, mix score
    tiers for context, and only rarely use labels (DIVERGE on stagnation)."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iters_since_improvement: int = 0
        self.diverge_counter: int = 0

    @staticmethod
    def _score(p: Program) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        s = self._score(program)
        if s > self.best_score + 0.01:
            self.best_score = max(self.best_score, s)
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def _weighted_pick(self, pool: List[EvolvedProgram], use_count: Dict[str, int]) -> EvolvedProgram:
        # Weight by score mildly, penalize reuse heavily.
        weights = []
        for p in pool:
            s = self._score(p)
            base = max(s, 0.001)
            use = use_count.get(p.id, 0)
            weights.append(base / (1.0 + 3.0 * use))
        return self.random_state.choices(pool, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) > 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4

        # Occasionally diverge when deeply stuck (but not every iteration).
        if self.iters_since_improvement >= 3 and self.diverge_counter < 2:
            self.diverge_counter += 1
            parent = self._weighted_pick(candidates, self.parent_use_count)
            self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}

        self.diverge_counter = 0

        # Parent: prefer upper-half scores but with anti-reuse weighting.
        scored = sorted(candidates, key=self._score, reverse=True)
        parent_pool = scored[: max(3, len(scored) // 2)]
        parent = self._weighted_pick(parent_pool, self.parent_use_count)
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Context: mix tiers (top, middle, low) for diverse perspectives,
        # penalizing previously used context programs.
        others = [p for p in candidates if p.id != parent.id]
        if not others:
            return {"": parent}, {"": []}
        k = min(n_ctx, len(others))
        top = others[: max(1, k // 2)]
        rest = others[max(1, k // 2):]
        picks: List[EvolvedProgram] = []
        if top:
            picks.append(self._weighted_pick(top, self.context_use_count))
        if rest and len(picks) < k:
            self.random_state.shuffle(rest)
            picks.extend(rest[: k - len(picks)])
        for p in picks:
            self.context_use_count[p.id] = self.context_use_count.get(p.id, 0) + 1

        return {"": parent}, {"": picks}


# EVOLVE-BLOCK-END