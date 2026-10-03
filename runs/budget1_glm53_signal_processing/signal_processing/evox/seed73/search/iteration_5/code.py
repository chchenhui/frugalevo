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
    """Diversity-first search for a plateaued population.

    The best program has been over-mutated with no gains, so we avoid it as a
    parent and instead mutate under-used high/mid-tier programs, providing
    the best program plus a diverse context (including failures) as reference.
    Occasionally we diverge from a mid-tier program to seek new directions.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.sample_count: int = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            if self.best_score > -math.inf and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)):
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = max(self.best_score, s)
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
        best = scored[0]
        best_id = best.id

        # Under-used, decent-scoring parents (exclude the over-used best).
        pool = [p for p in scored if p.id != best_id and self._score(p) > 0.0]
        if not pool:
            pool = scored

        # Weight by score but heavily penalize usage for diversity.
        weights = [max(self._score(p), 0.01) / (1.0 + 3 * self.parent_usage.get(p.id, 0)) for p in pool]
        parent = self.random_state.choices(pool, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Every 4th sample in stagnation: diverge from an under-used mid-tier program.
        if self.stagnation >= 4 and self.sample_count % 4 == 0:
            mid = [p for p in pool if 0.55 <= self._score(p) < 0.69]
            if mid:
                diverge_parent = self.random_state.choice(mid)
                self.parent_usage[diverge_parent.id] = self.parent_usage.get(diverge_parent.id, 0) + 1
                return {self.DIVERGE_LABEL: diverge_parent}, {"": []}

        # Context: current best (as target to beat) + diverse others.
        others = [p for p in candidates if p.id != parent.id]
        high = [p for p in others if self._score(p) >= 0.65 and p.id != best_id]
        mid = [p for p in others if 0.5 <= self._score(p) < 0.65]
        low = [p for p in others if 0.0 < self._score(p) < 0.5]
        self.random_state.shuffle(high)
        self.random_state.shuffle(mid)
        self.random_state.shuffle(low)
        ctx = [best] + high[:1] + mid[:1] + low[:1]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END