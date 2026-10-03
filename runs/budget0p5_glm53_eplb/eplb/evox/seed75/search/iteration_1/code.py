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
    """Adaptive search: exploit top programs with REFINE, diversify context,
    and force DIVERGE when stagnating."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.best_id = None
        self.stagnation = 0
        self.usage_counts: Dict[str, int] = {}
        self.sample_calls = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        s = self._score(program)
        if s > self.best_score:
            # meaningful improvement threshold
            if self.best_score == -float("inf") or (s - self.best_score) > 0.0005:
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = s
            self.best_id = program.id
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

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        # score-sorted population
        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(1, len(scored) // 3)]
        rest = scored[len(top):]

        # usage-aware parent selection: mostly top-tier, sometimes mid/low
        for p in candidates:
            self.usage_counts.setdefault(p.id, 0)

        if self.stagnation >= 4 and self.sample_calls % 3 == 0:
            # deeply stuck: diverge from a less-used program
            pool = sorted(candidates, key=lambda p: self.usage_counts.get(p.id, 0))
            parent = self.random_state.choice(pool[: max(2, len(pool) // 2)])
            self.usage_counts[parent.id] += 1
            return {self.DIVERGE_LABEL: parent}, {"": []}

        if self.stagnation >= 2 and self.best_id is not None:
            # stagnating but promising: refine the best
            parent = self.programs[self.best_id]
            self.usage_counts[parent.id] += 1
            ctx = [p for p in top if p.id != parent.id][:n_ctx]
            return {self.REFINE_LABEL: parent}, {"": ctx}

        # default: usage-weighted choice from top tier
        weights = [1.0 / (1 + self.usage_counts.get(p.id, 0)) for p in top]
        if sum(weights) <= 0:
            parent = self.random_state.choice(top)
        else:
            parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.usage_counts[parent.id] += 1

        # diverse context: mix of high scorers and low scorers (contrasts)
        ctx = []
        others = [p for p in candidates if p.id != parent.id]
        highs = [p for p in others if self._score(p) >= self._score(parent)][:2]
        lows = [p for p in others if self._score(p) < self._score(parent)]
        self.random_state.shuffle(lows)
        ctx.extend(highs)
        ctx.extend(lows[: max(0, n_ctx - len(ctx))])
        if len(ctx) < n_ctx:
            mids = [p for p in others if p not in ctx]
            self.random_state.shuffle(mids)
            ctx.extend(mids[: n_ctx - len(ctx)])

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END