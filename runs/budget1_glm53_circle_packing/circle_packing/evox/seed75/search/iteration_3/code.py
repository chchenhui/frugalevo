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
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iterations_since_improvement: int = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)) and isinstance(v, bool) is False:
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
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

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

        # Top tier: score >= 0.99 of best (valid high scorers)
        best = max(self._score(p) for p in candidates)
        top = [p for p in candidates if self._score(p) >= max(0.0, best - 0.002)]
        others = [p for p in candidates if p not in top]

        # Rotate parents: prefer least-used top-tier programs
        top.sort(key=lambda p: (self.parent_usage.get(p.id, 0), self.random_state.random()))
        parent = top[0]
        label = ""

        # Occasionally diverge from a low-usage mid-tier program when deeply stuck
        if self.iterations_since_improvement >= 6 and others and self.random_state.random() < 0.3:
            mid = [p for p in others if 0.1 <= self._score(p) <= 0.9]
            if mid:
                parent = self.random_state.choice(mid)
                label = self.DIVERGE_LABEL
                return {label: parent}, {}

        # If refining near best for a while, use REFINE on best
        if self.iterations_since_improvement >= 3 and self.random_state.random() < 0.5:
            best_p = max(candidates, key=self._score)
            if self.parent_usage.get(best_p.id, 0) < 5:
                parent = best_p
                label = self.REFINE_LABEL
                return {label: parent}, {}

        # Context: mix of other top-tier variants + one low scorer for contrast
        ctx = [p for p in top if p.id != parent.id]
        self.random_state.shuffle(ctx)
        context = ctx[: max(1, n_ctx - 1)]
        low = [p for p in others if 0.0 < self._score(p) < 0.9]
        if low:
            context.append(self.random_state.choice(low))
        context = context[:n_ctx]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END