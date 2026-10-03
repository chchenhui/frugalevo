# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Exploit-first search with stagnation-triggered divergence.

    Parents are drawn from the top score tier (weighted), context comes from
    other high scorers plus an occasional diverse mid-tier program. When
    progress stalls, alternate REFINE on the best program with DIVERGE on
    a good-but-not-best program.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.best_id: Optional[str] = None
        self.iterations_since_improvement: int = 0
        self.sample_counter: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        # meaningful improvement threshold: 1% relative or 0.01 absolute
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.best_id = program.id
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

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

        self.sample_counter += 1
        n_ctx = num_context_programs or 4

        scored = sorted(candidates, key=self._score, reverse=True)
        top_tier = scored[: max(4, len(scored) // 4)]  # top quartile

        stagnating = self.iterations_since_improvement >= 6

        if stagnating:
            # Alternate: refine the best, or diverge from a good alternate
            if self.sample_counter % 2 == 0 and self.best_id and self.best_id in self.programs:
                parent = self.programs[self.best_id]
                return {self.REFINE_LABEL: parent}, {self.REFINE_LABEL: []}
            # diverge from a top-tier program that isn't the current best
            pool = [p for p in top_tier if p.id != self.best_id]
            if not pool:
                pool = top_tier
            parent = self.random_state.choice(pool)
            return {self.DIVERGE_LABEL: parent}, {self.DIVERGE_LABEL: []}

        # Normal mode: exploit top tier, weighted toward better scores,
        # with mild usage penalty to avoid over-focusing on one program.
        weights = []
        for p in top_tier:
            u = self.usage_counts.get(p.id, 0)
            w = max(0.05, self._score(p) + 0.01) / (1.0 + u)
            weights.append(w)
        parent = self.random_state.choices(top_tier, weights=weights, k=1)[0]
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Context: other top-tier programs + one diverse mid/low program
        ctx = [p for p in top_tier if p.id != parent.id]
        self.random_state.shuffle(ctx)
        ctx = ctx[: max(0, n_ctx - 1)]
        rest = [p for p in scored[len(top_tier):] if p.id != parent.id]
        if rest:
            ctx.append(self.random_state.choice(rest))
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END