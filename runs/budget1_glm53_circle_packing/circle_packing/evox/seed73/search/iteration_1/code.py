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
    """Exploit-best search with adaptive refinement/divergence.

    Population is clustered near the top; we mostly mutate top-tier programs
    with diverse context, and switch to REFINE/DIVERGE labels when stuck.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = float("-inf")
        self.iterations_since_improvement = 0
        self.parent_use_count: Dict[str, int] = {}

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return float("-inf")

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

        self.parent_use_count[program.parent_id] = (
            self.parent_use_count.get(program.parent_id, 0) + 1
        )

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        programs = sorted(
            self.programs.values(), key=self._score, reverse=True
        )
        scored = [p for p in programs if self._score(p) > float("-inf")]

        # Stagnation -> labeled moves
        label = ""
        parent = None
        if self.iterations_since_improvement >= 3 and len(scored) >= 2:
            if self.random_state.random() < 0.5:
                # Refine the current best (skip if overused)
                parent = scored[0]
                if self.parent_use_count.get(parent.id, 0) >= 3:
                    parent = scored[1]
                label = self.REFINE_LABEL
            else:
                # Diverge from a mid-tier program for a new direction
                mid = scored[len(scored) // 2: len(scored) * 3 // 4] or scored
                parent = self.random_state.choice(mid)
                label = self.DIVERGE_LABEL
            return {label: parent}, {"": []}

        # Default: exploit top tier, with variety to avoid overuse
        top = scored[: max(2, len(scored) // 3)]
        weights = [1.0 / (1 + self.parent_use_count.get(p.id, 0)) for p in top]
        if sum(weights) <= 0:
            weights = [1.0] * len(top)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]

        # Context: best program + diverse others (different score tiers)
        ctx = []
        if scored[0].id != parent.id:
            ctx.append(scored[0])
        rest = [p for p in scored if p.id != parent.id and p.id not in {c.id for c in ctx}]
        self.random_state.shuffle(rest)
        ctx.extend(rest[: n_ctx - len(ctx)])
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END