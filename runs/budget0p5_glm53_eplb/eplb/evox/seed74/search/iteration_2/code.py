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
    """Exploit-heavy strategy: sample parents from the top tier, mix diverse
    context, and use REFINE/DIVERGE labels when progress stagnates."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.last_improve_iter = 0
        self.iteration = 0
        self.usage_counts: Dict[str, int] = {}

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.iteration = max(self.iteration, iteration)
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if self.best_score is None or s > self.best_score + 0.0001:
            self.best_score = s
            self.last_improve_iter = self.iteration

        if program.parent_id:
            self.usage_counts[program.parent_id] = self.usage_counts.get(program.parent_id, 0) + 1

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

        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(3, len(scored) // 3)]

        # Prefer less-used parents to avoid overuse.
        def usage(p):
            return self.usage_counts.get(p.id, 0)

        stale = (self.iteration - self.last_improve_iter) >= 3

        label = ""
        parent = None
        if stale:
            # Alternate: refine the best, or diverge from a decent underused program.
            if self.random_state.random() < 0.5 and self.best_score is not None:
                parent = scored[0]
                label = self.REFINE_LABEL
            else:
                pool = [p for p in top if self._score(p) > 0.5 * (scored[0].metrics.get("combined_score") or 1)]
                pool = [p for p in pool if usage(p) <= 1] or pool or top
                parent = self.random_state.choice(pool)
                label = self.DIVERGE_LABEL
            parent_dict = {label: parent}
            return parent_dict, {"": []}

        # Normal mode: weighted pick from top tier, favoring low usage.
        weights = [1.0 / (1 + usage(p)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]

        # Context: best program plus diverse mid-tier programs.
        n = num_context_programs or 4
        ctx = []
        if scored[0].id != parent.id:
            ctx.append(scored[0])
        rest = [p for p in scored if p.id != parent.id and p.id not in {c.id for c in ctx}]
        self.random_state.shuffle(rest)
        for p in rest:
            if len(ctx) >= n:
                break
            ctx.append(p)

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END