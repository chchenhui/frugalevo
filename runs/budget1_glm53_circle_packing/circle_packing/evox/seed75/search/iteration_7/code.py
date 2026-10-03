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
    """Search strategy tuned for a converged population near optimum.

    Key idea: repeated REFINE on the same top parents yielded nothing, while
    DIVERGE on underused mid/low-tier parents produced the only real gains.
    So we rotate parents by usage count and prefer DIVERGE on fresh parents,
    REFINE on the current best only sparingly.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = 0.0
        self.iters_since_improvement: int = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        # Track parent usage (parent selection state lives here for resume safety)
        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        score = self._score(program)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

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

        scored = [(self._score(p), p) for p in candidates]
        scored.sort(key=lambda x: -x[0])
        top = [p for s, p in scored[:10] if s > 0.0]
        mid = [p for s, p in scored if 0.5 < s <= 0.99]
        valid = top or mid or candidates

        # Least-used-first rotation to avoid overusing any single parent
        def usage(p):
            return self.parent_usage.get(p.id, 0)

        # Stagnation: strongly favor DIVERGE on a fresh (least-used) parent
        if self.iters_since_improvement > 5 and self.random_state.random() < 0.7:
            pool = sorted(valid, key=usage)[:6]
            parent = self.random_state.choice(pool)
            label = self.DIVERGE_LABEL
            examples: List[EvolvedProgram] = []
        else:
            r = self.random_state.random()
            if r < 0.5 and top:
                # Refine near the best, but rotate among top scorers
                parent = self.random_state.choice(top[:5])
                label = self.REFINE_LABEL
                examples = []
            else:
                # Diverse parent from mid-tier, no label
                pool = mid if mid else valid
                parent = self.random_state.choice(sorted(pool, key=usage)[:8])
                label = ""
                ctx_pool = [p for p in valid if p.id != parent.id]
                self.random_state.shuffle(ctx_pool)
                examples = ctx_pool[: num_context_programs or 4]

        return {label: parent}, {"": examples}


# EVOLVE-BLOCK-END