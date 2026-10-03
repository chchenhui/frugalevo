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
    """Top-tier focused search with usage balancing and stagnation labels.

    Principle 1: Exploit the top tier — the best programs are the most
    promising bases for mutation, but usage counts prevent over-exploiting
    any single program.
    Principle 2: When stuck, explicitly REFINE the best (its approach is
    clearly promising) and only occasionally DIVERGE when refinement stalls
    for a long stretch.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0

    # ---------- helpers ----------
    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def _is_meaningful(self, new: float, old: float) -> bool:
        if old <= 0 or old == -math.inf:
            return True
        return (new - old) > max(0.01, 0.01 * abs(old))

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            if self._is_meaningful(s, self.best_score):
                self.stagnation = 0
            else:
                self.stagnation += 1
            self.best_score = max(self.best_score, s)
        else:
            self.stagnation += 1

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

        n_ctx = num_context_programs or 4
        scored = sorted(candidates, key=self._score, reverse=True)

        # Stagnation ladder: refine best first, then diverge, then diversify.
        if self.stagnation >= 6:
            # Deeply stuck: diverge from best to seek a new direction.
            return {self.DIVERGE_LABEL: scored[0]}, {"": []}
        if self.stagnation >= 3:
            # Promising best needs refinement to reach its potential.
            return {self.REFINE_LABEL: scored[0]}, {"": []}

        # Normal mode: parent from top third, weighted by score and
        # inversely by usage to avoid determinism.
        top = scored[: max(1, len(scored) // 3)]
        weights = [
            max(self._score(p), 0.01) / (1.0 + self.parent_usage.get(p.id, 0))
            for p in top
        ]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: mostly top-tier (contrasting good approaches) plus one
        # diverse lower-tier program for a different perspective.
        others = [p for p in candidates if p.id != parent.id]
        top_ctx = [p for p in others if p in top]
        rest = [p for p in others if p not in top]
        self.random_state.shuffle(top_ctx)
        self.random_state.shuffle(rest)
        ctx = top_ctx[: max(1, n_ctx - 1)] + rest[:1]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END