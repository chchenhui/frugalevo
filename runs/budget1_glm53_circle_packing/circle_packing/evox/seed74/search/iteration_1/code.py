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
    """Score-aware adaptive search.

    Principles:
    1. Exploit top-tier programs as parents (avoid wasting iterations on
       low scorers), but rotate among them to avoid overuse.
    2. On stagnation, alternate REFINE on the best program with occasional
       DIVERGE attempts, and use diverse context (top + distinct-score
       programs) to give the LLM complementary perspectives.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.stagnation = 0
        self.parent_usage: Dict[str, int] = {}
        self.last_parent_id: Optional[str] = None

    def _score(self, program) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        s = self._score(program)
        if self.best_score is None or s > self.best_score:
            improved = self.best_score is None or (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score))
            self.best_score = s
            if improved:
                self.stagnation = 0
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

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
        top = [p for p in scored if self._score(p) >= 0.9 * self._score(scored[0])]

        # Rotate among top-tier parents, avoid repeating the last one.
        pool = [p for p in top if p.id != self.last_parent_id] or top
        # Prefer less-used parents.
        pool.sort(key=lambda p: self.parent_usage.get(p.id, 0))
        front_pool = [p for p in pool if self.parent_usage.get(p.id, 0) <= 2] or pool
        parent = self.random_state.choice(front_pool[: max(3, len(front_pool) // 2)])
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        self.last_parent_id = parent.id

        # Context: best program + diverse distinct-score programs.
        n_ctx = num_context_programs or 4
        context: List[EvolvedProgram] = []
        seen_scores = set()
        for p in scored:
            s = round(self._score(p), 4)
            if p.id == parent.id or s in seen_scores:
                continue
            seen_scores.add(s)
            context.append(p)
            if len(context) >= n_ctx:
                break
        if len(context) < n_ctx:
            for p in scored:
                if p.id != parent.id and all(c.id != p.id for c in context):
                    context.append(p)
                    if len(context) >= n_ctx:
                        break

        label = ""
        if self.stagnation >= 3:
            # Stuck: mostly refine the best, occasionally diverge.
            if self.stagnation % 4 == 3:
                label = self.DIVERGE_LABEL
                parent = scored[0]
                context = []
            else:
                label = self.REFINE_LABEL
                parent = scored[0]
                context = []

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END