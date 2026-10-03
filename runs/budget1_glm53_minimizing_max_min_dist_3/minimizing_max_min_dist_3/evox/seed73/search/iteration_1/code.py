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
    """Search strategy focused on refining the top tier of a converged population.

    The population is near-plateau; most gains come from mutating top programs
    (REFINE) with occasional DIVERGE attempts to escape local optima. Context
    mixes the best program with diverse mid/high scorers for inspiration.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -1.0
        self.stagnation = 0
        self.parent_use_count: Dict[str, int] = {}

    def _score(self, program) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.0001:
            self.best_score = s
            self.stagnation = 0
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

        scored = sorted(candidates, key=self._score, reverse=True)
        n = len(scored)

        # Occasionally diverge when deeply stuck, otherwise exploit top tier.
        if self.stagnation >= 6 and self.random_state.random() < 0.35:
            # Diverge from a good-but-not-best program to explore new structure.
            pool = scored[: max(2, n // 2)]
            parent = self.random_state.choice(pool)
            parent_dict = {self.DIVERGE_LABEL: parent}
            return parent_dict, {"": []}

        # Exploit: pick from top tier, avoiding overuse of any single program.
        top = scored[: max(3, n // 3)]
        # Prefer less-used parents within the top tier.
        top.sort(key=lambda p: (self.parent_use_count.get(p.id, 0), -self._score(p)))
        parent = top[0] if self.random_state.random() < 0.6 else self.random_state.choice(top)
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Refine label when stuck for a few iterations to push the best further.
        label = ""
        if self.stagnation >= 3 and self._score(parent) >= self.best_score - 0.005:
            label = self.REFINE_LABEL

        # Context: best program + diverse mid/high scorers (different perspectives).
        k = num_context_programs or 4
        context: List[EvolvedProgram] = []
        best_prog = scored[0]
        if best_prog.id != parent.id:
            context.append(best_prog)
        # Diverse band: sample across the upper half of the population.
        band = scored[1 : max(4, n // 2)] or scored[1:]
        pool = [p for p in band if p.id != parent.id and p.id not in {c.id for c in context}]
        self.random_state.shuffle(pool)
        context.extend(pool[: k - len(context)])

        parent_dict = {label: parent}
        return parent_dict, {"": context[:k]}


# EVOLVE-BLOCK-END