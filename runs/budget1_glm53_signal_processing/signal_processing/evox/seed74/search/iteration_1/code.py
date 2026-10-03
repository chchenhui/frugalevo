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
    """Adaptive parent/context selection with stagnation-driven labels."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.sample_calls: int = 0

    def _score(self, program) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        self.usage_count.setdefault(program.id, 0)

        s = self._score(program)
        # meaningful improvement threshold: >0.01 absolute
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) >= 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        # --- parent selection ---
        candidates.sort(key=self._score, reverse=True)
        top = candidates[: max(3, len(candidates) // 3)]

        if self.stagnation >= 4 and self.sample_calls % 3 == 0:
            # Stuck: alternate between refining the best and diverging from a mid-tier program.
            if self.sample_calls % 6 == 0:
                parent = self.random_state.choice(candidates[len(candidates) // 2:])
                return {self.DIVERGE_LABEL: parent}, {"": []}
            else:
                parent = candidates[0]
                return {self.REFINE_LABEL: parent}, {"": []}

        # Normal: pick from top tier, penalizing overused parents.
        weights = [1.0 / (1 + self.usage_count.get(p.id, 0)) for p in top]
        if sum(weights) <= 0:
            weights = [1.0] * len(top)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]

        # --- context selection: best program + diverse others (different score ranges) ---
        others = [p for p in candidates if p.id != parent.id]
        context: List[EvolvedProgram] = []
        best_other = others[0] if others else None
        if best_other and best_other.id != parent.id:
            context.append(best_other)
        rest = others[1:] if best_other else others
        self.random_state.shuffle(rest)
        # prefer less-used programs for context diversity
        rest.sort(key=lambda p: self.usage_count.get(p.id, 0))
        context.extend(rest[: max(0, n_ctx - len(context))])
        context = context[:n_ctx]

        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END