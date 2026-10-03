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
    """Search strategy focused on top-tier exploitation with diversity-aware
    context selection and adaptive stagnation handling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.last_parent_id: Optional[str] = None

    @staticmethod
    def _score(p) -> float:
        v = p.metrics.get("combined_score")
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
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.last_parent_id is not None:
            self.parent_usage[self.last_parent_id] = (
                self.parent_usage.get(self.last_parent_id, 0) + 1
            )
        self.last_parent_id = None

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

        n_ctx = num_context_programs or 4

        # Score-weighted parent pick among top half, penalizing overused parents.
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_pool = ranked[: max(4, len(ranked) // 2)]
        weights = []
        for p in top_pool:
            s = max(self._score(p), 0.0)
            w = (0.1 + s) ** 3
            w /= 1.0 + 2 * self.parent_usage.get(p.id, 0)
            weights.append(w)
        parent = self.random_state.choices(top_pool, weights=weights, k=1)[0]
        self.last_parent_id = parent.id

        # Context: top scorers + a couple of diverse mid/low-tier for contrast.
        context: List[EvolvedProgram] = [p for p in ranked[:3] if p.id != parent.id]
        rest = [p for p in ranked if p.id != parent.id and p not in context]
        self.random_state.shuffle(rest)
        context.extend(rest[: max(0, n_ctx - len(context))])
        context = context[:n_ctx]

        label = ""
        if self.stagnation >= 5:
            r = self.random_state.random()
            if r < 0.4 and ranked:
                # Refine current best approach.
                parent = ranked[0]
                self.last_parent_id = parent.id
                label = self.REFINE_LABEL
                context = [p for p in ranked[1:4]]
            elif r < 0.6:
                # Try a fundamentally different direction from a mid-tier program.
                mid = ranked[len(ranked) // 3: 2 * len(ranked) // 3]
                if mid:
                    parent = self.random_state.choice(mid)
                    self.last_parent_id = parent.id
                label = self.DIVERGE_LABEL
                context = []

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END