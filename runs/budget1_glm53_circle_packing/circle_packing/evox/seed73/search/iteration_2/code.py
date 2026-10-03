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
    """Adaptive sampler: exploit top-tier parents with usage balancing,
    use REFINE/DIVERGE labels only when stagnating."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.last_label: str = ""

    @staticmethod
    def _score(p: Program) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)) and v >= 0:
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + max(0.01, 0.01 * self.best_score):
            self.best_score = s
            self.stagnation = 0
        else:
            self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id}")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        k = num_context_programs or 4

        # Track usage of parents (via parent_id of children already produced).
        for p in candidates:
            if p.parent_id and p.parent_id not in self.usage_counts:
                self.usage_counts[p.parent_id] = 1

        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(3, len(scored) // 4)]  # top quartile
        mid = scored[len(top): max(len(top) + 5, len(scored) // 2)]

        label = ""
        if self.stagnation >= 3:
            # Alternate: refine the best, then diverge on a mid-tier parent.
            if self.last_label != "REFINE":
                label, parent = "REFINE", scored[0]
            else:
                label = "DIVERGE"
                pool = mid if mid else top
                parent = self.random_state.choice(pool)
            self.last_label = "REFINE" if label == "REFINE" else "DIVERGE"
        else:
            # Weighted pick among top tier, favoring less-used programs.
            weights = [1.0 / (1 + self.usage_counts.get(p.id, 0)) for p in top]
            parent = self.random_state.choices(top, weights=weights, k=1)[0]

        # Context: best program (if not parent) + diverse mid/low programs.
        context: List[EvolvedProgram] = []
        for p in scored:
            if p.id != parent.id and len(context) < 2:
                context.append(p)
        rest = [p for p in scored if p.id != parent.id and p not in context]
        self.random_state.shuffle(rest)
        while rest and len(context) < k:
            context.append(rest.pop())

        if label:
            return {label: parent}, {"": []}
        return {"": parent}, {"": context[:k]}


# EVOLVE-BLOCK-END