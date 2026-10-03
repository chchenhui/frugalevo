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
    """Diversified top-tier parent rotation with stagnation-driven labels.

    Key ideas:
    1. Rotate parents across the whole top tier (not just the single best),
       penalizing over-used parents, since repeatedly mutating the best
       produced only plateau-level children.
    2. On stagnation, cycle REFINE / DIVERGE on under-used top-tier programs,
       with diverse context (top + contrasting mid/low tier) in normal mode.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.label_counter: int = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            improved = (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)) \
                if self.best_score > -math.inf else True
            self.best_score = max(self.best_score, s)
            self.stagnation = 0 if improved else self.stagnation + 1
        else:
            self.stagnation += 1

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

        n_ctx = num_context_programs or 4
        scored = sorted(candidates, key=self._score, reverse=True)
        top = scored[: max(3, len(scored) // 3)]

        # Stagnation: cycle REFINE / DIVERGE on *under-used* top-tier programs
        # (not always the single best, which has been exhausted).
        if self.stagnation >= 3:
            self.label_counter += 1
            underused = sorted(top, key=lambda p: self.parent_usage.get(p.id, 0))
            target = underused[0] if self.label_counter % 3 else scored[0]
            label = self.REFINE_LABEL if self.label_counter % 2 else self.DIVERGE_LABEL
            self.parent_usage[target.id] = self.parent_usage.get(target.id, 0) + 1
            return {label: target}, {"": []}

        # Normal mode: weighted pick across top tier, penalizing reuse.
        weights = [max(self._score(p), 0.01) / (1.0 + self.parent_usage.get(p.id, 0)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: half from top tier (excluding parent), half diverse mid/low.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if p in top]
        rest = [p for p in others if p not in top]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(rest)
        ctx = top_others[: max(1, n_ctx // 2)] + rest[: n_ctx - min(len(top_others), max(1, n_ctx // 2))]
        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END