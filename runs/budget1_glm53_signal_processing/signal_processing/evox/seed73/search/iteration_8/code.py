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
    """Fresh-parent rotation with usage penalties and stagnation-driven labels.

    Key ideas for a plateaued population:
    - Never reuse a parent too often (usage-weighted sampling over the top tier).
    - On stagnation, REFINE a fresh top-tier parent (refinement of near-best
      programs historically produced the best children), occasionally DIVERGE
      from a different high scorer to seek a new direction.
    - Context mixes top performers with diverse mid/low-tier programs.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.label_counter: int = 0

    # ---------- helpers ----------
    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def _is_meaningful(self, new: float, old: float) -> bool:
        if old <= 0:
            return new > old + 0.01
        return (new - old) > max(0.01, 0.01 * abs(old))

    def _pick_fresh_top(self, scored: List[EvolvedProgram]) -> EvolvedProgram:
        """Pick among top tier, heavily penalizing already-used parents."""
        top = scored[: max(3, len(scored) // 4)]
        weights = []
        for p in top:
            usage = self.parent_usage.get(p.id, 0)
            w = max(self._score(p), 0.01) / (1.0 + 5.0 * usage)
            weights.append(w)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        return parent

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            meaningful = (
                self._is_meaningful(s, self.best_score)
                if self.best_score > -math.inf
                else True
            )
            self.best_score = max(self.best_score, s)
            self.stagnation = 0 if meaningful else self.stagnation + 1
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

        # Stagnation: alternate REFINE on a fresh top parent and DIVERGE on a
        # different fresh top scorer (avoid same-parent label reuse).
        if self.stagnation >= 3:
            self.label_counter += 1
            parent = self._pick_fresh_top(scored)
            if self.label_counter % 3 == 0:
                return {self.DIVERGE_LABEL: parent}, {"": []}
            return {self.REFINE_LABEL: parent}, {"": []}

        # Normal mode: fresh-parent rotation among top tier.
        parent = self._pick_fresh_top(scored)

        # Context: half top-tier (excluding parent), half diverse mid/low tier.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if self._score(p) >= 0.65]
        rest = [p for p in others if self._score(p) < 0.65]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(rest)
        half = max(1, n_ctx // 2)
        ctx = top_others[:half] + rest[: n_ctx - half]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END