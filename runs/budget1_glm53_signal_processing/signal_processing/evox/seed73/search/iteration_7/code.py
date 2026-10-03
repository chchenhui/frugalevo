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
    """Plateau-aware search.

    The population is near a plateau: the top ~20 programs are within 0.012 of
    the best, so breakthroughs must come from repeated refinement of *varied*
    near-best parents (not the single best, which is over-used). We rotate
    through the top tier with usage penalties, give diverse context (top-tier
    + mid-tier contrast), and occasionally diverge to escape local optima.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.step: int = 0

    # ---------- helpers ----------
    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def _is_meaningful(self, new: float, old: float) -> bool:
        if old <= 0 or old == -math.inf:
            return new > old + 0.01
        return (new - old) > max(0.01, 0.01 * abs(old))

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score and self._is_meaningful(s, self.best_score):
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.best_score = max(self.best_score, s)

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
        self.step += 1

        # Top tier: enough to cover the tight cluster near the best.
        top = scored[: max(5, min(20, len(scored) // 4))]

        # Occasional divergence to escape the plateau (every ~5 steps when stuck).
        if self.stagnation >= 4 and self.step % 5 == 0:
            # Diverge from an under-used top-tier program, no context.
            underused = sorted(top, key=lambda p: self.parent_usage.get(p.id, 0))
            parent = underused[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {"": []}

        # Parent: usage-penalized weighted pick from the top tier (varied
        # near-best parents, not always the single best).
        weights = []
        for p in top:
            usage = self.parent_usage.get(p.id, 0)
            w = max(self._score(p) - 0.5, 0.01) / (1.0 + 2.0 * usage)
            weights.append(w)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Label: mostly plain mutation; REFINE on the best occasionally when
        # stagnant, since breakthroughs historically came from refining near-best.
        if self.stagnation >= 3 and self.step % 3 == 0:
            best = scored[0]
            self.parent_usage[best.id] = self.parent_usage.get(best.id, 0) + 1
            return {self.REFINE_LABEL: best}, {"": []}

        # Context: 2 other top-tier programs + 2 mid-tier contrasts (mid-tier
        # parents with diverse context historically produced 0.70+ children).
        others_top = [p for p in top if p.id != parent.id]
        self.random_state.shuffle(others_top)
        mid = scored[len(scored) // 3: 2 * len(scored) // 3]
        self.random_state.shuffle(mid)
        ctx = others_top[:2] + [p for p in mid if p.id != parent.id][: max(0, n_ctx - 2)]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END