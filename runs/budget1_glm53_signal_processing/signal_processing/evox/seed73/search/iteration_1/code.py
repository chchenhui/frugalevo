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
    """Adaptive score-weighted search with stagnation-driven divergence.

    Parents are drawn with a preference for high-scoring but under-used
    programs; context mixes top-tier with diverse mid/low-tier programs to give
    contrasting examples. When progress stalls, we alternate between refining
    the current best and diverging from it.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.diverge_counter: int = 0

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

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            meaningful = self._is_meaningful(s, self.best_score) if self.best_score > -math.inf else True
            self.best_score = max(self.best_score, s)
            if meaningful:
                self.stagnation = 0
            else:
                self.stagnation += 1
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
        top = scored[: max(1, len(scored) // 3)]

        # Stagnation handling: alternate refine-best / diverge-best.
        if self.stagnation >= 3:
            self.diverge_counter += 1
            best = scored[0]
            if self.diverge_counter % 2 == 1:
                # Refine the current best approach.
                return {self.REFINE_LABEL: best}, {"": []}
            # Diverge from the best to seek a new direction.
            return {self.DIVERGE_LABEL: best}, {"": []}

        # Normal mode: score-weighted parent pick among top tier,
        # penalizing over-used parents to avoid determinism.
        weights = []
        for p in top:
            usage = self.parent_usage.get(p.id, 0)
            w = max(self._score(p), 0.01) * (1.0 / (1.0 + usage))
            weights.append(w)
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: mix of top-tier and diverse mid/low-tier programs.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if p in top]
        rest = [p for p in others if p not in top]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(rest)
        ctx = top_others[: max(1, n_ctx // 2)] + rest[: n_ctx - min(len(top_others), max(1, n_ctx // 2))]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END