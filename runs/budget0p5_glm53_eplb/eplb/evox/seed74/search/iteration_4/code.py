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
    """Diversity-rotating search database.

    Population is tightly converged near the best score; repeatedly refining the
    same top parent has failed. Strategy: rotate parents across score tiers
    (never reusing one recently), pair them with diverse context, and use
    DIVERGE_LABEL as the default mode (divergence on mid-tier parents produced
    the best children historically), with occasional REFINE on fresh top programs.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.recent_parent_ids: List[str] = []
        self.sample_count = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        return float(v) if isinstance(v, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        cands = list(self.programs.values())
        if not cands:
            raise ValueError("No candidates available for sampling")

        # Fresh (not recently used) candidates, sorted by score.
        fresh = [p for p in cands if p.id not in self.recent_parent_ids]
        if not fresh:
            fresh = cands
            self.recent_parent_ids = []

        fresh.sort(key=self._score, reverse=True)
        k = len(fresh)
        # Rotate across tiers: top, mid, low, top-mid, ...
        self.sample_count += 1
        slot = self.sample_count % 4
        if slot == 0:
            parent = self.random_state.choice(fresh[: max(1, k // 4)])
            label = self.REFINE_LABEL
        elif slot == 1:
            parent = self.random_state.choice(fresh[k // 4: max(k // 4 + 1, k // 2)])
            label = self.DIVERGE_LABEL
        elif slot == 2:
            parent = self.random_state.choice(fresh[max(k // 2, k - 1):])
            label = self.DIVERGE_LABEL
        else:
            parent = self.random_state.choice(fresh[: max(1, k // 2)])
            label = ""

        # Diverse context: top programs plus mid/low programs, excluding parent.
        pool = [p for p in cands if p.id != parent.id]
        pool.sort(key=self._score, reverse=True)
        top = pool[:2]
        rest = pool[2:]
        self.random_state.shuffle(rest)
        context = top + rest
        context = context[:n_ctx]

        self.recent_parent_ids.append(parent.id)
        self.recent_parent_ids = self.recent_parent_ids[-8:]

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END