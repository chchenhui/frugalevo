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
    """Converged-population search strategy.

    The population is heavily clustered near the best score with no recent
    improvement. Strategy: exploit the top cluster (rotating parents to avoid
    over-use), occasionally pair a mid-tier parent with top-tier context
    (which historically produced gains), and fall back to REFINE on the best
    when stalled.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = float("-inf")
        self.stall_count: int = 0
        self.sample_count: int = 0
        self.last_refine_sample: int = -100

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def _sorted_programs(self) -> List[EvolvedProgram]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored]

    def _least_used(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        return min(pool, key=lambda p: self.parent_usage.get(p.id, 0))

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        score = self._score(program)
        if score is not None:
            if score > self.best_score + 0.005:
                self.best_score = score
                self.stall_count = 0
            else:
                if self.best_score == float("-inf"):
                    self.best_score = score
                self.stall_count += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        self.sample_count += 1
        n_ctx = num_context_programs or 4

        ranked = self._sorted_programs()
        if not ranked:
            # No numeric scores: fall back to uniform random.
            parent = self.random_state.choice(list(self.programs.values()))
            return {"": parent}, {"": []}

        top_pool = ranked[:8]                      # exploit cluster near best
        mid_pool = ranked[4:16] or ranked[1:]      # mid-tier for exploration

        # --- Parent selection -------------------------------------------
        label = ""
        parent = None

        if self.stall_count >= 3 and self.sample_count - self.last_refine_sample >= 2:
            # Stalled: REFINE on the least-used top program (avoids hammering
            # the single best, which previously led to over-use and no gain).
            refine_pool = top_pool[:3]
            parent = self._least_used(refine_pool)
            label = self.REFINE_LABEL
            self.last_refine_sample = self.sample_count
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {label: parent}, {"": []}

        if mid_pool and self.random_state.random() < 0.25:
            # Mid-tier parent + top-tier context (historically productive combo).
            parent = self._least_used(mid_pool)
        else:
            # Rotate within the top cluster, preferring least-used parents.
            parent = self._least_used(top_pool)

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # --- Context selection -------------------------------------------
        # Top-scoring programs with distinct scores, excluding the parent:
        # complementary high-quality perspectives without redundancy.
        seen_scores = set()
        context: List[EvolvedProgram] = []
        for p in ranked:
            if p.id == parent.id:
                continue
            s = self._score(p)
            if s in seen_scores:
                continue
            seen_scores.add(s)
            context.append(p)
            if len(context) >= n_ctx:
                break

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END