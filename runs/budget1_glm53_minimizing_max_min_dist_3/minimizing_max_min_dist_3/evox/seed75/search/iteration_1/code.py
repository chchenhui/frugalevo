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
    """Best-first search with periodic divergence.

    The population is saturated at score 1.0, so the only way to improve is to
    push past the plateau. Strategy:
      - Mostly mutate the single best-scoring program (exploit).
      - Periodically issue a DIVERGE call on the best program to break the
        plateau, and a REFINE call after fresh improvements appear.
      - Context is drawn from top-scoring *distinct* programs for diversity.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -1.0
        self.since_improvement = 0
        self.sample_calls = 0
        self.last_diverge_call = -100

    # ---------- helpers ----------

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def _sorted_programs(self) -> List[EvolvedProgram]:
        return sorted(self.programs.values(), key=self._score, reverse=True)

    # ---------- required API ----------

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        # Count meaningful improvement (>1% relative or 0.01 absolute).
        if s > self.best_score + max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.since_improvement = 0
        else:
            self.since_improvement += 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        self.sample_calls += 1
        ranked = self._sorted_programs()
        n_ctx = num_context_programs or 4

        # Parent: mostly the best program; occasionally the second-best or a
        # random top-5 program to avoid overusing a single parent.
        r = self.random_state.random()
        if r < 0.6 or len(ranked) == 1:
            parent = ranked[0]
        elif r < 0.85 and len(ranked) > 1:
            parent = ranked[1]
        else:
            parent = self.random_state.choice(ranked[:min(5, len(ranked))])

        # Context: distinct top scorers (excluding parent), shuffled a bit.
        pool = [p for p in ranked if p.id != parent.id]
        # Prefer diversity: take top scorers but shuffle the slice so the
        # same context set isn't always returned.
        top = pool[:max(n_ctx * 2, 4)]
        self.random_state.shuffle(top)
        context = top[:n_ctx]

        # Decide label based on stagnation state.
        label = ""
        if self.since_improvement >= 4 and (self.sample_calls - self.last_diverge_call) >= 5:
            # Stuck: push for a fundamentally different construction.
            label = self.DIVERGE_LABEL
            self.last_diverge_call = self.sample_calls
            parent = ranked[0]
            context = []  # targeted divergence
        elif self.since_improvement == 0 and self.best_score > 0:
            # Fresh improvement appeared: refine it immediately.
            label = self.REFINE_LABEL
            parent = ranked[0]
            context = []

        parent_dict = {label: parent}
        context_programs_dict = {"": context}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END