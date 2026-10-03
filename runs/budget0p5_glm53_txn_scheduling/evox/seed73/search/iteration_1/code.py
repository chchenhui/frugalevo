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
    """Exploit-first search with stagnation-triggered divergence.

    - Parents are drawn mostly from the top tier of scores (exploitation),
      with occasional mid-tier picks for exploration.
    - Context mixes the global best with diverse mid/high scorers.
    - When no meaningful improvement happens for several adds, alternate
      between REFINE (on the best program) and DIVERGE (on a good, less-used
      program) to break out of local plateaus.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        # Only count meaningful improvements (>1% relative or >0.01 absolute)
        if score > self.best_score and score - self.best_score > max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = score
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

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

        # Sort by score descending.
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_tier = ranked[: max(3, len(ranked) // 4)]

        label = ""
        if self.stagnation_count >= 3:
            # Alternate between refining the best and diverging from a good,
            # under-used program to escape the plateau.
            if self.stagnation_count % 2 == 0:
                parent = ranked[0]
                label = self.REFINE_LABEL
                return {label: parent}, {}
            # Diverge from a top-tier program that hasn't been overused.
            pool = [p for p in top_tier if self.parent_usage.get(p.id, 0) < 3] or top_tier
            parent = self.random_state.choice(pool)
            return {self.DIVERGE_LABEL: parent}, {}

        # Normal sampling: exploit top tier 70% of the time, else explore mid/high tier.
        if self.random_state.random() < 0.7:
            parent = self.random_state.choice(top_tier)
        else:
            parent = self.random_state.choice(ranked[: max(5, len(ranked) // 2)])

        # Context: global best + diverse others (avoid parent, avoid repeats).
        context: List[EvolvedProgram] = []
        seen_ids = {parent.id}
        best = ranked[0]
        if best.id not in seen_ids:
            context.append(best)
            seen_ids.add(best.id)
        rest = [p for p in ranked if p.id not in seen_ids]
        self.random_state.shuffle(rest)
        for p in rest:
            if len(context) >= n_ctx:
                break
            context.append(p)

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END