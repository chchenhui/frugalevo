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
    """Search strategy: unlabeled mid-tier parents with diverse context.

    Evidence from the population: all past gains came from unlabeled parents
    paired with diverse context (including low/failure programs); labels on
    top programs consistently regressed. So we:
      1. Pick parents from the upper-mid tier (0.68-0.70), penalizing
         over-used parents to avoid determinism.
      2. Provide diverse context: best scorer + median + a low scorer + a
         random program, so the LLM sees contrasting quality ranges.
      3. Occasionally (when stagnant) apply REFINE_LABEL to the single best
         program with no context, and rarely DIVERGE on a mid-tier parent.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation_counter: int = 0

    @staticmethod
    def _score(p: Program) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        s = self._score(program)
        # Track stagnation: meaningful improvement only if > 0.01 absolute
        if s > self.best_score + 0.01:
            self.best_score = max(self.best_score, s)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        all_progs = sorted(self.programs.values(), key=self._score, reverse=True)
        best = all_progs[0]
        median = all_progs[len(all_progs) // 2]
        worst = all_progs[-1]

        # --- Stagnation response: occasionally refine the best program ---
        if self.stagnation_counter >= 4 and self._score(best) > 0:
            self.stagnation_counter = 0
            self.parent_usage[best.id] = self.parent_usage.get(best.id, 0) + 1
            return {self.REFINE_LABEL: best}, {self.REFINE_LABEL: []}

        # --- Parent selection: upper-mid tier with usage penalty ---
        scores = [self._score(p) for p in all_progs]
        if scores:
            top = scores[0]
        else:
            top = 0.0
        # Upper-mid tier: within ~3% of best but not the very best programs
        tier = [p for p in all_progs if 0.0 < top * 0.97 <= self._score(p) < top]
        if not tier:
            tier = [p for p in all_progs if 0.0 < self._score(p) < top]
        if not tier:
            tier = [best]

        # Weight by inverse usage to avoid overusing any single parent
        weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p in tier]
        if sum(weights) <= 0:
            parent = self.random_state.choice(tier)
        else:
            parent = self.random_state.choices(tier, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # --- Context: best + median + worst + one random (diverse quality ranges) ---
        context: List[EvolvedProgram] = []
        seen = {parent.id}
        for cand in (best, median, worst):
            if cand.id not in seen:
                context.append(cand)
                seen.add(cand.id)
        others = [p for p in all_progs if p.id not in seen]
        if others and len(context) < n_ctx:
            context.append(self.random_state.choice(others))
        context = context[:n_ctx]

        return {"": parent}, {"": context}


# EVOLVE-BLOCK-END