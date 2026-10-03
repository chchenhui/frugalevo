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
    """Plateau-aware search: when stuck, diverge from mid-tier parents (which
    historically produced breakthroughs) instead of re-refining the same top
    parents. Otherwise refine the freshest top-tier child. Context mixes top
    performers with contrasting (low/mid-tier) programs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation = 0
        self.sample_calls = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.01:
            self.best_score = s
            self.stagnation = 0
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
        candidates = [p for p in self.programs.values() if self._score(p) > 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        ranked = sorted(candidates, key=self._score, reverse=True)
        top_tier = ranked[:8]
        mid_tier = [p for p in ranked[8:30] if self._score(p) >= 0.55]

        # Least-used parents first to avoid overuse
        def usage(p):
            return self.parent_usage.get(p.id, 0)

        parent = None
        label = ""

        if self.stagnation >= 3 and self.sample_calls % 2 == 0 and mid_tier:
            # Stalled: diverge from a fresh mid-tier parent (breakthrough zone)
            fresh = sorted(mid_tier, key=usage)[: max(3, len(mid_tier) // 3)]
            parent = self.random_state.choice(fresh)
            label = self.DIVERGE_LABEL
        elif self.stagnation >= 3:
            # Stalled: refine the best, preferring least-used top parents
            fresh_top = sorted(top_tier, key=lambda p: (usage(p), -self._score(p)))
            parent = fresh_top[0]
            label = self.REFINE_LABEL
        else:
            # Normal: refine/exploit least-used top-tier programs
            fresh_top = sorted(top_tier, key=lambda p: (usage(p), -self._score(p)))
            parent = fresh_top[0]

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: top performers + contrasting mid/low programs, excluding parent
        others = [p for p in ranked if p.id != parent.id]
        top_ctx = sorted(others[:8], key=usage)[:2]
        rest = [p for p in others if p not in top_ctx]
        self.random_state.shuffle(rest)
        examples = top_ctx + rest[: n_ctx - len(top_ctx)]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END