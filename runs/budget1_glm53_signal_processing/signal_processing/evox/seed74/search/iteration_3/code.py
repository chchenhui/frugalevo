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
    """Search strategy tuned for a plateaued population near its ceiling:
    mostly REFINE the best program (least-used among top scorers), with
    occasional DIVERGE from a mid-tier parent plus diverse context to spark
    new directions. Context mixes top scorers with contrasting mid/low ones."""

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
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        ranked = sorted(candidates, key=self._score, reverse=True)
        valid = [p for p in ranked if self._score(p) > 0]
        top = valid[:8]
        mid = valid[8:24]
        low = valid[24:]

        # Default: refine a top scorer, rotating to avoid overuse
        pool = top if top else valid
        pool_sorted = sorted(pool, key=lambda p: (self.parent_usage.get(p.id, 0), -self._score(p)))
        parent = pool_sorted[0]
        label = self.REFINE_LABEL

        # Every 4th call, diverge from a mid-tier parent (fresh genetic material)
        if self.sample_calls % 4 == 0 and mid:
            parent = self.random_state.choice(mid)
            label = self.DIVERGE_LABEL

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: contrasting examples — a couple of top scorers plus
        # diverse mid/low programs, excluding the parent
        others = [p for p in valid if p.id != parent.id]
        top_ctx = others[: max(1, n_ctx // 2)]
        rest = [p for p in mid + low if p.id not in {q.id for q in top_ctx}]
        self.random_state.shuffle(rest)
        examples = top_ctx + rest[: n_ctx - len(top_ctx)]

        parent_dict = {label: parent}
        context_programs_dict = {"": examples}

        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END