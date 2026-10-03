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
    """Simple adaptive sampler: exploit top-tier parents with rotation,
    REFINE the best when stagnant, occasionally DIVERGE from mid-tier parents.
    Context mixes top programs with diverse lower-tier ones."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation = 0
        self.sample_calls = 0
        self.parent_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        return float(v) if isinstance(v, (int, float)) else -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
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
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s > 0]
        if not scored:
            raise ValueError("No candidates available for sampling")
        scored.sort(key=lambda x: x[1], reverse=True)

        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        top = [p for p, _ in scored[:10]]
        mid = [p for p, _ in scored[10:35]]
        low = [p for p, _ in scored[35:]]

        parent = None
        label = ""

        if self.stagnation >= 3:
            # Stalled: alternate REFINE on best vs DIVERGE from a mid/low parent
            if self.sample_calls % 3 == 0 and (mid or low):
                pool = mid if self.sample_calls % 2 == 0 else low
                parent = self.random_state.choice(pool)
                label = self.DIVERGE_LABEL
            else:
                parent = scored[0][0]
                label = self.REFINE_LABEL
        else:
            # Exploit top tier, rotate least-used parents to avoid overuse
            parent = min(top, key=lambda p: (self.parent_usage.get(p.id, 0), -self._score(p)))

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: top performers + diverse ones, excluding parent
        others = [p for p, _ in scored if p.id != parent.id]
        top_ctx = others[: n_ctx // 2]
        rest = [p for p in others if p not in top_ctx]
        self.random_state.shuffle(rest)
        examples = top_ctx + rest[: max(0, n_ctx - len(top_ctx))]

        parent_dict = {label: parent} if label else {"": parent}
        return parent_dict, {"": examples}


# EVOLVE-BLOCK-END