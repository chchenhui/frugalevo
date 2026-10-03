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
    """Late-stage search: alternate REFINE on the best program with DIVERGE
    from diverse mid-tier parents. Context pairs the best with contrasting
    approaches so the LLM can combine strengths."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -float("inf")
        self.stagnation = 0
        self.sample_calls = 0
        self.refine_streak = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        # meaningful improvement: >1% relative or >0.01 absolute
        if s > self.best_score and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)):
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
        candidates = [p for p in self.programs.values() if self._score(p) > 0]
        if not candidates:
            raise ValueError("No candidates available for sampling")
        n_ctx = num_context_programs or 4
        self.sample_calls += 1

        ranked = sorted(candidates, key=self._score, reverse=True)
        best = ranked[0]

        # Diverse mid-tier pool (exclude near-duplicates of best by score)
        mid_tier = [p for p in ranked[3:] if self._score(p) < 0.76]
        # Stagnating (which it is at start): alternate diverge/refine
        if self.stagnation > 0 and self.sample_calls % 2 == 0 and mid_tier:
            # DIVERGE from a random mid-tier parent (breakthroughs came from these)
            parent = self.random_state.choice(mid_tier[: max(5, len(mid_tier) // 2)])
            label = self.DIVERGE_LABEL
            self.refine_streak = 0
        else:
            # REFINE on best, but cap consecutive refines to avoid overuse
            parent = best
            label = self.REFINE_LABEL if self.stagnation >= 2 else ""
            self.refine_streak += 1

        parent_dict = {label: parent}

        # Context: best program + diverse others (excluding parent)
        others = [p for p in ranked if p.id != parent.id]
        top_ctx = [others[0]]
        rest = others[1:]
        self.random_state.shuffle(rest)
        examples = top_ctx + rest[: max(0, n_ctx - 1)]

        return parent_dict, {"": examples}


# EVOLVE-BLOCK-END