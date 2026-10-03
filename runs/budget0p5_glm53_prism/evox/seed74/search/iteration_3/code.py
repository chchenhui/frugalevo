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


def _score(p: Program) -> float:
    v = p.metrics.get("combined_score") if p.metrics else None
    if isinstance(v, (int, float)):
        return float(v)
    return -1e9


class EvolvedProgramDatabase(ProgramDatabase):
    """Plateau-breaking strategy.

    The population is heavily converged at the best score. We rotate parents
    among the top-scoring programs (least-used first), mostly REFINE the best,
    and periodically DIVERGE from the best when nothing improves. Context is
    drawn from distinct score levels (including a lower-tier program) to give
    the LLM contrasting perspectives.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.stall_count = 0
        self.best_score = -1e18
        self.diverge_counter = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None and hasattr(self, "last_iteration"):
            self.last_iteration = max(self.last_iteration, iteration)

        s = _score(program)
        if s > self.best_score + 0.01:  # meaningful improvement
            self.best_score = s
            self.stall_count = 0
        else:
            self.stall_count += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4

        # Top tier: programs at/near the best score.
        best = max(candidates, key=_score)
        top = [p for p in candidates if _score(p) >= _score(best) - 0.2]

        # Rotate parent among top tier, preferring least-used.
        top_sorted = sorted(top, key=lambda p: (self.usage_counts.get(p.id, 0), self.random_state.random()))
        parent = top_sorted[0]
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Label: mostly REFINE on the best when stalled; occasionally DIVERGE.
        label = ""
        if self.stall_count >= 3:
            self.diverge_counter += 1
            if self.diverge_counter % 4 == 0:
                label = self.DIVERGE_LABEL
                parent = best  # diverge from the best program itself
                self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1
            else:
                label = self.REFINE_LABEL
        parent_dict = {label: parent}

        # Context: distinct score levels, including a mid/low-tier program for contrast.
        by_score = sorted(candidates, key=_score, reverse=True)
        distinct: List[EvolvedProgram] = []
        seen_scores = set()
        for p in by_score:
            s = round(_score(p), 4)
            if s not in seen_scores and p.id != parent.id:
                distinct.append(p)
                seen_scores.add(s)
            if len(distinct) >= n_ctx - 1:
                break
        # Add one lower-tier program for a contrasting perspective.
        low_tier = [p for p in candidates if _score(p) < _score(best) - 1.0 and p.id != parent.id]
        if low_tier and len(distinct) < n_ctx:
            distinct.append(self.random_state.choice(low_tier))
        while len(distinct) < n_ctx and len(candidates) > 1:
            extra = self.random_state.choice([p for p in candidates if p.id != parent.id and p not in distinct])
            distinct.append(extra)

        return parent_dict, {"": distinct[:n_ctx]}


# EVOLVE-BLOCK-END