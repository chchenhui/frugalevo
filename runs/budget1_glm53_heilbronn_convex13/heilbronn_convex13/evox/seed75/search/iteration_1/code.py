# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Exploit-first search with adaptive refine/diverge.

    Principle: the population is top-heavy and stuck — mutate from the best
    programs (not low scorers), give diverse high-scoring context, and only
    diverge when no meaningful improvement has happened for a while.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -1.0
        self.stall_count = 0
        self.parent_use_count: Dict[str, int] = {}

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        s = self._score(program)
        if s > self.best_score:
            # only count meaningful improvements
            if s - self.best_score > max(0.01, 0.01 * abs(self.best_score)) or self.best_score < 0:
                self.stall_count = 0
            self.best_score = s
        else:
            self.stall_count += 1

        self.parent_use_count[program.id] = 0

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) >= 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # rank by score, focus on top tier
        candidates.sort(key=self._score, reverse=True)
        top = candidates[: max(3, len(candidates) // 3)]

        # prefer least-used parents in the top tier to avoid overuse
        top_sorted = sorted(top, key=lambda p: self.parent_use_count.get(p.id, 0))
        pool = top_sorted[: max(2, len(top_sorted) // 2)]
        parent = self.random_state.choice(pool)
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # context: diverse top programs (excluding parent), plus one mid-tier
        ctx_pool = [p for p in candidates if p.id != parent.id]
        top_ctx = [p for p in ctx_pool if p in top]
        self.random_state.shuffle(top_ctx)
        n = num_context_programs or 4
        examples = top_ctx[: max(1, n - 1)]
        rest = [p for p in ctx_pool if p not in top]
        if rest:
            examples.append(self.random_state.choice(rest))
        examples = examples[:n]

        label = ""
        if self.stall_count >= 5 and self.random_state.random() < 0.5:
            # stuck: alternate between refining best and diverging from it
            best = candidates[0]
            if self.random_state.random() < 0.6:
                parent, label = best, self.REFINE_LABEL
            else:
                parent, label = best, self.DIVERGE_LABEL
            self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
            if self.random_state.random() < 0.5:
                examples = []
            self.stall_count = 0

        return {"": parent}, {"": examples}


# EVOLVE-BLOCK-END