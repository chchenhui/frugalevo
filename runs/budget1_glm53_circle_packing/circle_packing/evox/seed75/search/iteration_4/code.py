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
    """Plateau-focused search: refine the best with diverse context,
    occasionally diverge on fresh mid-tier parents."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.usage_counts: Dict[str, int] = {}
        self.best_score: float = 0.0
        self.stagnation: int = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score") if program.metrics else None
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        # meaningful improvement threshold
        if score > self.best_score + 0.01:
            self.best_score = score
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

        # Rank by score, prefer least-used parents
        scored = sorted(candidates, key=self._score, reverse=True)

        # With stagnation deep, mostly refine best-tier; occasionally diverge on a fresh parent
        use_diverge = self.stagnation > 3 and self.random_state.random() < 0.3

        if use_diverge:
            # fresh, less-used, mid-tier parent (avoid overused & broken ones)
            pool = [p for p in scored if self._score(p) > 0.5 and self._score(p) < self.best_score]
            pool.sort(key=lambda p: self.usage_counts.get(p.id, 0))
            parent = pool[0] if pool else scored[0]
            self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}

        # Default: refine a top-tier parent, rotating among the best few to avoid overuse
        top = scored[:max(3, len(scored) // 4)]
        top.sort(key=lambda p: self.usage_counts.get(p.id, 0))
        parent = top[0]
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Context: diverse mix — one other top program, plus varied lower scorers (failures included)
        others = [p for p in candidates if p.id != parent.id]
        others.sort(key=self._score, reverse=True)
        ctx = []
        if others:
            ctx.append(others[0])  # another strong program
        low = [p for p in others if self._score(p) < 0.9]
        self.random_state.shuffle(low)
        ctx.extend(low[: n_ctx - len(ctx)])
        if len(ctx) < n_ctx:
            mids = [p for p in others if p not in ctx]
            self.random_state.shuffle(mids)
            ctx.extend(mids[: n_ctx - len(ctx)])

        label = self.REFINE_LABEL if self.stagnation > 5 else ""
        return {label: parent}, {"": ctx}
# EVOLVE-BLOCK-START