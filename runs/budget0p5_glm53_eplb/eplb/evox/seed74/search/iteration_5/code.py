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
    """Simple adaptive strategy: rotate diverse parents (weighted toward top
    tier but including mid-tier), always give the best programs as context,
    and only occasionally use DIVERGE when deeply stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.iters_since_improvement: int = 0
        self.sample_calls: int = 0

    @staticmethod
    def _score(p: Program) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        # meaningful improvement: >1% relative or >0.01 absolute
        if s > self.best_score + 0.01 or (self.best_score > 0 and s > self.best_score * 1.01):
            self.best_score = s
            self.iters_since_improvement = 0
        else:
            self.iters_since_improvement += 1
            self.best_score = max(self.best_score, s)

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

        self.sample_calls += 1
        n_ctx = num_context_programs or 4

        # Rank by score
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_half = ranked[: max(1, len(ranked) // 2)]

        # Prefer least-used parents to enforce rotation (avoid reusing one top parent)
        def use_key(p):
            return (self.parent_use_count.get(p.id, 0), self.random_state.random())

        # 70%: rotate among top half (exploit); 30%: mid/low tier (explore)
        if self.random_state.random() < 0.7:
            pool = top_half
        else:
            pool = ranked[len(ranked) // 2:] or top_half
        parent = min(pool, key=use_key)
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        # Context: best programs (excluding parent) + one diverse mid-tier program
        context = [p for p in ranked if p.id != parent.id][: max(1, n_ctx - 1)]
        mid = ranked[len(ranked) // 2: len(ranked) // 2 + 1]
        if mid and mid[0].id != parent.id:
            context.append(mid[0])
        context = context[:n_ctx]

        # Occasional DIVERGE when deeply stuck, but never on the best parent
        label = ""
        if (
            self.iters_since_improvement >= 6
            and self.sample_calls % 3 == 0
            and self._score(parent) < self.best_score - 0.001
        ):
            label = self.DIVERGE_LABEL
            context = []  # targeted divergence

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END