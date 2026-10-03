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
    """Adaptive search database.

    Principles:
    1. Exploit the top tier (with usage caps to avoid overuse) but mix in
       diverse mid/low programs as context to give the LLM contrasting ideas.
    2. On stagnation, alternate REFINE on the best program and DIVERGE on a
       fresh parent to escape plateaus.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1e18
        self.stagnation: int = 0
        self.sample_calls: int = 0

    @staticmethod
    def _score(program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return -1e18

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id in self.programs or program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        score = self._score(program)
        if score > self.best_score + 0.01:
            self.best_score = score
            self.stagnation = 0
        else:
            self.stagnation += 1

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

        # Rank programs by score.
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_tier = ranked[: max(1, len(ranked) // 4)]
        best = ranked[0]

        # Stagnation handling: alternate refine-best and diverge.
        if self.stagnation >= 3:
            if self.stagnation % 2 == 1:
                return {self.REFINE_LABEL: best}, {}
            # Diverge from a lightly-used mid/low tier program.
            pool = [p for p in ranked[len(ranked) // 2:]]
            if not pool:
                pool = ranked
            pool = sorted(pool, key=lambda p: self.parent_usage.get(p.id, 0))
            parent = self.random_state.choice(pool[: max(1, len(pool) // 2)])
            return {self.DIVERGE_LABEL: parent}, {}

        # Default: exploit top tier with usage cap for rotation.
        cap = 3
        fresh_top = [p for p in top_tier if self.parent_usage.get(p.id, 0) < cap]
        parent_pool = fresh_top if fresh_top else top_tier
        parent = self.random_state.choice(parent_pool)

        # Context: best program (if not parent) + diverse programs across score bands.
        ctx = []
        if best.id != parent.id:
            ctx.append(best)
        rest = [p for p in ranked if p.id != parent.id and p.id != best.id]
        if rest:
            self.random_state.shuffle(rest)
            # pick spread across bands
            step = max(1, len(rest) // max(1, n_ctx))
            ctx.extend(rest[::step][: n_ctx - len(ctx)])
        ctx = ctx[:n_ctx]

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END