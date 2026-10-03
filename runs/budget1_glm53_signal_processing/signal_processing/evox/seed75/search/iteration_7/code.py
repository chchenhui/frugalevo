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
    """Simple adaptive search: usage-penalized parent selection from top tier,
    diverse-tier context, and occasional REFINE on the best program when stuck."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.stagnation: int = 0
        self.refine_used_on: set = set()

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        return float(v) if isinstance(v, (int, float)) else -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score + 0.0001:
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

        scored = sorted(candidates, key=self._score, reverse=True)
        best = scored[0]
        n_ctx = num_context_programs or 4

        # Occasional REFINE on the best program when deeply stuck
        if self.stagnation >= 5 and best.id not in self.refine_used_on:
            self.refine_used_on.add(best.id)
            self.stagnation = 0
            return {self.REFINE_LABEL: best}, {}

        # Parent: weighted pick from top third, penalized by past usage
        top = scored[: max(3, len(scored) // 3)]
        weights = [1.0 / (1 + 3 * self.parent_usage.get(p.id, 0)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: best program + diverse tiers (mid, low) excluding parent
        rest = [p for p in scored if p.id != parent.id]
        ctx: List[EvolvedProgram] = []
        if best.id != parent.id:
            ctx.append(best)
        for lo, hi in ((0.25, 0.5), (0.5, 0.75)):
            band = rest[int(len(rest) * lo): max(int(len(rest) * hi), int(len(rest) * lo) + 1)]
            if band:
                ctx.append(self.random_state.choice(band))
        pool = [p for p in rest if p not in ctx]
        self.random_state.shuffle(pool)
        ctx.extend(pool[: n_ctx - len(ctx)])
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END