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
    """Stagnation-aware search: rotate diverse parents, use DIVERGE with
    diverse context when progress stalls (the only mode that produced gains
    in the observed history)."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -1.0
        self.stall_count = 0
        self.parent_usage: Dict[str, int] = {}
        self.sample_calls = 0

    @staticmethod
    def _score(p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
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
        if s > self.best_score + 0.01:  # meaningful improvement
            self.best_score = s
            self.stall_count = 0
        else:
            self.stall_count += 1

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

        # Score-ordered candidates, exclude broken programs.
        valid = [p for p in candidates if self._score(p) > 0.0]
        valid.sort(key=self._score, reverse=True)
        pool = valid if valid else candidates

        # Rotate parents among the top half (diversity, avoid overuse),
        # occasionally pick a mid-tier program for exploration.
        top = pool[: max(1, len(pool) // 2)]
        if self.random_state.random() < 0.25 and len(pool) > 4:
            parent = self.random_state.choice(pool[len(top):len(top) * 2] or pool)
        else:
            parent = self.random_state.choice(top)

        # Avoid overusing any single parent.
        if self.parent_usage.get(parent.id, 0) >= 2:
            others = [p for p in pool if self.parent_usage.get(p.id, 0) < 2]
            if others:
                parent = self.random_state.choice(others)
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Diverse context: best program + random picks across score bands.
        ctx: List[EvolvedProgram] = []
        best = pool[0]
        if best.id != parent.id:
            ctx.append(best)
        rest = [p for p in pool if p.id != parent.id and p.id != best.id]
        self.random_state.shuffle(rest)
        # Sample across bands for diversity.
        if len(rest) > n_ctx:
            step = max(1, len(rest) // (n_ctx - len(ctx)))
            rest = [rest[i] for i in range(0, len(rest), step)]
        ctx.extend(rest)
        ctx = ctx[:n_ctx]

        # Stagnating: history shows DIVERGE with diverse context is the only
        # mode that produced gains; REFINE on best repeatedly failed.
        if self.stall_count >= 3:
            parent_dict = {self.DIVERGE_LABEL: parent}
        else:
            parent_dict = {"": parent}

        return parent_dict, {"": ctx}


# EVOLVE-BLOCK-END