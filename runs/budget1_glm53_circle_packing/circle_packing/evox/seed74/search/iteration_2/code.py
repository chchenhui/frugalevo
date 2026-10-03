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
    """Exploit-heavy search: mutate top-tier programs, provide diverse context,
    and use REFINE/DIVERGE labels when progress stalls."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.stall_count = 0
        self.parent_usage: Dict[str, int] = {}
        self.sample_count = 0

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if self.best_score is None:
            self.best_score = s
        else:
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

        self.sample_count += 1
        n_ctx = num_context_programs or 4

        # rank programs by score (descending)
        ranked = sorted(candidates, key=self._score, reverse=True)
        top_tier = ranked[: max(2, len(ranked) // 3)]  # top third of population

        # ---- stagnation handling ----
        if self.stall_count >= 3 and self.sample_count % 4 == 0:
            # deeply stuck: diverge from best program
            parent = ranked[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.DIVERGE_LABEL: parent}, {}
        if self.stall_count >= 2 and self.sample_count % 3 == 0:
            # refine the best program
            parent = ranked[0]
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            return {self.REFINE_LABEL: parent}, {}

        # ---- default: exploit top tier with usage-based diversity ----
        # prefer least-used top-tier parents, with randomness
        weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p in top_tier]
        parent = self.random_state.choices(top_tier, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # ---- context: best program + diverse score bands ----
        ctx: List[EvolvedProgram] = []
        best = ranked[0]
        if best.id != parent.id:
            ctx.append(best)
        # add a mid-tier and a low-tier program for contrast
        mid = ranked[len(ranked) // 2] if len(ranked) > 2 else None
        low = ranked[-1] if len(ranked) > 1 else None
        for c in (mid, low):
            if c is not None and c.id != parent.id and all(x.id != c.id for x in ctx):
                ctx.append(c)
        # fill remaining slots randomly from the rest
        rest = [p for p in candidates if all(x.id != p.id for x in ctx) and p.id != parent.id]
        self.random_state.shuffle(rest)
        ctx.extend(rest[: n_ctx - len(ctx)])

        return {"": parent}, {"": ctx[:n_ctx]}


# EVOLVE-BLOCK-END