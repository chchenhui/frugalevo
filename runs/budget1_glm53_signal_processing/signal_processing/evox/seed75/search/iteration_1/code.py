# EVOLVE-BLOCK-START
import logging
import math
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
    """Adaptive search strategy.

    Principles:
    - Mostly exploit top programs, but track per-program usage so no parent
      is overused; occasionally sample mid-tier programs (they produced the
      best children in this population).
    - On stagnation, alternate DIVERGE (new direction from a good parent) and
      REFINE (deep polish of the best program).
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -math.inf
        self.stagnation_count = 0
        self.parent_usage: Dict[str, int] = {}
        self.sample_calls = 0

    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            # meaningful improvement check (>0.01 absolute)
            if self.best_score > -math.inf and (s - self.best_score) > 0.01:
                self.stagnation_count = 0
            elif self.best_score == -math.inf:
                self.stagnation_count = 0
            self.best_score = max(self.best_score, s)
        else:
            self.stagnation_count += 1

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
        rng = self.random_state

        # sort by score descending
        ranked = sorted(candidates, key=self._score, reverse=True)
        top = ranked[: max(2, len(ranked) // 3)]
        label = ""

        if self.stagnation_count >= 3:
            # Stuck: alternate divergence and refinement
            if self.sample_calls % 2 == 0:
                parent = rng.choice(top[:3])
                label = self.DIVERGE_LABEL
                return {label: parent}, {}
            else:
                parent = ranked[0]
                label = self.REFINE_LABEL
                return {label: parent}, {}

        # Normal mode: weighted pick among top tier, penalizing overuse
        def weight(p):
            u = self.parent_usage.get(p.id, 0)
            return 1.0 / (1.0 + u)

        # occasionally (20%) pick a mid-tier program for exploration
        if rng.random() < 0.2 and len(ranked) > 4:
            mid = ranked[len(ranked) // 3: 2 * len(ranked) // 3]
            parent = rng.choice(mid)
        else:
            total = sum(weight(p) for p in top)
            r = rng.random() * total
            acc = 0.0
            parent = top[0]
            for p in top:
                acc += weight(p)
                if r <= acc:
                    parent = p
                    break

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # Context: diverse mix - best program, a distinct-approach program
        # (different lineage), and random others; never the parent.
        others = [p for p in candidates if p.id != parent.id]
        ctx: List[EvolvedProgram] = []
        best_prog = ranked[0]
        if best_prog.id != parent.id:
            ctx.append(best_prog)
        # a program from a different parent lineage for diversity
        lineages = [p for p in others if p.parent_id != parent.parent_id]
        if lineages:
            ctx.append(rng.choice(lineages))
        rest = [p for p in others if all(c.id != p.id for c in ctx)]
        rng.shuffle(rest)
        ctx.extend(rest[: n_ctx - len(ctx)])
        ctx = ctx[:n_ctx]

        parent_dict = {"": parent}
        context_programs_dict = {"": ctx}
        return parent_dict, context_programs_dict


# EVOLVE-BLOCK-END