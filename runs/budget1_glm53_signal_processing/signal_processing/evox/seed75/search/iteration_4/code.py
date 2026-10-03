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


def _score(p) -> float:
    v = p.metrics.get("combined_score")
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Search strategy: usage-penalized top-tier parent rotation with
    diverse score-range context (best / median / worst / random).

    Key insight from prior runs: all breakthroughs came from unlabeled
    parents with diverse context; stagnation-triggered DIVERGE/REFINE
    labels on the same top programs produced only regressions.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.sample_calls = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        n_ctx = num_context_programs or 4
        self.sample_calls += 1
        candidates = [p for p in self.programs.values() if _score(p) > 0.0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        candidates.sort(key=_score, reverse=True)

        # --- Parent selection: top tier with usage penalty, occasional mid-tier pick ---
        top = candidates[: max(3, len(candidates) // 4)]
        if self.sample_calls % 4 == 0 and len(candidates) > 6:
            # Occasionally pick a mid-tier parent (25-75th percentile)
            mid = candidates[len(candidates) // 4: 3 * len(candidates) // 4]
            parent = self.random_state.choice(mid)
        else:
            # Weighted by score, penalized by past usage (avoid overusing same lineage)
            weights = []
            for p in top:
                w = max(_score(p), 0.01) ** 4
                w /= (1 + self.parent_usage.get(p.id, 0))
                weights.append(w)
            parent = self.random_state.choices(top, weights=weights, k=1)[0]

        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1

        # --- Context: diverse score ranges (best, median, worst, random) ---
        others = [p for p in candidates if p.id != parent.id]
        ctx: List[EvolvedProgram] = []
        if others:
            ctx.append(others[0])  # best
        if len(others) > 2:
            ctx.append(others[len(others) // 2])  # median
        if len(others) > 1:
            ctx.append(others[-1])  # worst (failure insight)
        # fill remaining slots randomly (dedup by id)
        seen = {p.id for p in ctx}
        pool = [p for p in others if p.id not in seen]
        self.random_state.shuffle(pool)
        ctx.extend(pool)
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END