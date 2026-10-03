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
    """Usage-capped parent rotation with REFINE emphasis on top tier.

    The prior run showed REFINE on top programs produced the best gains while
    repeated DIVERGE on the same parent regressed. So we rotate parents across
    the top tier (never reusing one consecutively), lean REFINE when stalled,
    and only occasionally DIVERGE — always on a fresh, under-used parent.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.last_parent_id: str = ""
        self.best_score: float = -math.inf
        self.stagnation: int = 0
        self.label_counter: int = 0

    # ---------- helpers ----------
    def _score(self, program: EvolvedProgram) -> float:
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -1.0

    def _is_meaningful(self, new: float, old: float) -> bool:
        if old <= 0:
            return new > old + 0.01
        return (new - old) > max(0.01, 0.01 * abs(old))

    # ---------- API ----------
    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        s = self._score(program)
        if s > self.best_score:
            meaningful = self._is_meaningful(s, self.best_score) if self.best_score > -math.inf else True
            self.best_score = max(self.best_score, s)
            self.stagnation = 0 if meaningful else self.stagnation + 1
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
        scored = sorted(candidates, key=self._score, reverse=True)
        # Top tier: best quarter of population (keeps quality high but diverse).
        top = scored[: max(2, len(scored) // 4)]

        # Stalled? Use labels, but ROTATE parents and prefer REFINE (which
        # historically produced the new bests here).
        if self.stagnation >= 3:
            self.label_counter += 1
            # Pick a fresh parent from the top tier, least-used first, never
            # the same one twice in a row.
            fresh = [p for p in top if p.id != self.last_parent_id]
            fresh.sort(key=lambda p: self.parent_usage.get(p.id, 0))
            parent = fresh[0] if fresh else scored[0]
            if self.label_counter % 4 == 0:
                # Occasional divergence on a mid-tier under-used program.
                mid = [p for p in scored if p.id != parent.id]
                mid.sort(key=lambda p: self.parent_usage.get(p.id, 0))
                parent = mid[len(mid) // 2] if mid else parent
                label = self.DIVERGE_LABEL
            else:
                label = self.REFINE_LABEL
            self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
            self.last_parent_id = parent.id
            return {label: parent}, {"": []}

        # Normal mode: score-weighted pick over top tier with usage penalty
        # and no immediate repeats.
        pool = [p for p in top if p.id != self.last_parent_id] or top
        weights = [max(self._score(p), 0.01) / (1.0 + self.parent_usage.get(p.id, 0)) for p in pool]
        parent = self.random_state.choices(pool, weights=weights, k=1)[0]
        self.parent_usage[parent.id] = self.parent_usage.get(parent.id, 0) + 1
        self.last_parent_id = parent.id

        # Context: 2 top-tier siblings + 2 diverse mid/low-tier programs for
        # contrasting perspectives.
        others = [p for p in candidates if p.id != parent.id]
        top_others = [p for p in others if p in top]
        rest = [p for p in others if p not in top]
        self.random_state.shuffle(top_others)
        self.random_state.shuffle(rest)
        ctx = top_others[:2] + rest[: max(0, n_ctx - 2)]
        ctx = ctx[:n_ctx]

        return {"": parent}, {"": ctx}


# EVOLVE-BLOCK-END