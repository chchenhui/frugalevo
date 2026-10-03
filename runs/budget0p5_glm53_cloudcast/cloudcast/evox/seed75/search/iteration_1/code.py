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
    """Adaptive search: exploit best programs but rotate usage, inject diversity,
    and use DIVERGE/REFINE labels when stagnating."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = -math.inf
        self.last_improve_iter = 0
        self.iter_counter = 0
        self.usage_count: Dict[str, int] = {}

    def _score(self, p: EvolvedProgram) -> float:
        v = p.metrics.get("combined_score") if p.metrics else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        return -math.inf

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
            self.iter_counter = max(self.iter_counter, iteration)
        s = self._score(program)
        if s > self.best_score and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)):
            self.best_score = s
            self.last_improve_iter = self.iter_counter
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        cands = list(self.programs.values())
        if not cands:
            raise ValueError("No candidates available for sampling")
        n_ctx = num_context_programs or 4
        self.iter_counter += 1

        # Score-sorted population
        scored = sorted(cands, key=self._score, reverse=True)
        stagnation = self.iter_counter - self.last_improve_iter

        # --- Parent selection: weighted by score, penalized by past usage ---
        top = scored[: max(4, len(scored) // 3)]
        weights = []
        for p in top:
            s = self._score(p)
            base = (s - scored[-1]._score if hasattr(scored[-1], "_score") else 0) or 0
            w = 1.0 + max(0.0, s - min(self._score(q) for q in top))
            w /= (1 + self.usage_count.get(p.id, 0))
            weights.append(max(w, 0.05))
        parent = self.random_state.choices(top, weights=weights, k=1)[0]

        # --- Label logic based on stagnation ---
        label = ""
        if stagnation >= 4:
            r = self.random_state.random()
            if r < 0.4:
                # Diverge from a mid/low scorer (different approach) to escape plateau
                label = self.DIVERGE_LABEL
                pool = scored[len(scored) // 3:] or scored
                parent = self.random_state.choice(pool)
            elif r < 0.8:
                # Refine the current best
                label = self.REFINE_LABEL
                parent = scored[0]

        self.usage_count[parent.id] = self.usage_count.get(parent.id, 0) + 1

        # --- Context: diverse, avoid reusing same combos ---
        others = [p for p in cands if p.id != parent.id]
        self.random_state.shuffle(others)
        # Mix: best scorer, worst scorer (failure insight), and random others
        ctx: List[EvolvedProgram] = []
        if others:
            ctx.append(scored[0] if scored[0].id != parent.id else others[0])
        if len(others) > 2:
            ctx.append(scored[-1] if scored[-1].id != parent.id else others[-1])
        for p in others:
            if len(ctx) >= n_ctx:
                break
            if p not in ctx:
                ctx.append(p)
        ctx = ctx[:n_ctx]

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END