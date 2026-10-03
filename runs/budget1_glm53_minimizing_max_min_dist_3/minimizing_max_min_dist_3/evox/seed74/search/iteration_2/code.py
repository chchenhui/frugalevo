# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive exploitation strategy for a saturated population.

    The population has converged at the score ceiling (1.0000). Since the
    downstream score is uncapped, the only way to improve is to push the best
    solutions *beyond* the ceiling. This strategy:
      1. Normally mutates top-tier programs (least-used first) with diverse
         top-tier context to consolidate and slightly exceed the ceiling.
      2. When stagnating (no meaningful gain for several iterations), it
         alternates between REFINE (push the best program further) and
         DIVERGE (try a fundamentally different geometry), avoiding repeats
         of divergence targets already tried.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        # State tracked in add() (persists across checkpoints)
        self.best_score: Optional[float] = None
        self.stagnation: int = 0
        # Usage / label tracking
        self.usage_counts: Dict[str, int] = {}
        self.diverge_tried: set() = set()  # program ids already diverged from
        self.refine_count: int = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        if score is not None:
            # Only count improvements as meaningful if > 1% relative or 0.01 absolute
            meaningful = False
            if self.best_score is None:
                meaningful = True
            elif score > self.best_score + 0.01 or score > self.best_score * 1.01:
                meaningful = True
            if meaningful:
                self.best_score = max(self.best_score or 0.0, score)
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
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4
        best = max(s for _, s in scored)

        # Top tier: within 1% of best (the saturated ceiling group)
        top = [p for p, s in scored if s >= best * 0.99]
        mid = [p for p, s in scored if s < best * 0.99]

        # ---- Stagnation mode: alternate REFINE / DIVERGE ----
        if self.stagnation >= 3:
            use_diverge = (len(self.diverge_tried) < max(1, len(top) // 2)) and (
                self.refine_count >= 2 or not self.diverge_tried
            )
            if use_diverge:
                # Diverge from a top program not yet diverged from (fresh geometry)
                fresh = [p for p in top if p.id not in self.diverge_tried]
                if fresh:
                    parent = min(fresh, key=lambda p: self.usage_counts.get(p.id, 0))
                    self.diverge_tried.add(parent.id)
                    self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1
                    return {self.DIVERGE_LABEL: parent}, {self.DIVERGE_LABEL: []}
            # Otherwise: refine the least-used top program, with a couple of
            # other top programs as context for comparison.
            parent = min(top, key=lambda p: self.usage_counts.get(p.id, 0))
            self.refine_count += 1
            self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1
            others = [p for p in top if p.id != parent.id]
            self.random_state.shuffle(others)
            ctx = others[: max(0, min(2, n_ctx))]
            return {self.REFINE_LABEL: parent}, {self.REFINE_LABEL: ctx}

        # ---- Normal mode: exploit top tier with least-usage rotation ----
        # Weighted choice favoring least-used top programs (avoids overuse)
        weights = [1.0 / (1 + self.usage_counts.get(p.id, 0)) for p in top]
        parent = self.random_state.choices(top, weights=weights, k=1)[0]
        self.usage_counts[parent.id] = self.usage_counts.get(parent.id, 0) + 1

        # Context: diverse top-tier programs (excluding parent) + one mid-tier
        # program for a contrasting perspective.
        others = [p for p in top if p.id != parent.id]
        self.random_state.shuffle(others)
        ctx = others[: max(0, n_ctx - 1)]
        if mid and len(ctx) < n_ctx:
            ctx.append(self.random_state.choice(mid))

        return {"": parent}, {"": ctx}
# EVOLVE-BLOCK-END