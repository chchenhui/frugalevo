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


def _score(p: EvolvedProgram) -> float:
    v = p.metrics.get("combined_score")
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Simple adaptive strategy: pick strong-but-not-overused parents,
    provide diverse context, and use REFINE sparingly on fresh top programs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_use_count: Dict[str, int] = {}
        self.combos_tried: set = set()
        self.calls = 0
        self.best_history: List[float] = []

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        s = _score(program)
        if s >= 0:
            self.best_history.append(max(self.best_history[-1] if self.best_history else s, s))
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        self.calls += 1
        n_ctx = num_context_programs or 4
        candidates = [p for p in self.programs.values() if _score(p) >= 0]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        # Top tier: top ~30% by score
        ranked = sorted(candidates, key=_score, reverse=True)
        top = ranked[: max(3, len(ranked) // 3)]

        # Prefer parents not overused; rotate through top tier
        top_sorted = sorted(
            top, key=lambda p: (self.parent_use_count.get(p.id, 0), -_score(p))
        )
        parent = top_sorted[self.calls % len(top_sorted)]
        # Occasionally take a different top parent for randomness
        if self.random_state.random() < 0.3:
            parent = self.random_state.choice(top)

        # Diverse context: best program + a few from different score bands
        context: List[EvolvedProgram] = []
        best = ranked[0]
        if best.id != parent.id:
            context.append(best)
        mid = ranked[len(ranked) // 2: len(ranked) * 3 // 4]
        low = ranked[-len(ranked) // 4:] if len(ranked) >= 4 else ranked
        pools = [mid, low, top]
        self.random_state.shuffle(pools)
        for pool in pools:
            self.random_state.shuffle(pool)
            for p in pool:
                if p.id != parent.id and all(c.id != p.id for c in context):
                    context.append(p)
                if len(context) >= n_ctx:
                    break
            if len(context) >= n_ctx:
                break

        # Track usage; avoid repeating identical parent+context combos
        self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1
        combo_key = (parent.id, tuple(sorted(c.id for c in context)))
        if combo_key in self.combos_tried and len(candidates) > n_ctx + 1:
            # swap one context program to break the repeat
            alt = [p for p in candidates if p.id != parent.id and p not in context]
            if alt:
                context[-1] = self.random_state.choice(alt)
        self.combos_tried.add((parent.id, tuple(sorted(c.id for c in context))))

        # Labels: mostly empty. REFINE occasionally on a fresh top parent
        # (rarely used before) when search is stagnating.
        label = ""
        stagnating = (
            len(self.best_history) > 5
            and self.best_history[-1] - self.best_history[-5] < 0.0005
        )
        if stagnating and self.random_state.random() < 0.25:
            fresh = [p for p in top if self.parent_use_count.get(p.id, 0) <= 1]
            if fresh:
                parent = self.random_state.choice(fresh)
                label = self.REFINE_LABEL
                context = []
                self.parent_use_count[parent.id] = self.parent_use_count.get(parent.id, 0) + 1

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END