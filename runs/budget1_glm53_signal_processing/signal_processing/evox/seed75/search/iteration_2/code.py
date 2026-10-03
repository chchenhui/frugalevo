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


def _score(p: Program) -> float:
    v = (p.metrics or {}).get("combined_score")
    if isinstance(v, (int, float)):
        return float(v)
    return -1.0


class EvolvedProgramDatabase(ProgramDatabase):
    """Search strategy: score-weighted parent selection with usage rotation,
    diverse context, and stagnation-triggered REFINE/DIVERGE labels."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.best_score: float = -1.0
        self.best_improve_iter: int = 0
        self.iteration: int = 0
        self.label_cycle: int = 0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
            self.iteration = max(self.iteration, iteration)

        s = _score(program)
        if s > self.best_score + 0.01:  # meaningful improvement
            self.best_score = s
            self.best_improve_iter = self.iteration

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")
        k = num_context_programs or 4
        rng = self.random_state

        scored = [(p, _score(p)) for p in candidates]
        scored = [(p, s) for p, s in scored if s >= 0]
        scored.sort(key=lambda x: -x[1])
        top = scored[: max(3, len(scored) // 4)]

        stagnation = self.iteration - self.best_improve_iter

        # --- Parent selection: rotate among top programs, penalizing overuse ---
        if top:
            weights = [1.0 / (1 + self.parent_usage.get(p.id, 0)) for p, _ in top]
            parent, parent_score = rng.choices(top, weights=weights, k=1)[0]
        else:
            parent = rng.choice(candidates)
            parent_score = _score(parent)

        # --- Context: diverse mix — best, median, worst, and random non-parent ---
        context: List[EvolvedProgram] = []
        others = [p for p in candidates if p.id != parent.id]
        if others:
            osorted = sorted(others, key=_score)
            picks = []
            if osorted:
                picks.append(osorted[-1])  # best other
                picks.append(osorted[len(osorted) // 2])  # median
                if len(osorted) > 2:
                    picks.append(osorted[0])  # worst (contrast)
                rng.shuffle(others)
                for p in others:
                    if p not in picks and len(picks) >= k:
                        break
                    if p not in picks:
                        picks.append(p)
            context = picks[:k]

        # --- Label logic on stagnation ---
        label = ""
        if stagnation >= 4:
            self.label_cycle += 1
            if self.label_cycle % 3 == 1:
                # refine the current best program
                best = max(candidates, key=_score)
                parent, label = best, self.REFINE_LABEL
                context = []
            elif self.label_cycle % 3 == 2:
                # diverge from an underused mid-tier program
                mid = [p for p in candidates if 0 < _score(p) < parent_score]
                if mid:
                    parent, label = rng.choice(mid), self.DIVERGE_LABEL
                context = []

        return {label: parent}, {"": context}


# EVOLVE-BLOCK-END