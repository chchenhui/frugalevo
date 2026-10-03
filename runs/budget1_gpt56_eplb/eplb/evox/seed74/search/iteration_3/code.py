# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive population search with score-aware, diverse sampling."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.label_usage: Dict[str, int] = {}
        self.best_seen_score: Optional[float] = None
        self.last_meaningful_improvement_iteration = 0
        self.last_label_iteration = -1000

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if current_iteration is None:
            current_iteration = 0
        self.last_iteration = max(getattr(self, "last_iteration", 0), current_iteration)

        if program.parent_id:
            self.parent_usage[program.parent_id] = (
                self.parent_usage.get(program.parent_id, 0) + 1
            )
        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        label = program.parent_info[0] if program.parent_info else ""
        if label:
            self.label_usage[label] = self.label_usage.get(label, 0) + 1
            self.last_label_iteration = current_iteration

        score = self._score(program)
        if score is not None:
            if self.best_seen_score is None:
                self.best_seen_score = score
                self.last_meaningful_improvement_iteration = current_iteration
            elif score > self.best_seen_score:
                improvement = score - self.best_seen_score
                meaningful = (
                    improvement > 0.01
                    or improvement > abs(self.best_seen_score) * 0.01
                )
                self.best_seen_score = score
                if meaningful:
                    self.last_meaningful_improvement_iteration = current_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        scored = [(s, p) for s, p in scored if s is not None]
        if not scored:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        scored.sort(key=lambda item: item[0], reverse=True)
        scores = [item[0] for item in scored]
        lo, hi = min(scores), max(scores)
        span = max(hi - lo, 1e-9)

        # Favor good programs, but strongly reward underused parents.  This
        # preserves useful lower-score approaches rather than collapsing onto
        # the repeatedly observed top-score variants.
        parent_pool = [item[1] for item in scored]
        weights = []
        for program in parent_pool:
            normalized_score = (self._score(program) - lo) / span
            novelty = 2.0 / (1 + self.parent_usage.get(program.id, 0))
            weights.append(0.4 + normalized_score + novelty)
        parent = self.random_state.choices(parent_pool, weights=weights, k=1)[0]

        current_iteration = getattr(self, "last_iteration", 0)
        stagnant_for = current_iteration - self.last_meaningful_improvement_iteration
        can_refine = (
            stagnant_for >= 16
            and self.label_usage.get(self.REFINE_LABEL, 0) < 2
            and current_iteration - self.last_label_iteration >= 6
        )

        # A rare focused refinement is appropriate after a long plateau.  Pick
        # among several elite programs so the same best program is not targeted
        # repeatedly.
        if can_refine:
            elite = [p for _, p in scored[: min(3, len(scored))]]
            least_used = min(self.parent_usage.get(p.id, 0) for p in elite)
            elite = [p for p in elite if self.parent_usage.get(p.id, 0) == least_used]
            return {self.REFINE_LABEL: self.random_state.choice(elite)}, {}

        count = max(0, num_context_programs or 0)
        available = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model an elite reference, a contrasting middle candidate,
        # and randomized remaining examples to avoid repeated context sets.
        for index in (0, len(scored) // 2):
            if len(contexts) >= count:
                break
            candidate = scored[index][1]
            if candidate.id != parent.id and candidate.id not in {p.id for p in contexts}:
                contexts.append(candidate)

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        while remaining and len(contexts) < count:
            weights = [1.0 / (1 + self.context_usage.get(p.id, 0)) for p in remaining]
            chosen = self.random_state.choices(remaining, weights=weights, k=1)[0]
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END