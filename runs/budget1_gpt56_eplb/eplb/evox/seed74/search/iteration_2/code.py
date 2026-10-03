# EVOLVE-BLOCK-START
import logging
import math
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
    """Adaptive, diversity-preserving search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen_score = float("-inf")
        self.stagnant_additions = 0
        self.add_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.recent_parent_ids: List[str] = []
        self.label_use_count: Dict[str, int] = {
            self.DIVERGE_LABEL: 0,
            self.REFINE_LABEL: 0,
        }

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        score = float(value)
        return score if math.isfinite(score) else None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Add a program and retain lightweight progress/lineage statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        if score is not None:
            meaningful_delta = max(0.01, abs(self.best_seen_score) * 0.01)
            if self.best_seen_score == float("-inf") or score > self.best_seen_score + meaningful_delta:
                self.best_seen_score = score
                self.stagnant_additions = 0
            else:
                self.stagnant_additions += 1
                if score > self.best_seen_score:
                    self.best_seen_score = score

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            self.recent_parent_ids.append(program.parent_id)
            self.recent_parent_ids = self.recent_parent_ids[-4:]

        if program.parent_info and program.parent_info[0] in self.label_use_count:
            label = program.parent_info[0]
            self.label_use_count[label] += 1

        self.add_count += 1
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def _choose_parent(
        self, pool: List[EvolvedProgram], avoid_recent: bool = True
    ) -> EvolvedProgram:
        """Randomly favor underused parents without collapsing to one elite."""
        choices = pool[:]
        if avoid_recent:
            fresh = [p for p in choices if p.id not in self.recent_parent_ids]
            if fresh:
                choices = fresh

        weights = [
            1.0 / (1.0 + self.parent_use_count.get(p.id, 0))
            for p in choices
        ]
        return self.random_state.choices(choices, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        valid = [p for p, score in scored if score is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        ranked = sorted(valid, key=lambda p: self._score(p) or float("-inf"), reverse=True)
        n = len(ranked)
        elite = ranked[:max(1, min(3, n))]
        middle = ranked[max(1, n // 4):max(2, (3 * n) // 4)] or ranked
        lower = ranked[max(1, n // 2):] or ranked

        label = ""
        # A prolonged plateau calls for occasional explicitly directed attempts,
        # but ordinary diverse mutations remain the dominant behavior.
        if self.stagnant_additions >= 10 and self.random_state.random() < 0.25:
            if self.label_use_count[self.REFINE_LABEL] <= self.label_use_count[self.DIVERGE_LABEL]:
                label = self.REFINE_LABEL
                parent = self._choose_parent(elite)
            else:
                label = self.DIVERGE_LABEL
                parent = self._choose_parent(middle or lower)
        else:
            # During this observed plateau, upper/middle approaches deserve more
            # attempts than repeatedly mutating the current best.
            roll = self.random_state.random()
            if self.stagnant_additions >= 6 and roll < 0.55:
                parent = self._choose_parent(middle)
            elif roll < 0.82:
                parent = self._choose_parent(elite)
            else:
                parent = self._choose_parent(lower)

        if label:
            return {label: parent}, {}

        requested = 4 if num_context_programs is None else max(0, num_context_programs)
        available = [p for p in ranked if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model contrasting evidence: an elite reference, a mid-score
        # alternative, and a lower-score approach that may contain a distinct idea.
        tiers = [elite, middle, lower]
        for tier in tiers:
            options = [p for p in tier if p.id != parent.id and p.id not in {c.id for c in contexts}]
            if options and len(contexts) < requested:
                contexts.append(self.random_state.choice(options))

        remaining = [
            p for p in available if p.id not in {c.id for c in contexts}
        ]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, requested - len(contexts))])

        return {"": parent}, {"": contexts[:requested]}


# EVOLVE-BLOCK-END