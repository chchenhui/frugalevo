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
    """Adaptive score-aware population manager for short optimization runs."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program: Optional[EvolvedProgram] = None
        self.random_state = random.Random(getattr(config, "random_seed", None))

        # State is updated only in add(), so it is available after restores
        # which replay program additions.
        self.best_score_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_iteration_seen = 0
        self.parent_use_count: Dict[str, int] = {}
        self.parent_success_count: Dict[str, int] = {}
        self.context_use_count: Dict[str, int] = {}
        self.label_trial_count: Dict[str, int] = {}
        self.score_by_id: Dict[str, float] = {}

    @staticmethod
    def _score(program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    @staticmethod
    def _meaningful_improvement(old_score: float, new_score: float) -> bool:
        if not math.isfinite(old_score) or new_score <= old_score:
            return False
        gain = new_score - old_score
        relative_gain = gain / max(abs(old_score), 1e-12)
        return gain > 0.01 or relative_gain > 0.01

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store a program and update lineage-based search statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        program_iteration = iteration
        if program_iteration is None:
            program_iteration = program.iteration_found
        if not isinstance(program_iteration, int):
            program_iteration = self.last_iteration_seen

        self.programs[program.id] = program
        self.last_iteration_seen = max(self.last_iteration_seen, program_iteration)

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        if score is not None:
            self.score_by_id[program.id] = score

            if self._meaningful_improvement(self.best_score_seen, score):
                self.last_meaningful_iteration = program_iteration
            if score > self.best_score_seen:
                self.best_score_seen = score

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            parent_score = self.score_by_id.get(program.parent_id)
            if score is not None and parent_score is not None:
                if self._meaningful_improvement(parent_score, score):
                    self.parent_success_count[program.parent_id] = (
                        self.parent_success_count.get(program.parent_id, 0) + 1
                    )

        for context_id in program.other_context_ids or []:
            self.context_use_count[context_id] = (
                self.context_use_count.get(context_id, 0) + 1
            )

        if program.parent_info and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self.label_trial_count[label] = self.label_trial_count.get(label, 0) + 1

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def _weighted_choice(
        self, programs: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        total = sum(weights)
        if total <= 0:
            return self.random_state.choice(programs)
        return self.random_state.choices(programs, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_count = max(0, num_context_programs or 0)
        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]

        if valid:
            valid.sort(key=lambda item: item[1], reverse=True)
            best_score = valid[0][1]
            worst_score = valid[-1][1]
            spread = max(best_score - worst_score, 1e-12)

            # Most mutations come from the strong half, while underused parents
            # and occasional broad exploration avoid repeatedly cloning one idea.
            elite_size = max(1, int(math.ceil(len(valid) * 0.65)))
            parent_pool = [p for p, _ in valid[:elite_size]]
            if len(valid) > elite_size and self.random_state.random() < 0.18:
                parent_pool = [p for p, _ in valid]

            parent_weights: List[float] = []
            for parent in parent_pool:
                score = self._score(parent) or worst_score
                quality = (score - worst_score) / spread
                uses = self.parent_use_count.get(parent.id, 0)
                successes = self.parent_success_count.get(parent.id, 0)
                parent_weights.append(
                    0.35
                    + 2.2 * quality
                    + 1.25 / math.sqrt(uses + 1)
                    + 0.45 * min(successes, 3)
                )
            parent = self._weighted_choice(parent_pool, parent_weights)
        else:
            parent = self.random_state.choice(candidates)
            valid = []

        # A stalled search gets one or two explicitly different attempts, but
        # labels are not the normal mechanism.
        stalled_for = self.last_iteration_seen - self.last_meaningful_iteration
        if (
            len(candidates) >= 6
            and stalled_for >= 6
            and self.label_trial_count.get(self.DIVERGE_LABEL, 0) < 2
        ):
            return {self.DIVERGE_LABEL: parent}, {}
        if (
            len(candidates) >= 8
            and stalled_for >= 4
            and self.parent_success_count.get(parent.id, 0) > 0
            and self.label_trial_count.get(self.REFINE_LABEL, 0) < 1
        ):
            return {self.REFINE_LABEL: parent}, {}

        remaining = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        while remaining and len(contexts) < context_count:
            weights: List[float] = []
            parent_score = self._score(parent)

            for candidate in remaining:
                candidate_score = self._score(candidate)
                quality = 0.4
                contrast = 0.0

                if valid and candidate_score is not None:
                    scores = [s for _, s in valid]
                    low, high = min(scores), max(scores)
                    quality = 0.35 + (candidate_score - low) / max(high - low, 1e-12)
                    if parent_score is not None:
                        contrast = abs(candidate_score - parent_score) / max(
                            high - low, 1e-12
                        )

                used_as_context = self.context_use_count.get(candidate.id, 0)
                lineage_bonus = 0.25 if candidate.parent_id != parent.parent_id else 0.0
                weights.append(
                    quality
                    + 0.75 / math.sqrt(used_as_context + 1)
                    + 0.35 * contrast
                    + lineage_bonus
                )

            chosen = self._weighted_choice(remaining, weights)
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END