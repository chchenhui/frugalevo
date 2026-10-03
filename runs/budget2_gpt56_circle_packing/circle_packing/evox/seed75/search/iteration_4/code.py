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
    """Adaptive late-stage search with elite refinement and varied context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.last_meaningful_improvement: int = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}
        self.label_history: List[str] = []
        self.initial_program = None

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        current_iteration = iteration
        if not isinstance(current_iteration, int):
            current_iteration = (
                program.iteration_found
                if isinstance(program.iteration_found, int)
                else 0
            )
        self.last_iteration = max(getattr(self, "last_iteration", 0), current_iteration)

        # Record actual outcomes of prior choices here, so state survives
        # checkpointing/replay through add().
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1

        if isinstance(program.parent_info, tuple) and program.parent_info:
            label = program.parent_info[0]
            if label in (self.DIVERGE_LABEL, self.REFINE_LABEL):
                self.label_history.append(label)
                self.label_history = self.label_history[-8:]

        score = self._score(program)
        if score is not None:
            parent = self.get(program.parent_id) if program.parent_id else None
            parent_score = self._score(parent) if parent is not None else None
            if parent_score is not None and score > parent_score:
                self.parent_wins[program.parent_id] = self.parent_wins.get(program.parent_id, 0) + 1

            if self.best_score_seen is None:
                self.best_score_seen = score
                self.last_meaningful_improvement = current_iteration
            else:
                gain = score - self.best_score_seen
                meaningful = gain > 0.01 or (
                    self.best_score_seen != 0
                    and gain / abs(self.best_score_seen) > 0.01
                )
                if score > self.best_score_seen:
                    self.best_score_seen = score
                if meaningful:
                    self.last_meaningful_improvement = current_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        numeric = [(s, p) for s, p in scored if s is not None]
        numeric.sort(key=lambda item: item[0], reverse=True)

        if not numeric:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        # Late in a plateau, search around the small elite frontier rather than
        # repeatedly mutating mediocre programs.
        elite_size = min(8, len(numeric))
        elite = numeric[:elite_size]
        low_score = elite[-1][0]
        weights = []
        for score, program in elite:
            quality = 1.0 + 8.0 * max(0.0, score - low_score)
            reuse_penalty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            win_bonus = 1.0 + 0.25 * self.parent_wins.get(program.id, 0)
            weights.append(quality * reuse_penalty * win_bonus)

        parent = self.random_state.choices(
            [p for _, p in elite], weights=weights, k=1
        )[0]

        current_iteration = getattr(self, "last_iteration", 0)
        stalled = (
            current_iteration - self.last_meaningful_improvement >= 6
            and len(numeric) >= 3
        )

        # Refinement was productive in this population. Use it sparingly, with
        # an unlabeled/contextual generation between refinement attempts.
        recent_refines = self.label_history[-3:].count(self.REFINE_LABEL)
        if stalled and recent_refines == 0:
            return {self.REFINE_LABEL: parent}, {"": []}

        context_count = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        used_ids = {parent.id}

        # Give the model contrasting high-quality constructions, preferring
        # distinct score levels over many near-identical elite duplicates.
        seen_scores = set()
        for score, program in numeric:
            rounded = round(score, 4)
            if program.id not in used_ids and rounded not in seen_scores:
                contexts.append(program)
                used_ids.add(program.id)
                seen_scores.add(rounded)
            if len(contexts) >= context_count:
                break

        if len(contexts) < context_count:
            remaining = [p for _, p in numeric if p.id not in used_ids]
            self.random_state.shuffle(remaining)
            contexts.extend(remaining[: context_count - len(contexts)])

        return {"": parent}, {"": contexts[:context_count]}


# EVOLVE-BLOCK-END