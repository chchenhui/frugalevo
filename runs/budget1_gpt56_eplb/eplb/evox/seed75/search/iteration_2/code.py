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
    """Small adaptive database balancing productive-parent reuse and exploration."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.last_iteration = 0
        self.last_meaningful_improvement_iteration = 0
        self.last_label_iteration = -1000

        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.parent_trials: Dict[str, int] = {}
        self.parent_positive_gain: Dict[str, float] = {}
        self.parent_best_child_score: Dict[str, float] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        found_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if isinstance(found_iteration, int):
            self.last_iteration = max(self.last_iteration, found_iteration)

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_usage[parent_id] = self.parent_usage.get(parent_id, 0) + 1
            self.parent_trials[parent_id] = self.parent_trials.get(parent_id, 0) + 1

            parent = self.get(parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None:
                previous_best = self.parent_best_child_score.get(parent_id)
                if previous_best is None or child_score > previous_best:
                    self.parent_best_child_score[parent_id] = child_score
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                if gain > 0:
                    self.parent_positive_gain[parent_id] = (
                        self.parent_positive_gain.get(parent_id, 0.0) + gain
                    )

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        parent_info = getattr(program, "parent_info", ("", ""))
        if isinstance(parent_info, tuple) and parent_info and parent_info[0]:
            self.last_label_iteration = self.last_iteration

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.best_program_id = program.id
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                old_best = self.best_score_seen
                delta = score - old_best
                relative = delta / max(abs(old_best), 1e-12)
                self.best_score_seen = score
                self.best_program_id = program.id
                if delta > 0.01 or relative > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        all_programs = list(self.programs.values())
        scored = [(self._score(p), p) for p in all_programs]
        numeric = [(score, p) for score, p in scored if score is not None]

        if not numeric:
            if not all_programs:
                raise ValueError("No candidates available for sampling")
            return {"": self.random_state.choice(all_programs)}, {"": []}

        numeric.sort(key=lambda item: item[0], reverse=True)
        scores = [score for score, _ in numeric]
        low, high = min(scores), max(scores)
        total_trials = max(1, sum(self.parent_trials.values()))

        # Score is useful, but direct observed child improvement is the main signal.
        weights: List[float] = []
        for score, program in numeric:
            quality = 0.45 + 0.95 * (score - low) / max(high - low, 1e-9)
            trials = self.parent_trials.get(program.id, 0)
            positive_gain = self.parent_positive_gain.get(program.id, 0.0)
            productivity = positive_gain / max(1, trials)
            exploration = math.sqrt(math.log(total_trials + 2.0) / (trials + 1.0))

            # A successful lower-score parent remains competitive, while repeatedly
            # sampled parents gradually lose priority.
            weight = quality + 45.0 * productivity + 0.28 * exploration
            weight /= 1.0 + 0.12 * self.parent_usage.get(program.id, 0)
            weights.append(max(weight, 0.001))

        parent = self.random_state.choices(
            [program for _, program in numeric], weights=weights, k=1
        )[0]

        stagnation = self.last_iteration - self.last_meaningful_improvement_iteration
        label_cooldown = self.last_iteration - self.last_label_iteration

        # One occasional targeted reset is useful after a real plateau, but normal
        # sampling remains the default because contexts are usually valuable.
        if (
            stagnation >= 10
            and label_cooldown >= 5
            and len(numeric) >= 6
            and self.random_state.random() < 0.35
        ):
            productive = sorted(
                numeric,
                key=lambda item: (
                    self.parent_positive_gain.get(item[1].id, 0.0),
                    item[0],
                ),
                reverse=True,
            )
            target = productive[0][1] if productive else parent
            return {self.DIVERGE_LABEL: target}, {}

        context_count = max(0, num_context_programs or 0)
        if context_count == 0:
            return {"": parent}, {"": []}

        available = [p for _, p in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        chosen_ids = set()

        def add_context(candidate: EvolvedProgram) -> None:
            if candidate.id not in chosen_ids and len(contexts) < context_count:
                contexts.append(candidate)
                chosen_ids.add(candidate.id)

        # Present the best known solution, then a productive alternative and a
        # contrasting score band rather than repeatedly feeding only weak programs.
        if available:
            add_context(available[0])

        productive_alternatives = sorted(
            available,
            key=lambda p: (
                self.parent_positive_gain.get(p.id, 0.0)
                / max(1, self.parent_trials.get(p.id, 0)),
                self._score(p) or float("-inf"),
            ),
            reverse=True,
        )
        if productive_alternatives:
            add_context(productive_alternatives[0])

        if available:
            add_context(available[len(available) // 2])

        remaining = [p for p in available if p.id not in chosen_ids]
        while remaining and len(contexts) < context_count:
            context_weights = [
                1.0 / (1.0 + self.context_usage.get(p.id, 0))
                for p in remaining
            ]
            selected = self.random_state.choices(
                remaining, weights=context_weights, k=1
            )[0]
            add_context(selected)
            remaining = [p for p in remaining if p.id != selected.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END