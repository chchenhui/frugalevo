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
    """Adaptive elite search with lineage-aware exploration."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program: Optional[EvolvedProgram] = None
        self.best_score_seen: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.last_iteration = 0
        self.last_meaningful_improvement_iteration = 0
        self.last_label_iteration = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.parent_trials: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}
        self.label_usage: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = (
            iteration
            if isinstance(iteration, int)
            else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.iteration_found == 0 or current_iteration == 0:
            self.initial_program = program

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_usage[parent_id] = self.parent_usage.get(parent_id, 0) + 1
            self.parent_trials[parent_id] = self.parent_trials.get(parent_id, 0) + 1

            parent = self.get(parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if (
                child_score is not None
                and parent_score is not None
                and child_score > parent_score
            ):
                self.parent_wins[parent_id] = self.parent_wins.get(parent_id, 0) + 1

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        parent_info = getattr(program, "parent_info", ("", ""))
        if isinstance(parent_info, tuple) and parent_info:
            label = parent_info[0]
            if label in (self.DIVERGE_LABEL, self.REFINE_LABEL):
                self.label_usage[label] = self.label_usage.get(label, 0) + 1
                self.last_label_iteration = self.last_iteration

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.best_program_id = program.id
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                previous_best = self.best_score_seen
                gain = score - previous_best
                relative_gain = gain / max(abs(previous_best), 1e-12)
                self.best_score_seen = score
                self.best_program_id = program.id

                if gain > 0.01 or relative_gain > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        programs = list(self.programs.values())
        if not programs:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(program), program) for program in programs]
        numeric = [(score, program) for score, program in scored if score is not None]
        numeric.sort(key=lambda item: item[0], reverse=True)

        if not numeric:
            return {"": self.random_state.choice(programs)}, {"": []}

        stagnation = self.last_iteration - self.last_meaningful_improvement_iteration
        elite_size = min(len(numeric), max(4, int(math.ceil(len(numeric) * 0.30))))
        elite = numeric[:elite_size]

        # During a deep plateau, explicitly request the intervention type that has
        # been used least. In this population REFINE is especially valuable because
        # divergence has already been tried repeatedly.
        if (
            stagnation >= 10
            and self.last_iteration - self.last_label_iteration >= 5
            and len(numeric) >= 4
        ):
            refine_count = self.label_usage.get(self.REFINE_LABEL, 0)
            diverge_count = self.label_usage.get(self.DIVERGE_LABEL, 0)

            if refine_count <= diverge_count:
                target = elite[0][1]
                return {self.REFINE_LABEL: target}, {}

            choices = [program for _, program in elite]
            weights = [
                1.0 / (1.0 + self.parent_usage.get(program.id, 0))
                for program in choices
            ]
            target = self.random_state.choices(choices, weights=weights, k=1)[0]
            return {self.DIVERGE_LABEL: target}, {}

        # Favor strong candidates, but retain multiple elite lineages and reward
        # parents whose previous children actually improved upon them.
        parent_choices = [program for _, program in elite]
        best_elite_score = elite[0][0]
        worst_elite_score = elite[-1][0]
        weights: List[float] = []

        for score, program in elite:
            normalized_score = 1.0 + (
                (score - worst_elite_score)
                / max(best_elite_score - worst_elite_score, 1e-9)
            )
            trials = self.parent_trials.get(program.id, 0)
            wins = self.parent_wins.get(program.id, 0)
            success_bonus = 1.0 + (wins / max(trials, 1))
            reuse_penalty = 1.0 + 0.7 * self.parent_usage.get(program.id, 0)
            weights.append(normalized_score * success_bonus / reuse_penalty)

        parent = self.random_state.choices(parent_choices, weights=weights, k=1)[0]

        context_count = max(0, num_context_programs or 0)
        if context_count == 0:
            return {"": parent}, {"": []}

        available = [program for _, program in numeric if program.id != parent.id]
        contexts: List[EvolvedProgram] = []
        used_ids = set()

        # Sample high quality alternatives rather than always attaching the same
        # best solution. This is important where many candidates tie near the top.
        top_pool = available[: min(len(available), max(6, elite_size * 2))]
        while top_pool and len(contexts) < min(2, context_count):
            weights = [
                1.0 / (1.0 + self.context_usage.get(program.id, 0))
                for program in top_pool
            ]
            chosen = self.random_state.choices(top_pool, weights=weights, k=1)[0]
            contexts.append(chosen)
            used_ids.add(chosen.id)
            top_pool = [program for program in top_pool if program.id != chosen.id]

        # A contrasting middle/lower candidate can preserve useful alternative
        # placement strategies without making low-score programs the parent.
        remaining = [program for program in available if program.id not in used_ids]
        if remaining and len(contexts) < context_count:
            contrast_pool = remaining[len(remaining) // 2 :]
            if contrast_pool:
                chosen = self.random_state.choice(contrast_pool)
                contexts.append(chosen)
                used_ids.add(chosen.id)

        remaining = [program for program in available if program.id not in used_ids]
        while remaining and len(contexts) < context_count:
            weights = [
                1.0 / (1.0 + self.context_usage.get(program.id, 0))
                for program in remaining
            ]
            chosen = self.random_state.choices(remaining, weights=weights, k=1)[0]
            contexts.append(chosen)
            used_ids.add(chosen.id)
            remaining = [program for program in remaining if program.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END