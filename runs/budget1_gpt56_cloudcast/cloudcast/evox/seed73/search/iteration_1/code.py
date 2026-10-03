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
    """Adaptive score-aware search with parent and context diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen_score: Optional[float] = None
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
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
            iteration if iteration is not None else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if isinstance(program.parent_info, tuple) and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1
                if isinstance(current_iteration, int):
                    self.last_label_iteration = max(
                        self.last_label_iteration, current_iteration
                    )

        score = self._score(program)
        if score is not None:
            if self.best_seen_score is None:
                self.best_seen_score = score
            else:
                improvement = score - self.best_seen_score
                relative = (
                    improvement / abs(self.best_seen_score)
                    if self.best_seen_score != 0
                    else 0.0
                )
                meaningful = improvement > 0.01 or relative > 0.01
                if meaningful:
                    self.best_seen_score = score
                    self.stagnation_steps = 0
                else:
                    self.stagnation_steps += 1
                    if score > self.best_seen_score:
                        self.best_seen_score = score

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

        context_count = max(0, num_context_programs or 0)
        scored = [(self._score(p), p) for p in candidates]
        numeric = [(score, p) for score, p in scored if score is not None]

        if numeric:
            numeric.sort(key=lambda item: item[0], reverse=True)
            elite_size = max(2, (len(numeric) + 2) // 3)
            elite = [p for _, p in numeric[:elite_size]]

            # Most mutations come from strong solutions, but favor parents that
            # have not already been repeatedly tried.
            if self.random_state.random() < 0.80:
                minimum_use = min(self.parent_uses.get(p.id, 0) for p in elite)
                parent_pool = [
                    p for p in elite
                    if self.parent_uses.get(p.id, 0) <= minimum_use + 1
                ]
            else:
                # Periodically test a non-elite branch rather than converging
                # permanently on equivalent top-score programs.
                parent_pool = [p for _, p in numeric[elite_size:]] or elite
            parent = self.random_state.choice(parent_pool)
        else:
            parent = self.random_state.choice(candidates)
            numeric = []

        label = ""
        label_gap = self.last_iteration - self.last_label_iteration
        if self.stagnation_steps >= 10 and label_gap >= 5:
            # A prolonged plateau calls for a deliberately different direction.
            if self.random_state.random() < 0.30:
                label = self.DIVERGE_LABEL
        elif self.stagnation_steps >= 6 and label_gap >= 5:
            # Shorter stalls can still benefit from focused polishing of an
            # underused strong candidate.
            if self.random_state.random() < 0.15:
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {}

        available = [p for p in candidates if p.id != parent.id]
        selected: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}

        # Give the model both the best alternative and a contrasting branch.
        ordered_context = []
        if numeric:
            ordered_context.extend(
                [p for _, p in numeric if p.id != parent.id]
            )
            ordered_context.extend(
                [p for _, p in reversed(numeric) if p.id != parent.id]
            )
        self.random_state.shuffle(available)
        ordered_context.extend(available)

        for candidate in ordered_context:
            if len(selected) >= context_count:
                break
            if candidate.id == parent.id or candidate.id in {p.id for p in selected}:
                continue
            if candidate.solution in seen_solutions and len(available) > context_count:
                continue
            selected.append(candidate)
            seen_solutions.add(candidate.solution)

        # If all solutions are textually similar, still provide distinct
        # program histories rather than returning too little context.
        if len(selected) < context_count:
            for candidate in available:
                if len(selected) >= context_count:
                    break
                if candidate.id not in {p.id for p in selected}:
                    selected.append(candidate)

        return {"": parent}, {"": selected}


# EVOLVE-BLOCK-END