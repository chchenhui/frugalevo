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
    """Adaptive elite-and-diversity search for packing optimizers."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen = None
        self.last_meaningful_iteration = 0
        self.parent_children: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}
        self.added_count = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program
        self.added_count += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        parent = self.get(program.parent_id) if program.parent_id else None
        parent_score = self._score(parent) if parent else None

        if parent is not None:
            self.parent_children[parent.id] = self.parent_children.get(parent.id, 0) + 1
            if score is not None and parent_score is not None:
                gain = score - parent_score
                if gain > max(0.01, abs(parent_score) * 0.01):
                    self.parent_wins[parent.id] = self.parent_wins.get(parent.id, 0) + 1

        if score is not None:
            if self.best_seen is None:
                self.best_seen = score
            elif score > self.best_seen + max(0.01, abs(self.best_seen) * 0.01):
                self.best_seen = score
                self.last_meaningful_iteration = (
                    iteration if iteration is not None else program.iteration_found
                )
            else:
                self.best_seen = max(self.best_seen, score)

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(self._score(p), p) for p in self.programs.values()]
        scored = [(s, p) for s, p in scored if s is not None]

        if not scored:
            candidates = list(self.programs.values())
            if not candidates:
                raise ValueError("No candidates available for sampling")
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        scored.sort(key=lambda item: item[0], reverse=True)
        n = len(scored)
        elite = [p for _, p in scored[:max(3, min(8, n // 2 + 1))]]
        middle = [p for _, p in scored[max(1, n // 4):max(2, 3 * n // 4)]]
        current_iteration = max(
            [getattr(p, "iteration_found", 0) for _, p in scored] + [self.last_iteration]
        )
        stalled = current_iteration - self.last_meaningful_iteration >= 6

        # Favor strong parents that have not already been repeatedly exhausted.
        def parent_value(p: EvolvedProgram) -> float:
            score = self._score(p) or 0.0
            uses = self.parent_children.get(p.id, 0)
            wins = self.parent_wins.get(p.id, 0)
            return score + 0.003 * wins - 0.0015 * uses + self.random_state.random() * 0.002

        if stalled and middle and self.random_state.random() < 0.35:
            parent = max(middle, key=parent_value)
        else:
            parent = max(elite, key=parent_value)

        label = ""
        # A rare explicit divergence is useful only after a genuine plateau.
        if stalled and self.random_state.random() < 0.18:
            label = self.DIVERGE_LABEL
            return {label: parent}, {"" : []}

        # Context combines near-best implementations with one contrasting attempt.
        context_count = max(0, num_context_programs or 0)
        pool = [p for p in elite if p.id != parent.id]
        self.random_state.shuffle(pool)
        context = pool[:max(0, context_count - 1)]

        contrasting = [
            p for _, p in scored[max(1, n // 2):]
            if p.id != parent.id and p.id not in {x.id for x in context}
        ]
        if contrasting and len(context) < context_count:
            context.append(self.random_state.choice(contrasting))

        if len(context) < context_count:
            remaining = [
                p for _, p in scored
                if p.id != parent.id and p.id not in {x.id for x in context}
            ]
            self.random_state.shuffle(remaining)
            context.extend(remaining[:context_count - len(context)])

        return {label: parent}, {"": context[:context_count]}


# EVOLVE-BLOCK-END