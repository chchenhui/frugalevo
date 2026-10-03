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
    """Adaptive elite search with bounded exploration and diverse context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}
        self.best_seen_score = float("-inf")
        self.last_meaningful_improvement = 0
        self.add_count = 0

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
        self.add_count += 1

        current_iteration = iteration
        if current_iteration is None and isinstance(program.iteration_found, int):
            current_iteration = program.iteration_found
        if current_iteration is not None:
            self.last_iteration = max(self.last_iteration, current_iteration)

        # Reconstruct useful search state from loaded programs as well as new ones.
        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                if gain > 0:
                    self.parent_gains[program.parent_id] = (
                        self.parent_gains.get(program.parent_id, 0.0) + gain
                    )

        if isinstance(program.parent_info, tuple) and len(program.parent_info) >= 1:
            label = program.parent_info[0]
            if label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen_score == float("-inf"):
                self.best_seen_score = score
            else:
                absolute_gain = score - self.best_seen_score
                relative_gain = absolute_gain / max(abs(self.best_seen_score), 1e-12)
                if absolute_gain > 0.01 or relative_gain > 0.01:
                    self.last_meaningful_improvement = current_iteration or 0
                self.best_seen_score = max(self.best_seen_score, score)

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
        valid = [item for item in scored if item[0] is not None]
        valid.sort(key=lambda item: item[0], reverse=True)

        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        # Restrict normal mutation to the stronger half, while favoring parents
        # that have not already been repeatedly consumed.
        elite_count = max(2, (len(valid) + 1) // 2)
        elite = valid[:elite_count]
        weights = []
        for rank, (_, program) in enumerate(elite):
            reuse_penalty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            gain_bonus = 1.0 + min(2.0, self.parent_gains.get(program.id, 0.0) * 100.0)
            weights.append((elite_count - rank) * reuse_penalty * gain_bonus)
        parent = self.random_state.choices([p for _, p in elite], weights=weights, k=1)[0]

        iteration = getattr(self, "last_iteration", 0)
        stalled = iteration - self.last_meaningful_improvement >= 5

        # A deeply stalled, tightly clustered population merits an occasional
        # clean break rather than endlessly remixing near-identical contexts.
        if (
            stalled
            and self.label_uses.get(self.DIVERGE_LABEL, 0) < 2
            and self.random_state.random() < 0.45
        ):
            return {self.DIVERGE_LABEL: parent}, {}

        count = max(0, num_context_programs or 0)
        available = [p for _, p in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Include the best known reference, then deliberately span score tiers.
        if available and count:
            contexts.append(available[0])

        if len(contexts) < count and available:
            middle = available[len(available) // 2]
            if middle.id not in {p.id for p in contexts}:
                contexts.append(middle)

        if len(contexts) < count and available:
            tail = available[-1]
            if tail.id not in {p.id for p in contexts}:
                contexts.append(tail)

        remaining = [p for p in available if p.id not in {x.id for x in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[: max(0, count - len(contexts))])

        return {"": parent}, {"": contexts[:count]}


# EVOLVE-BLOCK-END