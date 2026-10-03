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
    """Adaptive quality-diverse search for near-saturated populations."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = -float("inf")
        self.last_meaningful_improvement = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.last_iteration = getattr(self, "last_iteration", 0)

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            threshold = max(0.01, abs(self.best_seen) * 0.01) if math.isfinite(self.best_seen) else 0.0
            if score > self.best_seen:
                if not math.isfinite(self.best_seen) or score - self.best_seen > threshold:
                    self.last_meaningful_improvement = self.last_iteration
                self.best_seen = score

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

        scored = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in scored if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best = valid[0][1]
        # At a near-optimal plateau, use several excellent but differently-used
        # parents rather than repeatedly mutating one identical best solution.
        elite = [(p, s) for p, s in valid if s >= best - 0.015]
        elite = elite[:max(8, min(20, len(elite)))]

        weights = []
        for program, score in elite:
            uses = self.parent_uses.get(program.id, 0)
            weights.append((1.0 + max(0.0, score - (best - 0.02)) * 20.0) / (1.0 + uses))
        parent = self.random_state.choices([p for p, _ in elite], weights=weights, k=1)[0]

        stalled = self.last_iteration - self.last_meaningful_improvement >= 12
        label = ""
        if stalled and len(elite) > 1 and self.random_state.random() < 0.30:
            # Pick the less-used intervention during a genuine plateau.
            diverges = self.label_uses.get(self.DIVERGE_LABEL, 0)
            refines = self.label_uses.get(self.REFINE_LABEL, 0)
            label = self.DIVERGE_LABEL if diverges <= refines else self.REFINE_LABEL
            return {label: parent}, {}

        count = max(0, num_context_programs or 0)
        pool = [p for p, _ in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # First include strong alternatives, then deliberately include a
        # different score level to expose a contrasting construction.
        for p, _ in elite:
            if p.id != parent.id and p.id not in {x.id for x in contexts}:
                contexts.append(p)
            if len(contexts) >= min(2, count):
                break

        remaining = [p for p in pool if p.id not in {x.id for x in contexts}]
        if remaining and len(contexts) < count:
            middle = remaining[len(remaining) // 3 : max(len(remaining) // 3 + 1, 2 * len(remaining) // 3)]
            if middle:
                contexts.append(self.random_state.choice(middle))

        self.random_state.shuffle(remaining)
        for p in remaining:
            if len(contexts) >= count:
                break
            if p.id not in {x.id for x in contexts}:
                contexts.append(p)

        return {"": parent}, {"": contexts[:count]}


# EVOLVE-BLOCK-END