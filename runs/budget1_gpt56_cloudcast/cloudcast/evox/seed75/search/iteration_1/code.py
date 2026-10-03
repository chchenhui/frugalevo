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
        self.best_score_seen: Optional[float] = None
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_successes: Dict[str, int] = {}
        self.last_seen_iteration = 0

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def _meaningful_improvement(self, old: Optional[float], new: float) -> bool:
        if old is None:
            return True
        gain = new - old
        return gain > 0.01 or (old != 0 and gain / abs(old) > 0.01)

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        found_at = iteration if iteration is not None else program.iteration_found
        if isinstance(found_at, int):
            self.last_seen_iteration = max(self.last_seen_iteration, found_at)
            self.last_iteration = max(self.last_iteration, found_at)

        score = self._score(program)
        if score is not None:
            if self._meaningful_improvement(self.best_score_seen, score):
                self.best_score_seen = score
                self.stagnation_steps = 0
            else:
                self.stagnation_steps += 1

        # Child results reveal whether a parent has been productive.
        if program.parent_id:
            parent = self.get(program.parent_id)
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and self._meaningful_improvement(parent_score, score):
                self.parent_successes[program.parent_id] = (
                    self.parent_successes.get(program.parent_id, 0) + 1
                )

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
        numeric = [(p, s) for p, s in scored if s is not None]
        numeric.sort(key=lambda item: item[1], reverse=True)

        # Usually exploit the upper half, but periodically admit a different
        # lineage to avoid repeatedly mutating identical plateau solutions.
        if numeric:
            elite_size = max(3, (len(numeric) + 1) // 2)
            pool = [p for p, _ in numeric[:elite_size]]
            if self.random_state.random() < 0.25:
                pool = [p for p, _ in numeric]
        else:
            pool = candidates[:]

        weights = []
        for program in pool:
            uses = self.parent_uses.get(program.id, 0)
            successes = self.parent_successes.get(program.id, 0)
            # Successful, underexplored parents receive more opportunities.
            weights.append((1.0 + successes) / (1.0 + uses))

        parent = self.random_state.choices(pool, weights=weights, k=1)[0]
        label = ""

        # Labels are reserved for an established plateau, and only occasionally
        # used so normal parent/context selection remains the primary mechanism.
        if self.stagnation_steps >= 8:
            phase = self.last_seen_iteration % 6
            if phase == 0:
                label = self.DIVERGE_LABEL
            elif phase == 3:
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"": []}

        count = max(0, num_context_programs or 0)
        remaining = [p for p in candidates if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        seen_ids = set()
        seen_solutions = {parent.solution}

        # Prefer contrasting solutions, retaining both strong examples and
        # occasional weaker alternatives that may represent another approach.
        ordered = sorted(
            remaining,
            key=lambda p: self._score(p) if self._score(p) is not None else float("-inf"),
            reverse=True,
        )
        if ordered:
            ordered = [ordered[0]] + (
                [ordered[-1]] if len(ordered) > 1 else []
            ) + ordered[1:-1]
        self.random_state.shuffle(ordered[2:])

        for program in ordered:
            if len(contexts) >= count:
                break
            if program.id not in seen_ids and program.solution not in seen_solutions:
                contexts.append(program)
                seen_ids.add(program.id)
                seen_solutions.add(program.solution)

        for program in remaining:
            if len(contexts) >= count:
                break
            if program.id not in seen_ids:
                contexts.append(program)
                seen_ids.add(program.id)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END