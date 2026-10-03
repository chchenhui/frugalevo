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
    """Adaptive score-aware search with parent-use balancing."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.parent_uses: Dict[str, int] = {}
        self.parent_best_child: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program
        step = iteration if isinstance(iteration, int) else program.iteration_found

        score = self._score(program)
        if score is not None:
            previous_best = self.best_seen
            if score > self.best_seen:
                self.best_seen = score
                improvement = score - previous_best
                threshold = min(0.01, max(0.0, previous_best) * 0.01)
                if previous_best == float("-inf") or improvement > threshold:
                    self.last_meaningful_iteration = step

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            if score is not None:
                old = self.parent_best_child.get(program.parent_id, float("-inf"))
                self.parent_best_child[program.parent_id] = max(old, score)

        if isinstance(program.parent_info, tuple) and program.parent_info:
            label = program.parent_info[0]
            if label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        count = max(1, len(scored))
        current_iteration = max(
            [p.iteration_found for p, _ in scored] + [getattr(self, "last_iteration", 0)]
        )
        stalled = current_iteration - self.last_meaningful_iteration

        # Favor strong or previously productive parents, while penalizing parents
        # that have already been repeatedly used.
        pool = scored[:max(3, int(math.ceil(count * (0.65 if stalled < 6 else 0.85))))]
        weights = []
        for rank, (program, score) in enumerate(pool):
            child_best = self.parent_best_child.get(program.id, score)
            productivity = max(0.0, child_best - score)
            reuse_penalty = 1.0 + self.parent_uses.get(program.id, 0)
            weights.append(((len(pool) - rank) + 1.0 + productivity * 25.0) / reuse_penalty)
        parent = self.random_state.choices([p for p, _ in pool], weights=weights, k=1)[0]

        label = ""
        if stalled >= 8 and self.label_uses.get(self.DIVERGE_LABEL, 0) < 2:
            # A plateau merits a deliberately different attempt from a good,
            # but not necessarily most overused, candidate.
            label = self.DIVERGE_LABEL
        elif stalled >= 4 and self.label_uses.get(self.REFINE_LABEL, 0) < 2:
            label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"": []}

        wanted = max(0, num_context_programs or 0)
        remaining = [p for p, _ in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Keep one elite reference, then add randomly chosen alternatives.
        elite = remaining[:max(1, min(5, len(remaining)))]
        if elite and wanted:
            contexts.append(self.random_state.choice(elite))

        remaining = [p for p in remaining if p.id not in {x.id for x in contexts}]
        while remaining and len(contexts) < wanted:
            index = self.random_state.randrange(len(remaining))
            contexts.append(remaining.pop(index))

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END