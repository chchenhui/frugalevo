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
    """Adaptive elite-and-diversity search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen_score = float("-inf")
        self.last_meaningful_improvement = -1
        self.add_count = 0
        self.parent_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score") if isinstance(program.metrics, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any) -> str:
        self.programs[program.id] = program
        self.add_count += 1

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        current_iteration = iteration if iteration is not None else program.iteration_found
        if score is not None:
            if self.best_seen_score == float("-inf"):
                self.best_seen_score = score
                self.last_meaningful_improvement = current_iteration
            elif score > self.best_seen_score:
                improvement = score - self.best_seen_score
                relative = improvement / max(abs(self.best_seen_score), 1e-9)
                self.best_seen_score = score
                if improvement > 0.01 or relative > 0.01:
                    self.last_meaningful_improvement = current_iteration

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = [p for p in self.programs.values() if self._score(p) is not None]
        if not candidates:
            candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        ranked = sorted(candidates, key=lambda p: self._score(p) or float("-inf"), reverse=True)
        n = len(ranked)
        elite = ranked[:max(1, (n + 3) // 4)]
        middle = ranked[n // 4:max(n // 2, 1)]
        lower = ranked[max(n // 2, 1):]

        current_iteration = max(
            self.last_iteration,
            max((p.iteration_found for p in candidates), default=0),
        )
        stalled = current_iteration - self.last_meaningful_improvement

        def least_used(pool: List[EvolvedProgram]) -> EvolvedProgram:
            minimum = min(self.parent_uses.get(p.id, 0) for p in pool)
            return self.random_state.choice(
                [p for p in pool if self.parent_uses.get(p.id, 0) == minimum]
            )

        # During a genuine plateau, alternate targeted refinement and a change
        # of direction rather than repeatedly presenting the same best program.
        if stalled >= 8:
            refine_count = self.label_uses.get(self.REFINE_LABEL, 0)
            diverge_count = self.label_uses.get(self.DIVERGE_LABEL, 0)
            if refine_count <= diverge_count:
                return {self.REFINE_LABEL: least_used(elite)}, {}
            explore_pool = middle or lower or ranked
            return {self.DIVERGE_LABEL: least_used(explore_pool)}, {}

        # Mostly exploit good candidates, but select underused parents to avoid
        # repeatedly mutating one of the identical top-score solutions.
        parent_pool = elite if self.random_state.random() < 0.70 else (middle or ranked)
        parent = least_used(parent_pool)

        limit = max(0, num_context_programs or 0)
        context: List[EvolvedProgram] = []
        pools = [elite, middle, lower]
        for pool in pools:
            choices = [p for p in pool if p.id != parent.id and p.id not in {x.id for x in context}]
            if choices and len(context) < limit:
                self.random_state.shuffle(choices)
                context.append(choices[0])

        remaining = [p for p in ranked if p.id != parent.id and p.id not in {x.id for x in context}]
        self.random_state.shuffle(remaining)
        context.extend(remaining[:max(0, limit - len(context))])

        return {"": parent}, {"": context[:limit]}


# EVOLVE-BLOCK-END