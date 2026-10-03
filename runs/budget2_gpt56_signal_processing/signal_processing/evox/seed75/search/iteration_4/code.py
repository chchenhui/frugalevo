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
    """Small adaptive elite pool with lineage-aware exploitation and plateau escapes."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.parent_gain_count: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        current_iteration = iteration
        if current_iteration is None:
            current_iteration = getattr(program, "iteration_found", 0)
        if isinstance(current_iteration, int):
            self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_gain_sum[program.parent_id] = (
                    self.parent_gain_sum.get(program.parent_id, 0.0) + gain
                )
                self.parent_gain_count[program.parent_id] = (
                    self.parent_gain_count.get(program.parent_id, 0) + 1
                )

        if program.parent_info and program.parent_info[0]:
            label = program.parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1

        score = self._score(program)
        if score is not None:
            meaningful = max(0.01, abs(self.best_seen) * 0.01) if self.best_seen != float("-inf") else 0.0
            if self.best_seen == float("-inf") or score > self.best_seen + meaningful:
                self.best_seen = score
                self.last_meaningful_iteration = self.last_added_iteration
            else:
                self.best_seen = max(self.best_seen, score)

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(self._score(p), p) for p in self.programs.values()]
        scored = [(score, program) for score, program in scored if score is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        pool_size = min(len(scored), max(6, int(math.ceil(len(scored) * 0.30))))
        elite = scored[:pool_size]
        best_score = elite[0][0]

        weights: List[float] = []
        for score, program in elite:
            uses = self.parent_uses.get(program.id, 0)
            gains = self.parent_gain_count.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / gains if gains else 0.0
            quality = math.exp((score - best_score) * 28.0)
            lineage_value = 1.0 + max(-0.3, min(0.5, mean_gain * 20.0))
            novelty = 1.0 / math.sqrt(1.0 + uses)
            weights.append(max(0.05, quality * lineage_value * novelty))

        parent = self.random_state.choices(
            [program for _, program in elite], weights=weights, k=1
        )[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        # On a genuine plateau, explicitly alternate between polishing the best
        # evidence and requesting a different direction from a strong alternative.
        if stalled >= 7:
            refine_count = self.label_uses.get(self.REFINE_LABEL, 0)
            diverge_count = self.label_uses.get(self.DIVERGE_LABEL, 0)
            if refine_count <= diverge_count:
                return {self.REFINE_LABEL: scored[0][1]}, {}
            alternatives = elite[1:] if len(elite) > 1 else elite
            return {self.DIVERGE_LABEL: self.random_state.choice(alternatives)[1]}, {}

        limit = max(0, num_context_programs or 0)
        available = [program for _, program in scored if program.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Mostly show successful nearby solutions, plus one different score tier
        # to preserve useful alternative implementation ideas.
        top_count = min(len(available), max(2, limit - 1))
        if top_count:
            contexts.extend(self.random_state.sample(
                available[:max(top_count, min(len(available), 8))],
                min(top_count, len(available[:max(top_count, min(len(available), 8))]))
            ))

        if len(contexts) < limit and len(available) > top_count:
            middle = available[top_count:max(top_count + 12, len(available))]
            if middle:
                candidate = self.random_state.choice(middle)
                if candidate.id not in {p.id for p in contexts}:
                    contexts.append(candidate)

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:limit - len(contexts)])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END