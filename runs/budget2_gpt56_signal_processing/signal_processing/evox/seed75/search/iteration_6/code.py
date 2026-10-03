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
    """Elite-biased search with parent rotation and focused context examples."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.parent_gain: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        metrics = getattr(program, "metrics", None)
        if not isinstance(metrics, dict):
            return None
        value = metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = iteration
        if not isinstance(current_iteration, int):
            found = getattr(program, "iteration_found", 0)
            current_iteration = found if isinstance(found, int) else 0
        self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_gain[program.parent_id] = (
                    self.parent_gain.get(program.parent_id, 0.0) + gain
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        parent_info = getattr(program, "parent_info", None)
        if (
            isinstance(parent_info, tuple)
            and len(parent_info) >= 1
            and isinstance(parent_info[0], str)
            and parent_info[0]
        ):
            self.label_uses[parent_info[0]] = self.label_uses.get(parent_info[0], 0) + 1

        score = self._score(program)
        if score is not None:
            meaningful_delta = min(0.01, max(0.0, abs(self.best_seen)) * 0.01)
            if self.best_seen == float("-inf") or score > self.best_seen + meaningful_delta:
                self.best_seen = score
                self.last_meaningful_iteration = current_iteration
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
        scored = [
            (score, program)
            for program in self.programs.values()
            for score in [self._score(program)]
            if score is not None
        ]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        total = len(scored)

        # The population has already converged near its best score. Search among
        # strong candidates, but rotate enough to avoid repeatedly cloning 0.6712.
        pool_size = min(total, max(8, int(math.ceil(total * 0.35))))
        pool = scored[:pool_size]
        best_score = pool[0][0]

        weights: List[float] = []
        for score, program in pool:
            quality = math.exp(max(-5.0, (score - best_score) * 35.0))
            reuse_penalty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            historical_gain = max(0.0, self.parent_gain.get(program.id, 0.0))
            gain_bonus = 1.0 + min(0.75, historical_gain * 20.0)
            weights.append(quality * reuse_penalty * gain_bonus)

        parent = self.random_state.choices(
            [program for _, program in pool], weights=weights, k=1
        )[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        parent_label_count = sum(
            1
            for candidate in self.programs.values()
            if candidate.parent_info
            and len(candidate.parent_info) >= 2
            and candidate.parent_info[1] == parent.id
            and candidate.parent_info[0]
        )

        # Labels are deliberately rare: prior evidence indicates normal
        # elite/context mutations are more reliable than repeated divergence.
        if stalled >= 12 and parent_label_count == 0 and self.random_state.random() < 0.10:
            return {self.REFINE_LABEL: parent}, {}

        limit = max(0, num_context_programs or 0)
        if limit == 0:
            return {"": parent}, {"": []}

        candidates = [p for _, p in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # First show close-to-best alternatives: these contain useful details
        # without distracting the model with clearly inferior solutions.
        elite_contexts = candidates[: min(len(candidates), max(10, limit * 3))]
        while elite_contexts and len(contexts) < min(limit, 3):
            least_used = min(self.context_uses.get(p.id, 0) for p in elite_contexts)
            choices = [
                p for p in elite_contexts
                if self.context_uses.get(p.id, 0) == least_used
                and p.id not in {c.id for c in contexts}
            ]
            if not choices:
                break
            contexts.append(self.random_state.choice(choices))

        # A single upper-mid example supplies a different implementation
        # perspective while avoiding low-quality noise.
        if len(contexts) < limit:
            start = min(len(candidates), max(10, total // 4))
            end = min(len(candidates), max(start + 1, total // 2))
            alternatives = [
                p for p in candidates[start:end]
                if p.id not in {c.id for c in contexts}
            ]
            if alternatives:
                contexts.append(self.random_state.choice(alternatives))

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END