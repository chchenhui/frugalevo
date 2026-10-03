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
    """Elite-focused search with occasional targeted plateau escapes."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current = iteration
        if current is None:
            current = getattr(program, "iteration_found", 0)
        if isinstance(current, int):
            self.last_added_iteration = max(self.last_added_iteration, current)
            self.last_iteration = max(self.last_iteration, current)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            meaningful = max(0.01, abs(self.best_seen) * 0.01) if self.best_seen != float("-inf") else 0.0
            if self.best_seen == float("-inf") or score > self.best_seen + meaningful:
                self.best_seen = score
                self.last_meaningful_iteration = self.last_added_iteration
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
        scored = [(score, p) for score, p in scored if score is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score = scored[0][0]
        elite = [(score, p) for score, p in scored if score >= best_score - 0.012]
        if len(elite) < 3:
            elite = scored[:min(6, len(scored))]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        label_attempts = [
            p for p in self.programs.values()
            if p.parent_info and p.parent_info[0]
        ]

        # During a real plateau, explicitly try one careful refinement or a
        # different direction before returning to normal elite recombination.
        if stalled >= 6 and len(label_attempts) < 4:
            label = self.REFINE_LABEL if len(label_attempts) % 2 == 0 else self.DIVERGE_LABEL
            candidates = sorted(
                elite,
                key=lambda item: (
                    sum(
                        1 for child in label_attempts
                        if child.parent_info and child.parent_info[1] == item[1].id
                    ),
                    -item[0],
                ),
            )
            return {label: candidates[0][1]}, {}

        weights = []
        for score, program in elite:
            quality = math.exp((score - best_score) * 45.0)
            reuse_penalty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            weights.append(quality * reuse_penalty)
        parent = self.random_state.choices(
            [program for _, program in elite], weights=weights, k=1
        )[0]

        limit = max(0, num_context_programs or 0)
        available = [(score, p) for score, p in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Show strong alternatives first, then one contrasting viable approach.
        top_pool = available[:min(12, len(available))]
        while top_pool and len(contexts) < min(3, limit):
            least_used = min(self.context_uses.get(p.id, 0) for _, p in top_pool)
            choices = [item for item in top_pool if self.context_uses.get(item[1].id, 0) == least_used]
            chosen = self.random_state.choice(choices)
            contexts.append(chosen[1])
            top_pool = [item for item in top_pool if item[1].id != chosen[1].id]

        remaining = [
            p for _, p in available
            if p.id not in {context.id for context in contexts}
        ]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, limit - len(contexts))])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END