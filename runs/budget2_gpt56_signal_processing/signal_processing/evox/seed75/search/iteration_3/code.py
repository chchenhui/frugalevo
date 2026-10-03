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
    """Small-population elite search with controlled parent and context rotation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.last_added_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = iteration
        if not isinstance(current_iteration, int) or isinstance(current_iteration, bool):
            current_iteration = program.iteration_found
        if isinstance(current_iteration, int) and not isinstance(current_iteration, bool):
            self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                improvement = child_score - parent_score
                threshold = max(0.01, abs(parent_score) * 0.01)
                if improvement > threshold:
                    self.parent_wins[program.parent_id] = (
                        self.parent_wins.get(program.parent_id, 0) + 1
                    )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen == float("-inf"):
                self.best_seen = score
                self.last_meaningful_iteration = self.last_added_iteration
            else:
                threshold = max(0.01, abs(self.best_seen) * 0.01)
                if score > self.best_seen + threshold:
                    self.last_meaningful_iteration = self.last_added_iteration
                self.best_seen = max(self.best_seen, score)

        if isinstance(iteration, int) and not isinstance(iteration, bool):
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
        population_size = len(scored)

        # The population is already near its best observed score, so primarily
        # mutate strong candidates while rotating among the top quarter.
        elite_size = min(population_size, max(4, int(math.ceil(population_size * 0.30))))
        elite = scored[:elite_size]
        best_score = elite[0][0]

        weights: List[float] = []
        for score, program in elite:
            quality = math.exp((score - best_score) * 30.0)
            novelty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            proven = 1.0 + 0.35 * self.parent_wins.get(program.id, 0)
            weights.append(quality * novelty * proven)

        parent = self.random_state.choices(
            [program for _, program in elite], weights=weights, k=1
        )[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        prior_label_uses = sum(
            1
            for candidate in self.programs.values()
            if candidate.parent_info
            and candidate.parent_info[1] == parent.id
            and candidate.parent_info[0]
        )

        # A plateau near the frontier is more likely to benefit from a focused
        # refinement, with occasional divergence reserved for a deeper stall.
        if stalled >= 8 and prior_label_uses == 0:
            if self.random_state.random() < 0.22:
                return {self.REFINE_LABEL: parent}, {}
        if stalled >= 12 and prior_label_uses < 2:
            if self.random_state.random() < 0.16:
                return {self.DIVERGE_LABEL: parent}, {}

        limit = max(0, num_context_programs or 0)
        if limit == 0:
            return {"": parent}, {"": []}

        available = [(score, p) for score, p in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        used_ids = set()

        # Include close high-quality alternatives first. Distinct score levels
        # make the prompt less redundant than repeated near-identical elites.
        for score, candidate in available:
            if len(contexts) >= min(limit, 2):
                break
            if all(abs(score - self._score(existing)) > 0.002 for existing in contexts):
                contexts.append(candidate)
                used_ids.add(candidate.id)

        # Add one strong but less obvious alternative from the upper half.
        upper_half = available[:max(1, len(available) // 2)]
        while len(contexts) < min(limit, 3):
            choices = [p for _, p in upper_half if p.id not in used_ids]
            if not choices:
                break
            least_used = min(self.context_uses.get(p.id, 0) for p in choices)
            choices = [p for p in choices if self.context_uses.get(p.id, 0) == least_used]
            chosen = self.random_state.choice(choices)
            contexts.append(chosen)
            used_ids.add(chosen.id)

        # Fill any remaining slots with underused candidates, retaining some
        # exploration without allowing weak programs to dominate the prompt.
        remaining = [p for _, p in available if p.id not in used_ids]
        while len(contexts) < limit and remaining:
            least_used = min(self.context_uses.get(p.id, 0) for p in remaining)
            choices = [p for p in remaining if self.context_uses.get(p.id, 0) == least_used]
            chosen = self.random_state.choice(choices)
            contexts.append(chosen)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END