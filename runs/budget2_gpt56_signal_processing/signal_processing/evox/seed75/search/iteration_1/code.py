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
    """Adaptive elite search with novelty-aware parent and context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.last_added_iteration = 0

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

        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_added_iteration = max(self.last_added_iteration, current_iteration)

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            threshold = min(0.01, max(0.01, abs(self.best_seen)) * 0.01)
            if self.best_seen == float("-inf") or score > self.best_seen + threshold:
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
        scored = [(s, p) for s, p in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[0], reverse=True)
        count = len(scored)
        elite_count = min(count, max(3, int(math.ceil(count * 0.4))))
        elite = scored[:elite_count]

        # Favor good programs, but deliberately rotate among less-used elites.
        weights = []
        best_score = elite[0][0]
        for score, program in elite:
            quality = math.exp((score - best_score) * 20.0)
            novelty = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            weights.append(quality * novelty)
        parent = self.random_state.choices([p for _, p in elite], weights=weights, k=1)[0]

        stalled = self.last_added_iteration - self.last_meaningful_iteration
        prior_labels = sum(
            1 for p in self.programs.values()
            if p.parent_info and p.parent_info[1] == parent.id and p.parent_info[0]
        )

        # Labels are reserved for genuine plateaus and are not repeatedly aimed
        # at the same candidate.
        if stalled >= 7 and prior_labels < 2 and self.random_state.random() < 0.28:
            return {self.DIVERGE_LABEL: parent}, {}
        if stalled >= 4 and prior_labels < 2 and self.random_state.random() < 0.20:
            return {self.REFINE_LABEL: parent}, {}

        limit = max(0, num_context_programs or 0)
        available = [p for _, p in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model both near-best evidence and one contrasting perspective.
        bands = [
            available[:max(1, len(available) // 3)],
            available[len(available) // 3:max(2, 2 * len(available) // 3)],
            available[max(2, 2 * len(available) // 3):],
        ]
        for band in bands:
            if len(contexts) >= limit or not band:
                continue
            least_used = min(self.context_uses.get(p.id, 0) for p in band)
            choices = [p for p in band if self.context_uses.get(p.id, 0) == least_used]
            contexts.append(self.random_state.choice(choices))

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, limit - len(contexts))])

        return {"": parent}, {"": contexts[:limit]}


# EVOLVE-BLOCK-END