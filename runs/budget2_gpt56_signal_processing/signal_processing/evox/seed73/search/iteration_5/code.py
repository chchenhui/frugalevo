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
    """Late-stage elite search with occasional targeted escape attempts."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_observed_score = -float("inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gains: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.last_label_iteration = -1000000
        self.last_seen_iteration = 0

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
        score = self._score(program)
        current_iteration = (
            iteration if isinstance(iteration, int) else program.iteration_found
        )
        if isinstance(current_iteration, int):
            self.last_seen_iteration = max(self.last_seen_iteration, current_iteration)

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_gains[program.parent_id] = (
                    self.parent_gains.get(program.parent_id, 0.0) + gain
                )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        if program.parent_info and program.parent_info[0]:
            if isinstance(current_iteration, int):
                self.last_label_iteration = max(
                    self.last_label_iteration, current_iteration
                )

        if score is not None:
            threshold = max(0.01, 0.01 * abs(self.best_observed_score))
            if not math.isfinite(self.best_observed_score) or score > self.best_observed_score + threshold:
                self.best_observed_score = score
                self.stagnation_steps = 0
            else:
                self.best_observed_score = max(self.best_observed_score, score)
                self.stagnation_steps += 1

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)
        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        valid = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in valid if s is not None]

        if not valid:
            if not candidates:
                raise ValueError("No candidates available for sampling")
            return {"": self.random_state.choice(candidates)}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best_score = valid[0][1]

        # Late in search, focus on the tightly clustered high-quality solutions
        # rather than spending mutations on the broad lower half of the pool.
        band = [(p, s) for p, s in valid if s >= best_score - 0.025]
        if len(band) < 3:
            band = valid[:min(len(valid), 6)]

        parent_choices = [p for p, _ in band]
        weights: List[float] = []
        for program, score in band:
            uses = self.parent_uses.get(program.id, 0)
            mean_gain = self.parent_gains.get(program.id, 0.0) / max(1, uses)
            score_weight = 1.0 + 8.0 * max(0.0, score - (best_score - 0.025))
            novelty_weight = 1.0 / (1.0 + 0.45 * uses)
            gain_weight = max(0.5, min(1.8, 1.0 + 12.0 * mean_gain))
            weights.append(max(0.05, score_weight * novelty_weight * gain_weight))

        parent = self.random_state.choices(parent_choices, weights=weights, k=1)[0]
        label = ""

        # A plateau after several near-best attempts is evidence that small
        # variations are exhausted. Use one infrequent, focused escape attempt.
        label_is_recent = (
            self.last_seen_iteration - self.last_label_iteration < 5
        )
        if self.stagnation_steps >= 7 and not label_is_recent and len(band) > 1:
            alternatives = [p for p, _ in band if p.id != valid[0][0].id]
            if alternatives:
                parent = self.random_state.choice(alternatives)
                label = self.DIVERGE_LABEL
                return {label: parent}, {"": []}

        context_count = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        remaining = [p for p, _ in valid if p.id != parent.id]

        # Give the model several strong, distinct references. The best is
        # retained as an anchor, while low-use elite programs provide variation.
        if context_count and remaining:
            contexts.append(remaining[0])

        elite_contexts = [
            p for p, s in valid
            if p.id != parent.id
            and p.id not in {c.id for c in contexts}
            and s >= best_score - 0.035
        ]

        while elite_contexts and len(contexts) < context_count:
            context_weights = [
                1.0 / (1.0 + self.context_uses.get(p.id, 0))
                for p in elite_contexts
            ]
            chosen = self.random_state.choices(
                elite_contexts, weights=context_weights, k=1
            )[0]
            contexts.append(chosen)
            elite_contexts = [p for p in elite_contexts if p.id != chosen.id]

        while remaining and len(contexts) < context_count:
            pool = [p for p in remaining if p.id not in {c.id for c in contexts}]
            if not pool:
                break
            chosen = self.random_state.choice(pool)
            contexts.append(chosen)

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END