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
    """Adaptive score-aware evolutionary search database."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.add_count = 0
        self.best_numeric_score: Optional[float] = None
        self.last_improvement_add_count = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_wins: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Add a program and update persistent search-progress statistics."""
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.add_count += 1
        score = self._score(program)

        # A meaningful gain must exceed either 0.01 absolute or 1% relative.
        if score is not None:
            if self.best_numeric_score is None:
                self.best_numeric_score = score
                self.last_improvement_add_count = self.add_count
            else:
                gain = score - self.best_numeric_score
                relative_gain = gain / max(abs(self.best_numeric_score), 1e-9)
                if gain > 0.01 or relative_gain > 0.01:
                    self.best_numeric_score = score
                    self.last_improvement_add_count = self.add_count
                elif score > self.best_numeric_score:
                    self.best_numeric_score = score

        # Track which parents have actually been productive, using only
        # information already present on the generated child.
        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                if gain > 0.01 or gain / max(abs(parent_score), 1e-9) > 0.01:
                    self.parent_wins[parent_id] = self.parent_wins.get(parent_id, 0) + 1

        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        context_count = max(0, num_context_programs or 0)
        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]

        if not numeric:
            parent = self.random_state.choice(candidates)
            contexts = [p for p in candidates if p.id != parent.id]
            self.random_state.shuffle(contexts)
            return {"": parent}, {"": contexts[:context_count]}

        numeric.sort(key=lambda item: item[1], reverse=True)
        # Strong parents are preferred, but keep a broad top half available so
        # repeated copies of the incumbent do not monopolize all mutations.
        elite_size = max(2, (len(numeric) + 1) // 2)
        elite = numeric[:elite_size]

        weights: List[float] = []
        for rank, (program, _) in enumerate(elite):
            usage = self.parent_uses.get(program.id, 0)
            wins = self.parent_wins.get(program.id, 0)
            win_rate = wins / max(1, usage)
            score_weight = 1.0 + 4.0 * (elite_size - rank) / elite_size
            diversity_weight = 1.0 + 2.0 / (1.0 + usage)
            weights.append(score_weight * diversity_weight * (1.0 + win_rate))

        parent = self.random_state.choices(
            [program for program, _ in elite], weights=weights, k=1
        )[0]

        stagnation = self.add_count - self.last_improvement_add_count

        # During a real plateau, periodically request a clean conceptual jump
        # from a strong but not necessarily identical incumbent.
        if stagnation >= 12 and self.add_count % 5 == 0:
            alternatives = [p for p, _ in elite if p.id != parent.id]
            if alternatives:
                parent = self.random_state.choice(alternatives)
            return {self.DIVERGE_LABEL: parent}, {}

        available = [p for p, _ in numeric if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Give the model contrasting evidence: one high-quality reference, one
        # middle-ranked alternative, and one lower-ranked approach when present.
        if available and context_count:
            bands = [
                available[:max(1, len(available) // 4)],
                available[len(available) // 4:max(2, len(available) // 2)],
                available[max(2, len(available) // 2):],
            ]
            for band in bands:
                if band and len(contexts) < context_count:
                    contexts.append(self.random_state.choice(band))

        used_ids = {parent.id}
        used_ids.update(p.id for p in contexts)
        remainder = [p for p in available if p.id not in used_ids]
        self.random_state.shuffle(remainder)
        contexts.extend(remainder[:max(0, context_count - len(contexts))])

        return {"": parent}, {"": contexts[:context_count]}


# EVOLVE-BLOCK-END