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
    """Adaptive rank-based search with stagnation-aware exploration."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.stagnation = 0
        self.parent_uses: Dict[str, int] = {}
        self.context_uses: Dict[str, int] = {}
        self.parent_gain_sum: Dict[str, float] = {}
        self.parent_gain_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)

        parent_id = getattr(program, "parent_id", "")
        if parent_id:
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            parent = self.get(parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                self.parent_gain_sum[parent_id] = (
                    self.parent_gain_sum.get(parent_id, 0.0) + child_score - parent_score
                )
                self.parent_gain_count[parent_id] = (
                    self.parent_gain_count.get(parent_id, 0) + 1
                )

        for context_id in getattr(program, "other_context_ids", []) or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_seen == float("-inf"):
                self.best_seen = score
            else:
                improvement = score - self.best_seen
                meaningful = improvement > max(0.01, abs(self.best_seen) * 0.01)
                if meaningful:
                    self.best_seen = score
                    self.stagnation = 0
                else:
                    self.stagnation += 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def _weighted_choice(
        self, programs: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        total = sum(weights)
        if total <= 0:
            return self.random_state.choice(programs)
        point = self.random_state.random() * total
        running = 0.0
        for program, weight in zip(programs, weights):
            running += weight
            if running >= point:
                return program
        return programs[-1]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(self._score(p), p) for p in candidates]
        valid = [(score, p) for score, p in scored if score is not None]

        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[0], reverse=True)
        ordered = [p for _, p in valid]
        n = len(ordered)

        # Rank rewards quality, while low reuse and productive parents avoid
        # repeatedly mutating the same plateaued top candidate.
        weights: List[float] = []
        for rank, program in enumerate(ordered):
            quality = ((n - rank) / n) ** 2
            reuse_bonus = 1.0 / (1.0 + self.parent_uses.get(program.id, 0))
            count = self.parent_gain_count.get(program.id, 0)
            mean_gain = self.parent_gain_sum.get(program.id, 0.0) / count if count else 0.0
            gain_bonus = max(-0.25, min(0.5, mean_gain / 2.0))
            weights.append(0.15 + quality + 0.65 * reuse_bonus + gain_bonus)

        # During a plateau, deliberately give viable mid-ranked approaches a
        # chance: the observed population has useful "springboard" parents.
        if self.stagnation >= 8 and n >= 4:
            middle = ordered[max(1, n // 4):max(2, (3 * n) // 4)]
            if middle and self.random_state.random() < 0.40:
                middle_weights = [
                    1.0 / (1.0 + self.parent_uses.get(p.id, 0)) for p in middle
                ]
                parent = self._weighted_choice(middle, middle_weights)
            else:
                parent = self._weighted_choice(ordered, weights)
        else:
            parent = self._weighted_choice(ordered, weights)

        label = ""
        parent_rank = ordered.index(parent)
        if self.stagnation >= 12 and self.random_state.random() < 0.45:
            label = self.REFINE_LABEL if parent_rank < max(1, n // 4) else self.DIVERGE_LABEL
            return {label: parent}, {"": []}

        wanted = max(0, num_context_programs or 0)
        available = [p for p in ordered if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Mix a strong reference with contrasting mid/broad examples.
        pools = [
            available[:max(1, n // 5)],
            available[max(1, n // 4):max(2, (3 * n) // 4)],
            available,
        ]
        for pool in pools:
            if len(contexts) >= wanted:
                break
            choices = [p for p in pool if p.id not in {c.id for c in contexts}]
            if choices:
                use_weights = [
                    1.0 / (1.0 + self.context_uses.get(p.id, 0)) for p in choices
                ]
                contexts.append(self._weighted_choice(choices, use_weights))

        remaining = [p for p in available if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        contexts.extend(remaining[:max(0, wanted - len(contexts))])

        return {"": parent}, {"": contexts[:wanted]}


# EVOLVE-BLOCK-END