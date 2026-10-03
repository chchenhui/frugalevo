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
    """
    Compact adaptive search:
    - exploit diverse high-scoring frontier candidates normally
    - use targeted empty-context divergence when the frontier is stalled
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))

        self.best_observed_score = -float("inf")
        self.stagnation_steps = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.label_gain: Dict[str, float] = {}
        self.last_iteration = getattr(self, "last_iteration", -1)

    def _score(self, program: Optional[EvolvedProgram]) -> Optional[float]:
        if program is None:
            return None
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def _meaningful(self, score: float) -> bool:
        if not math.isfinite(self.best_observed_score):
            return True
        return score > self.best_observed_score + max(
            0.01, 0.01 * abs(self.best_observed_score)
        )

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        score = self._score(program)
        parent = self.get(program.parent_id) if program.parent_id else None
        parent_score = self._score(parent)

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )

        gain = 0.0
        if score is not None and parent_score is not None:
            gain = score - parent_score
            self.parent_gain[program.parent_id] = (
                self.parent_gain.get(program.parent_id, 0.0) + gain
            )

        for context_id in program.other_context_ids or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        label = ""
        if program.parent_info:
            label = program.parent_info[0] or ""
        if label:
            self.label_uses[label] = self.label_uses.get(label, 0) + 1
            self.label_gain[label] = self.label_gain.get(label, 0.0) + gain

        if score is not None:
            if self._meaningful(score):
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

        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        valid = [(p, self._score(p)) for p in candidates]
        valid = [(p, s) for p, s in valid if s is not None]
        if not valid:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        valid.sort(key=lambda item: item[1], reverse=True)
        best_score = valid[0][1]
        worst_score = valid[-1][1]
        span = max(best_score - worst_score, 1e-9)

        # Keep normal mutation on the top third, rather than repeatedly sampling
        # broad mediocre population regions.
        frontier_size = min(len(valid), max(5, (len(valid) + 2) // 3))
        frontier = valid[:frontier_size]

        def parent_weight(program: EvolvedProgram, score: float) -> float:
            quality = (score - worst_score) / span
            uses = self.parent_uses.get(program.id, 0)
            average_gain = self.parent_gain.get(program.id, 0.0) / max(1, uses)
            return max(
                0.05,
                0.4 + 2.2 * quality + 1.0 / (1 + uses)
                + max(-0.2, min(0.5, 15.0 * average_gain)),
            )

        # The observed population is stalled and its best result came from a
        # labeled, empty-context divergence of a near-frontier solution.
        # Use it selectively, targeting underused near-best alternatives rather
        # than always diverging from the current champion.
        if self.stagnation_steps >= 6:
            alternatives = frontier[1:] if len(frontier) > 1 else frontier
            weights = [
                parent_weight(p, s) * (1.0 + 0.5 / (1 + self.parent_uses.get(p.id, 0)))
                for p, s in alternatives
            ]
            parent = self.random_state.choices(
                [p for p, _ in alternatives], weights=weights, k=1
            )[0]
            return {self.DIVERGE_LABEL: parent}, {}

        weights = [parent_weight(p, s) for p, s in frontier]
        parent = self.random_state.choices(
            [p for p, _ in frontier], weights=weights, k=1
        )[0]

        context_count = max(0, num_context_programs or 0)
        if context_count == 0:
            return {"": parent}, {"": []}

        available = [(p, s) for p, s in valid if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Show one best reference plus alternatives from different score bands.
        if available:
            contexts.append(available[0][0])

        targets = [0.92, 0.72, 0.50, 0.30]
        while available and len(contexts) < context_count:
            used_ids = {p.id for p in contexts}
            choices = [(p, s) for p, s in available if p.id not in used_ids]
            if not choices:
                break

            target = targets[(len(contexts) - 1) % len(targets)]
            weights = []
            for program, score in choices:
                normalized = (score - worst_score) / span
                diversity = 1.0 - min(1.0, abs(normalized - target))
                novelty = 1.0 / (1 + self.context_uses.get(program.id, 0))
                weights.append(max(0.05, 0.3 + diversity + novelty))

            contexts.append(
                self.random_state.choices(
                    [p for p, _ in choices], weights=weights, k=1
                )[0]
            )

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END