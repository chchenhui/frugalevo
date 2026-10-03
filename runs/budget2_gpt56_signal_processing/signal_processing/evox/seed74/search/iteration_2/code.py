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
    """Elite-biased search with balanced reuse and diverse supporting context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.parent_uses: Dict[str, int] = {}
        self.parent_best_child: Dict[str, float] = {}
        self.context_uses: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.last_label_iteration = -100

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program
        step = iteration if isinstance(iteration, int) else program.iteration_found
        score = self._score(program)

        if score is not None:
            old_best = self.best_seen
            if score > old_best:
                self.best_seen = score
                meaningful = (
                    old_best == float("-inf")
                    or score - old_best > min(0.01, max(0.0, old_best) * 0.01)
                )
                if meaningful:
                    self.last_meaningful_iteration = step

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )
            if score is not None:
                old_child = self.parent_best_child.get(
                    program.parent_id, float("-inf")
                )
                self.parent_best_child[program.parent_id] = max(old_child, score)

        for context_id in getattr(program, "other_context_ids", []) or []:
            self.context_uses[context_id] = self.context_uses.get(context_id, 0) + 1

        parent_info = getattr(program, "parent_info", None)
        if isinstance(parent_info, tuple) and parent_info and parent_info[0]:
            label = parent_info[0]
            self.label_uses[label] = self.label_uses.get(label, 0) + 1
            self.last_label_iteration = step

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def _pick_balanced(
        self, candidates: List[Tuple[EvolvedProgram, float]]
    ) -> EvolvedProgram:
        weights: List[float] = []
        size = len(candidates)

        for rank, (program, score) in enumerate(candidates):
            child_score = self.parent_best_child.get(program.id, score)
            child_gain = max(0.0, child_score - score)
            rank_weight = float(size - rank)
            reuse_penalty = 1.0 + self.parent_uses.get(program.id, 0)
            weights.append((rank_weight + child_gain * 35.0) / reuse_penalty)

        return self.random_state.choices(
            [program for program, _ in candidates], weights=weights, k=1
        )[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [
            (program, score)
            for program in self.programs.values()
            for score in [self._score(program)]
            if score is not None
        ]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        current_iteration = max(
            [program.iteration_found for program, _ in scored]
            + [getattr(self, "last_iteration", 0)]
        )
        stalled = current_iteration - self.last_meaningful_iteration
        count = len(scored)

        # The current population already has several near-best candidates.
        # Primarily mutate elite programs, but widen slightly on a plateau.
        fraction = 0.40 if stalled < 6 else 0.60
        pool_size = min(count, max(5, int(math.ceil(count * fraction))))
        parent_pool = scored[:pool_size]
        parent = self._pick_balanced(parent_pool)

        # Labels are reserved for a genuine sustained plateau and are not
        # repeatedly applied to every generation.
        label = ""
        label_cooldown = current_iteration - self.last_label_iteration
        if stalled >= 8 and label_cooldown >= 4:
            if self.label_uses.get(self.DIVERGE_LABEL, 0) < 1:
                alternatives = parent_pool[1:] or parent_pool
                parent = self._pick_balanced(alternatives)
                label = self.DIVERGE_LABEL
            elif self.label_uses.get(self.REFINE_LABEL, 0) < 1:
                parent = self._pick_balanced(parent_pool[:min(3, len(parent_pool))])
                label = self.REFINE_LABEL

        if label:
            return {label: parent}, {"": []}

        wanted = max(0, num_context_programs or 0)
        available = [(p, s) for p, s in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Supply one near-elite reference plus candidates from distinct score
        # bands, reducing repeated context combinations and convergence.
        bands = [
            available[:max(1, min(5, len(available)))],
            available[max(1, count // 4):max(2, count // 2)],
            available[max(2, count // 2):max(3, (3 * count) // 4)],
            available[max(3, (3 * count) // 4):],
        ]

        used_ids = set()
        for band in bands:
            choices = [(p, s) for p, s in band if p.id not in used_ids]
            if not choices or len(contexts) >= wanted:
                continue

            weights = [
                1.0 / (1.0 + self.context_uses.get(program.id, 0))
                for program, _ in choices
            ]
            chosen = self.random_state.choices(
                [program for program, _ in choices], weights=weights, k=1
            )[0]
            contexts.append(chosen)
            used_ids.add(chosen.id)

        remaining = [(p, s) for p, s in available if p.id not in used_ids]
        while remaining and len(contexts) < wanted:
            weights = [
                1.0 / (1.0 + self.context_uses.get(program.id, 0))
                for program, _ in remaining
            ]
            chosen = self.random_state.choices(
                [program for program, _ in remaining], weights=weights, k=1
            )[0]
            contexts.append(chosen)
            remaining = [(p, s) for p, s in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END