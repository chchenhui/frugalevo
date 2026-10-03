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
    """Small adaptive population search with mixed-score context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen: Optional[float] = None
        self.best_program_id: Optional[str] = None
        self.last_iteration = 0
        self.last_meaningful_improvement_iteration = 0
        self.parent_usage: Dict[str, int] = {}
        self.context_usage: Dict[str, int] = {}
        self.parent_best_gain: Dict[str, float] = {}
        self.parent_total_gain: Dict[str, float] = {}
        self.parent_children: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            score = float(value)
            if math.isfinite(score):
                return score
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        self.programs[program.id] = program

        current_iteration = (
            iteration if iteration is not None else getattr(program, "iteration_found", 0)
        )
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        # Record actual historical choices. This also reconstructs useful state
        # when programs are replayed from a saved database.
        if program.parent_id:
            parent_id = program.parent_id
            self.parent_usage[parent_id] = self.parent_usage.get(parent_id, 0) + 1
            self.parent_children[parent_id] = self.parent_children.get(parent_id, 0) + 1

            parent = self.get(parent_id)
            child_score = self._score(program)
            parent_score = self._score(parent) if parent is not None else None
            if child_score is not None and parent_score is not None:
                gain = child_score - parent_score
                self.parent_total_gain[parent_id] = (
                    self.parent_total_gain.get(parent_id, 0.0) + gain
                )
                self.parent_best_gain[parent_id] = max(
                    self.parent_best_gain.get(parent_id, gain), gain
                )

        for context_id in program.other_context_ids or []:
            self.context_usage[context_id] = self.context_usage.get(context_id, 0) + 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen is None:
                self.best_score_seen = score
                self.best_program_id = program.id
                self.last_meaningful_improvement_iteration = self.last_iteration
            elif score > self.best_score_seen:
                old_best = self.best_score_seen
                gain = score - old_best
                relative_gain = gain / max(abs(old_best), 1e-12)
                self.best_score_seen = score
                self.best_program_id = program.id
                if gain > 0.01 or relative_gain > 0.01:
                    self.last_meaningful_improvement_iteration = self.last_iteration

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        logger.debug("Added program %s to the evolve database", program.id)
        return program.id

    def _choose_weighted(
        self, programs: List[EvolvedProgram], weights: List[float]
    ) -> EvolvedProgram:
        return self.random_state.choices(programs, weights=weights, k=1)[0]

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        ranked = [(self._score(p), p) for p in candidates]
        ranked = [(score, p) for score, p in ranked if score is not None]
        ranked.sort(key=lambda item: item[0], reverse=True)

        if not ranked:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        # The current population is tightly clustered near the top. Prefer this
        # promising band, but occasionally revisit a mid-tier idea: historically,
        # a non-best parent plus varied context produced the best child.
        n = len(ranked)
        top_end = max(2, int(math.ceil(n * 0.65)))
        mid_start = max(1, int(math.floor(n * 0.25)))
        mid_end = max(mid_start + 1, int(math.ceil(n * 0.80)))

        if n >= 6 and self.random_state.random() < 0.25:
            parent_pool = ranked[mid_start:mid_end]
        else:
            parent_pool = ranked[:top_end]

        pool_programs = [p for _, p in parent_pool]
        low_score = parent_pool[-1][0]
        high_score = parent_pool[0][0]
        parent_weights: List[float] = []

        for score, program in parent_pool:
            quality = 1.0 + (score - low_score) / max(high_score - low_score, 1e-9)
            children = self.parent_children.get(program.id, 0)
            average_gain = self.parent_total_gain.get(program.id, 0.0) / max(children, 1)
            best_gain = self.parent_best_gain.get(program.id, 0.0)

            # Reward parents that have already generated improvements, while
            # avoiding repeated mutations of the same parent.
            offspring_signal = max(0.0, average_gain) * 120.0 + max(0.0, best_gain) * 80.0
            reuse_penalty = 1.0 + 0.65 * self.parent_usage.get(program.id, 0)
            parent_weights.append((quality + offspring_signal) / reuse_penalty)

        parent = self._choose_weighted(pool_programs, parent_weights)

        context_count = max(0, num_context_programs or 0)
        available = [p for _, p in ranked if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        # Supply contrasting score bands rather than four near-duplicates.
        # This preserves useful strong solutions while exposing alternate,
        # lower-scoring approaches that may contain a missing idea.
        if context_count and available:
            bands = [
                available[:max(1, len(available) // 4)],
                available[max(1, len(available) // 4):max(2, len(available) // 2)],
                available[max(2, len(available) // 2):max(3, 3 * len(available) // 4)],
                available[max(3, 3 * len(available) // 4):],
            ]

            for band in bands:
                if len(contexts) >= context_count or not band:
                    continue
                choices = [p for p in band if p.id not in {c.id for c in contexts}]
                if not choices:
                    continue
                weights = [
                    1.0 / (1.0 + self.context_usage.get(p.id, 0))
                    for p in choices
                ]
                contexts.append(self._choose_weighted(choices, weights))

            remaining = [p for p in available if p.id not in {c.id for c in contexts}]
            while len(contexts) < context_count and remaining:
                weights = [
                    1.0 / (1.0 + self.context_usage.get(p.id, 0))
                    for p in remaining
                ]
                chosen = self._choose_weighted(remaining, weights)
                contexts.append(chosen)
                remaining = [p for p in remaining if p.id != chosen.id]

        # Do not force divergence/refinement labels during this plateau: prior
        # labeled attempts underperformed, while mixed-context mutations helped.
        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END