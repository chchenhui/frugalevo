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
    """Adaptive elite search with balanced parent reuse and contrasting context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.parent_uses: Dict[str, int] = {}
        self.parent_gain: Dict[str, float] = {}
        self.parent_children: Dict[str, int] = {}
        self.label_uses: Dict[str, int] = {}
        self.last_iteration_seen = 0

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
        if isinstance(step, int):
            self.last_iteration_seen = max(self.last_iteration_seen, step)

        score = self._score(program)
        previous_best = self.best_seen
        if score is not None and score > self.best_seen:
            self.best_seen = score
            # A gain is meaningful only when it clears both requested criteria.
            threshold = max(0.01, max(0.0, previous_best) * 0.01)
            if previous_best == float("-inf") or score - previous_best > threshold:
                self.last_meaningful_iteration = step if isinstance(step, int) else self.last_iteration_seen

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1
            self.parent_children[parent_id] = self.parent_children.get(parent_id, 0) + 1

            parent = self.get(parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                old_gain = self.parent_gain.get(parent_id, float("-inf"))
                self.parent_gain[parent_id] = max(old_gain, gain)

        if isinstance(program.parent_info, tuple) and len(program.parent_info) > 0:
            label = program.parent_info[0]
            if label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored: List[Tuple[EvolvedProgram, float]] = []
        for program in self.programs.values():
            score = self._score(program)
            if score is not None:
                scored.append((program, score))

        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        total = len(scored)
        current_iteration = max(
            self.last_iteration_seen,
            getattr(self, "last_iteration", 0),
            max(program.iteration_found for program, _ in scored),
        )
        stalled = current_iteration - self.last_meaningful_iteration

        # The useful search region is broad enough to preserve alternate filter
        # designs, but strongly favors the current high-quality frontier.
        elite_size = min(total, max(6, int(math.ceil(total * 0.40))))
        pool = scored[:elite_size]

        weights: List[float] = []
        for rank, (program, score) in enumerate(pool):
            rank_strength = (elite_size - rank) / float(elite_size)
            gain = max(0.0, self.parent_gain.get(program.id, 0.0))
            uses = self.parent_uses.get(program.id, 0)
            children = self.parent_children.get(program.id, 0)

            # Productive parents are valuable, but a previously unused elite
            # remains competitive so the search does not collapse to one tree edge.
            productivity = 1.0 + min(1.5, gain * 35.0)
            reuse_penalty = 1.0 + 0.45 * uses + 0.15 * children
            weights.append((0.7 + 2.3 * rank_strength) * productivity / reuse_penalty)

        parent = self.random_state.choices(
            [program for program, _ in pool], weights=weights, k=1
        )[0]

        # Labels are reserved for a genuine plateau. Refinement is attempted
        # first because this population already has a strong, converged frontier.
        if stalled >= 8 and self.label_uses.get(self.REFINE_LABEL, 0) < 1:
            return {self.REFINE_LABEL: parent}, {"": []}
        if stalled >= 13 and self.label_uses.get(self.DIVERGE_LABEL, 0) < 1:
            explore_pool = scored[:min(total, max(10, int(total * 0.65)))]
            parent = self.random_state.choice([program for program, _ in explore_pool])
            return {self.DIVERGE_LABEL: parent}, {"": []}

        wanted = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def add_context(candidates: List[EvolvedProgram]) -> None:
            available = [p for p in candidates if p.id not in used_ids]
            if available and len(contexts) < wanted:
                chosen = self.random_state.choice(available)
                contexts.append(chosen)
                used_ids.add(chosen.id)

        # Give the model a strong reference, a nearby competing solution, and
        # a contrasting older approach rather than a collection of random code.
        add_context([program for program, _ in scored[:4]])
        add_context([program for program, _ in scored[4:min(total, 14)]])

        if total > 8:
            middle_start = max(0, int(total * 0.35))
            middle_end = max(middle_start + 1, int(total * 0.70))
            add_context([program for program, _ in scored[middle_start:middle_end]])

        remaining = [program for program, _ in scored if program.id not in used_ids]
        while remaining and len(contexts) < wanted:
            chosen = self.random_state.choice(remaining)
            contexts.append(chosen)
            used_ids.add(chosen.id)
            remaining = [p for p in remaining if p.id != chosen.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END