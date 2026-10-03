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
    """Frontier-focused search with parent reuse control and contrasting context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.last_iteration_seen = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_best_gain: Dict[str, float] = {}
        self.parent_context_uses: Dict[Tuple[str, str], int] = {}
        self.label_uses: Dict[str, int] = {}

    def _score(self, program: Optional[EvolvedProgram]) -> Optional[float]:
        if program is None:
            return None
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
            if previous_best == float("-inf"):
                self.last_meaningful_iteration = self.last_iteration_seen
            else:
                gain = score - previous_best
                # A meaningful gain clears either the relative or absolute test.
                if gain > 0.01 or gain > max(0.0, previous_best) * 0.01:
                    self.last_meaningful_iteration = self.last_iteration_seen

        if program.parent_id:
            parent_id = program.parent_id
            self.parent_uses[parent_id] = self.parent_uses.get(parent_id, 0) + 1

            parent_score = self._score(self.get(parent_id))
            if score is not None and parent_score is not None:
                gain = score - parent_score
                old_gain = self.parent_best_gain.get(parent_id, float("-inf"))
                self.parent_best_gain[parent_id] = max(old_gain, gain)

            for context_id in program.other_context_ids:
                if isinstance(context_id, str):
                    key = (parent_id, context_id)
                    self.parent_context_uses[key] = (
                        self.parent_context_uses.get(key, 0) + 1
                    )

        if isinstance(program.parent_info, tuple) and program.parent_info:
            label = program.parent_info[0]
            if isinstance(label, str) and label:
                self.label_uses[label] = self.label_uses.get(label, 0) + 1

        if isinstance(iteration, int):
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
        wanted = max(0, num_context_programs or 0)

        # The recent best was produced from a 0.6539 frontier candidate. Focus
        # on this narrow frontier, while avoiding repeatedly mutating one ID.
        frontier_size = min(total, max(8, int(math.ceil(total * 0.28))))
        frontier = scored[:frontier_size]

        weights: List[float] = []
        for rank, (program, score) in enumerate(frontier):
            rank_weight = 1.0 + 2.5 * (frontier_size - rank) / frontier_size
            gain = max(0.0, self.parent_best_gain.get(program.id, 0.0))
            productive_bonus = 1.0 + min(1.2, gain * 40.0)
            reuse_penalty = 1.0 + 0.55 * self.parent_uses.get(program.id, 0)
            weights.append(rank_weight * productive_bonus / reuse_penalty)

        parent = self.random_state.choices(
            [program for program, _ in frontier], weights=weights, k=1
        )[0]

        contexts: List[EvolvedProgram] = []
        used_ids = {parent.id}

        def choose(candidates: List[EvolvedProgram]) -> None:
            available = [p for p in candidates if p.id not in used_ids]
            if not available or len(contexts) >= wanted:
                return
            # Prefer context combinations not already tried with this parent.
            unseen = [
                p for p in available
                if self.parent_context_uses.get((parent.id, p.id), 0) == 0
            ]
            chosen = self.random_state.choice(unseen or available)
            contexts.append(chosen)
            used_ids.add(chosen.id)

        # Strong reference, competing frontier implementation, then two
        # deliberately different score bands to encourage useful hybridization.
        choose([p for p, _ in scored[:3]])
        choose([p for p, _ in scored[3:min(total, 20)]])
        choose([p for p, _ in scored[
            int(total * 0.30):max(int(total * 0.30) + 1, int(total * 0.60))
        ]])
        choose([p for p, _ in scored[
            int(total * 0.60):max(int(total * 0.60) + 1, int(total * 0.85))
        ]])

        remaining = [p for p, _ in scored if p.id not in used_ids]
        while remaining and len(contexts) < wanted:
            choose(remaining)
            remaining = [p for p in remaining if p.id not in used_ids]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END