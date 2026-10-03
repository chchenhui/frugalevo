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
    """Adaptive elite search with lineage-aware diversity."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = None
        self.last_meaningful_iteration = -1
        self.parent_uses: Dict[str, int] = {}
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

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        score = self._score(program)
        step = iteration if iteration is not None else program.iteration_found

        if score is not None:
            if self.best_seen is None:
                self.best_seen = score
                self.last_meaningful_iteration = step
            else:
                absolute_gain = score - self.best_seen
                relative_gain = absolute_gain / max(abs(self.best_seen), 1e-12)
                if absolute_gain > 0.01 or relative_gain > 0.01:
                    self.best_seen = score
                    self.last_meaningful_iteration = step
                elif score > self.best_seen:
                    self.best_seen = score

        if program.parent_id:
            self.parent_uses[program.parent_id] = self.parent_uses.get(program.parent_id, 0) + 1
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                if gain > 0.01 or gain / max(abs(parent_score), 1e-12) > 0.01:
                    self.parent_wins[program.parent_id] = self.parent_wins.get(program.parent_id, 0) + 1

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            candidates = list(self.programs.values())
            if not candidates:
                raise ValueError("No candidates available for sampling")
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        scored.sort(key=lambda item: item[1], reverse=True)
        n = len(scored)
        elite = scored[:max(1, int(math.ceil(n * 0.35)))]
        upper = scored[:max(1, int(math.ceil(n * 0.70)))]

        current_iteration = max(
            [p.iteration_found for p, _ in scored if isinstance(p.iteration_found, int)] or [0]
        )
        stalled = current_iteration - self.last_meaningful_iteration >= 8

        # Usually exploit strong solutions, but a plateau deliberately broadens
        # the parent pool rather than repeatedly mutating one identical elite.
        pool = upper if stalled else elite
        low, high = pool[-1][1], pool[0][1]
        weights = []
        for program, score in pool:
            quality = (score - low) / max(high - low, 1e-12)
            uses = self.parent_uses.get(program.id, 0)
            wins = self.parent_wins.get(program.id, 0)
            weights.append(1.0 + 2.0 * quality + 0.5 * wins + 1.0 / (1.0 + uses))
        parent = self.random_state.choices([p for p, _ in pool], weights=weights, k=1)[0]

        count = max(0, num_context_programs or 0)
        remaining = [(p, s) for p, s in scored if p.id != parent.id]
        contexts: List[EvolvedProgram] = []
        seen_solutions = {parent.solution}

        # Draw representatives from several score bands: elite evidence plus
        # alternative constructions that may contain a useful geometric idea.
        bands = [
            remaining[:max(1, n // 4)],
            remaining[max(1, n // 4):max(2, n // 2)],
            remaining[max(2, n // 2):],
        ]
        while len(contexts) < count and remaining:
            band = bands[len(contexts) % len(bands)]
            options = [p for p, _ in band if p.solution not in seen_solutions]
            if not options:
                options = [p for p, _ in remaining if p.solution not in seen_solutions]
            if not options:
                break
            chosen = self.random_state.choice(options)
            contexts.append(chosen)
            seen_solutions.add(chosen.solution)
            remaining = [(p, s) for p, s in remaining if p.id != chosen.id]
            bands = [[(p, s) for p, s in band if p.id != chosen.id] for band in bands]

        label = ""
        if stalled and current_iteration - self.last_meaningful_iteration >= 16:
            label = self.DIVERGE_LABEL
            contexts = []
        elif stalled and self.parent_wins.get(parent.id, 0) > 0:
            label = self.REFINE_LABEL
            contexts = []

        return {label: parent}, {"": contexts}


# EVOLVE-BLOCK-END