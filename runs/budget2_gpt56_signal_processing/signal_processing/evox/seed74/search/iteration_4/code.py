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
    """Plateau-aware elite search with score-diverse context."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_seen = float("-inf")
        self.last_meaningful_iteration = -1
        self.last_iteration_seen = 0
        self.parent_uses: Dict[str, int] = {}
        self.parent_best_gain: Dict[str, float] = {}
        self.label_uses: Dict[str, int] = {}

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
        old_best = self.best_seen
        if score is not None and score > self.best_seen:
            self.best_seen = score
            # Meaningful means clearing either the relative or absolute bar.
            threshold = min(0.01, max(0.0, old_best) * 0.01)
            if old_best == float("-inf") or score - old_best > threshold:
                self.last_meaningful_iteration = self.last_iteration_seen

        if program.parent_id:
            self.parent_uses[program.parent_id] = (
                self.parent_uses.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if score is not None and parent_score is not None:
                gain = score - parent_score
                self.parent_best_gain[program.parent_id] = max(
                    gain, self.parent_best_gain.get(program.parent_id, float("-inf"))
                )

        if isinstance(program.parent_info, tuple) and program.parent_info:
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
        current = max(
            self.last_iteration_seen,
            getattr(self, "last_iteration", 0),
            max(p.iteration_found for p, _ in scored),
        )
        stalled = current - self.last_meaningful_iteration

        # Keep one representative per score initially. This avoids spending the
        # whole window mutating the many identical best-score descendants.
        representatives: List[Tuple[EvolvedProgram, float]] = []
        seen_scores = set()
        for program, score in scored:
            key = round(score, 8)
            if key not in seen_scores:
                representatives.append((program, score))
                seen_scores.add(key)

        elite_count = min(len(representatives), max(5, int(total * 0.35)))
        elite = representatives[:elite_count]

        weights: List[float] = []
        for rank, (program, score) in enumerate(elite):
            quality = 1.0 + 2.5 * (elite_count - rank) / max(1, elite_count)
            gain = max(0.0, self.parent_best_gain.get(program.id, 0.0))
            productive = 1.0 + min(1.2, gain * 30.0)
            reuse = 1.0 + 0.55 * self.parent_uses.get(program.id, 0)
            weights.append(quality * productive / reuse)

        parent = self.random_state.choices(
            [p for p, _ in elite], weights=weights, k=1
        )[0]

        # The population is visibly stuck at a repeated frontier. First request
        # a targeted improvement of an elite design, then request a genuinely
        # different direction if that does not break the plateau.
        if stalled >= 7 and self.label_uses.get(self.REFINE_LABEL, 0) < 2:
            best_choices = [p for p, s in representatives[:4]]
            parent = self.random_state.choice(best_choices)
            return {self.REFINE_LABEL: parent}, {}

        if stalled >= 10 and self.label_uses.get(self.DIVERGE_LABEL, 0) < 2:
            upper = representatives[:min(len(representatives), max(8, int(total * 0.60)))]
            parent = self.random_state.choice([p for p, _ in upper])
            return {self.DIVERGE_LABEL: parent}, {}

        wanted = max(0, num_context_programs or 0)
        contexts: List[EvolvedProgram] = []
        used = {parent.id}

        # Supply contrasting strong approaches: current best, near-best, and
        # useful upper-middle alternatives rather than duplicate frontier code.
        bands = [
            representatives[:3],
            representatives[3:8],
            representatives[max(0, int(len(representatives) * 0.35)):
                            max(1, int(len(representatives) * 0.70))],
        ]
        for band in bands:
            options = [p for p, _ in band if p.id not in used]
            if options and len(contexts) < wanted:
                choice = self.random_state.choice(options)
                contexts.append(choice)
                used.add(choice.id)

        remaining = [p for p, _ in representatives if p.id not in used]
        while remaining and len(contexts) < wanted:
            choice = self.random_state.choice(remaining)
            contexts.append(choice)
            used.add(choice.id)
            remaining = [p for p in remaining if p.id != choice.id]

        return {"": parent}, {"": contexts}


# EVOLVE-BLOCK-END