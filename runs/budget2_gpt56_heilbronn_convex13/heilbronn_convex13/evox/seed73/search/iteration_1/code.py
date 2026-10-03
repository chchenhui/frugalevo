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
    """Adaptive elite search with diversity-aware context selection."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.initial_program = None
        self.best_seen_score: Optional[float] = None
        self.stall_count = 0
        self.add_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.parent_success_count: Dict[str, int] = {}

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                return value
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        score = self._score(program)
        previous_best = self.best_seen_score

        if score is not None:
            if previous_best is None:
                self.best_seen_score = score
            elif score > previous_best:
                required_gain = max(0.01, abs(previous_best) * 0.01)
                if score - previous_best > required_gain:
                    self.stall_count = 0
                else:
                    self.stall_count += 1
                self.best_seen_score = score
            else:
                self.stall_count += 1

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )
            parent = self.get(program.parent_id)
            parent_score = self._score(parent) if parent is not None else None
            if (
                score is not None
                and parent_score is not None
                and score - parent_score > max(0.01, abs(parent_score) * 0.01)
            ):
                self.parent_success_count[program.parent_id] = (
                    self.parent_success_count.get(program.parent_id, 0) + 1
                )

        self.add_count += 1
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
        if not candidates:
            raise ValueError("No candidates available for sampling")

        scored = [(p, self._score(p)) for p in candidates]
        numeric = [(p, s) for p, s in scored if s is not None]
        if not numeric:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": []}

        numeric.sort(key=lambda item: item[1], reverse=True)
        best_score = numeric[0][1]
        elite_size = max(3, min(len(numeric), int(math.ceil(len(numeric) * 0.35))))
        elite = numeric[:elite_size]

        # Mostly exploit strong designs, while reducing repeated use of the
        # same parent and rewarding lineages that previously improved.
        choices: List[EvolvedProgram] = []
        weights: List[float] = []
        for program, score in elite:
            quality = 1.0 + 4.0 * max(0.0, score / best_score)
            used = self.parent_use_count.get(program.id, 0)
            successes = self.parent_success_count.get(program.id, 0)
            weight = quality * (1.0 + 0.5 * successes) / (1.0 + 0.35 * used)
            choices.append(program)
            weights.append(weight)

        # Periodically test a good but non-elite alternative to avoid all
        # future work collapsing onto one nearly-identical construction.
        if len(numeric) > elite_size and self.random_state.random() < 0.22:
            upper_middle = numeric[elite_size : min(len(numeric), elite_size + 6)]
            parent = self.random_state.choice(upper_middle)[0]
        else:
            parent = self.random_state.choices(choices, weights=weights, k=1)[0]

        # Labels are reserved for a genuine plateau.  Early plateau response
        # refines the incumbent; a deeper plateau asks for a fresh direction.
        label = ""
        if self.stall_count >= 16 and self.stall_count % 6 in (0, 1):
            label = self.DIVERGE_LABEL
            alternatives = [p for p, _ in numeric[1:elite_size]]
            if alternatives:
                parent = self.random_state.choice(alternatives)
        elif self.stall_count >= 9 and self.stall_count % 5 == 0:
            label = self.REFINE_LABEL
            parent = numeric[0][0]

        if label:
            return {label: parent}, {}

        context_limit = max(0, num_context_programs or 0)
        context: List[EvolvedProgram] = []
        used_ids = {parent.id}

        # Include the best known construction when it is not already parent.
        for program, _ in numeric:
            if program.id not in used_ids:
                context.append(program)
                used_ids.add(program.id)
                break

        # Add high-quality, score-distinct examples rather than redundant copies.
        for program, score in numeric:
            if len(context) >= context_limit:
                break
            if program.id in used_ids:
                continue
            if all(
                abs(score - (self._score(existing) or 0.0)) > 0.002
                for existing in context
            ):
                context.append(program)
                used_ids.add(program.id)

        remaining = [p for p, _ in numeric if p.id not in used_ids]
        self.random_state.shuffle(remaining)
        for program in remaining:
            if len(context) >= context_limit:
                break
            context.append(program)

        return {"": parent}, {"": context[:context_limit]}


# EVOLVE-BLOCK-END