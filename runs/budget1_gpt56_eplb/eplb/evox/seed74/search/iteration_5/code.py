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
    """Adaptive score-band search with parent reuse control."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score_seen = float("-inf")
        self.last_meaningful_improvement_iteration = 0
        self.added_program_count = 0
        self.parent_use_count: Dict[str, int] = {}
        self.label_use_count: Dict[str, int] = {
            "refine": 0,
            "diverge": 0,
        }

    def _score(self, program: EvolvedProgram) -> Optional[float]:
        value = program.metrics.get("combined_score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def add(
        self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs: Any
    ) -> str:
        """Store programs while rebuilding useful search-history signals."""
        self.programs[program.id] = program
        self.added_program_count += 1

        current_iteration = iteration
        if current_iteration is None:
            current_iteration = program.iteration_found
        if isinstance(current_iteration, int):
            self.last_iteration = max(self.last_iteration, current_iteration)

        if program.parent_id:
            self.parent_use_count[program.parent_id] = (
                self.parent_use_count.get(program.parent_id, 0) + 1
            )

        parent_label = ""
        if program.parent_info and len(program.parent_info) >= 1:
            parent_label = program.parent_info[0]
        if parent_label == self.REFINE_LABEL:
            self.label_use_count["refine"] += 1
        elif parent_label == self.DIVERGE_LABEL:
            self.label_use_count["diverge"] += 1

        score = self._score(program)
        if score is not None:
            if self.best_score_seen == float("-inf"):
                self.best_score_seen = score
                self.last_meaningful_improvement_iteration = int(current_iteration or 0)
            else:
                absolute_gain = score - self.best_score_seen
                relative_gain = absolute_gain / max(abs(self.best_score_seen), 1e-12)
                if absolute_gain > 0.01 or relative_gain > 0.01:
                    self.best_score_seen = score
                    self.last_meaningful_improvement_iteration = int(current_iteration or 0)
                elif score > self.best_score_seen:
                    self.best_score_seen = score

        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)

        logger.debug("Added program %s", program.id)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs: Any
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        scored = [(p, self._score(p)) for p in self.programs.values()]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            raise ValueError("No scored candidates available for sampling")

        scored.sort(key=lambda item: item[1], reverse=True)
        programs = [p for p, _ in scored]
        scores = [s for _, s in scored]
        n = len(programs)
        context_count = max(0, num_context_programs or 0)

        current_iteration = max(
            self.last_iteration,
            max((p.iteration_found for p in programs), default=0),
        )
        stalled = (
            current_iteration - self.last_meaningful_improvement_iteration >= 8
            and n >= 8
        )

        # The observed useful children came from strong, but not always best,
        # parents.  Draw from the top half and penalize repeatedly used parents.
        elite_count = max(3, (n + 1) // 2)
        elite = programs[:elite_count]
        rank_weights: List[float] = []
        for rank, candidate in enumerate(elite):
            reuse = self.parent_use_count.get(candidate.id, 0)
            rank_weights.append((elite_count - rank) / (1.0 + reuse))

        parent = self.random_state.choices(elite, weights=rank_weights, k=1)[0]
        label = ""

        # At a real plateau, explicitly try the missed "refine" operation once
        # before requesting a more radical isolated alternative.
        if stalled and self.label_use_count["refine"] == 0:
            parent = programs[0]
            label = self.REFINE_LABEL
        elif stalled and self.label_use_count["diverge"] == 0:
            # A high-but-not-best parent is more likely to contain a distinct
            # viable approach than the repeatedly copied incumbent.
            band_start = min(max(1, n // 4), n - 1)
            band_end = min(n, max(band_start + 1, n // 2 + 1))
            alternatives = programs[band_start:band_end]
            parent = self.random_state.choice(alternatives)
            label = self.DIVERGE_LABEL

        if label:
            return {label: parent}, {}

        # Context deliberately mixes the incumbent, a near-best alternative,
        # and lower-score examples instead of repeatedly supplying only best code.
        pool = [p for p in programs if p.id != parent.id]
        contexts: List[EvolvedProgram] = []

        if pool and context_count:
            contexts.append(pool[0])  # incumbent / best known reference

        if len(pool) > 2 and len(contexts) < context_count:
            upper_start = min(len(pool) - 1, max(1, len(pool) // 4))
            upper_end = min(len(pool), max(upper_start + 1, len(pool) // 2))
            candidate = self.random_state.choice(pool[upper_start:upper_end])
            if candidate.id not in {p.id for p in contexts}:
                contexts.append(candidate)

        remaining = [p for p in pool if p.id not in {c.id for c in contexts}]
        self.random_state.shuffle(remaining)
        while remaining and len(contexts) < context_count:
            contexts.append(remaining.pop())

        return {"": parent}, {"": contexts[:context_count]}


# EVOLVE-BLOCK-END