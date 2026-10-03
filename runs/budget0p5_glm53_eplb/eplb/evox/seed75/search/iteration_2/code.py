# EVOLVE-BLOCK-START
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


@dataclass
class EvolvedProgram(Program):
    """Program for the evolved database."""


class EvolvedProgramDatabase(ProgramDatabase):
    """Adaptive search: diversify parent/context, escalate to labels on stagnation."""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score = None
        self.stagnation = 0
        self.parent_use_count: Dict[str, int] = {}
        self.label_use_count: Dict[str, int] = {}
        self.diverge_since_improve = 0

    @staticmethod
    def _score(p) -> float:
        v = p.metrics.get("combined_score", 0.0) if isinstance(p.metrics, dict) else 0.0
        return float(v) if isinstance(v, (int, float)) else 0.0

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        self.programs[program.id] = program
        s = self._score(program)
        if self.best_score is None or s > self.best_score + 0.001:
            improved = self.best_score is not None and (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score))
            self.best_score = s
            self.stagnation = 0
            self.diverge_since_improve = 0
            if improved:
                # reward lineage: reset usage pressure on parent
                pid = program.parent_id
                if pid in self.parent_use_count:
                    self.parent_use_count[pid] = max(0, self.parent_use_count[pid] - 2)
        else:
            self.stagnation += 1

        self.parent_use_count[program.parent_id] = self.parent_use_count.get(program.parent_id, 0) + 1
        if program.parent_info and program.parent_info[0]:
            key = program.parent_info[0]
            self.label_use_count[key] = self.label_use_count.get(key, 0) + 1

        if iteration is not None:
            self.last_iteration = max(getattr(self, "last_iteration", 0), iteration)
        if self.config.db_path:
            self._save_program(program)
        self._update_best_program(program)
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        if not self.programs:
            raise ValueError("No candidates available for sampling")

        cands = list(self.programs.values())
        scored = sorted(cands, key=self._score, reverse=True)
        n = len(scored)
        k = num_context_programs or 4

        # --- Parent selection: usage-weighted, mostly top-half with exploration ---
        if self.stagnation >= 3 and self.diverge_since_improve < 3:
            # Stagnant: pick a mid/low-tier parent and diverge to escape plateau
            pool = scored[n // 3: n] or scored
            parent = min(pool, key=lambda p: self.parent_use_count.get(p.id, 0))
            self.diverge_since_improve += 1
            return {self.DIVERGE_LABEL: parent}, {"": []}

        # Weighted choice over top 60% by score, penalizing overused parents
        pool = scored[: max(1, (2 * n) // 3)]
        weights = [1.0 / (1 + self.parent_use_count.get(p.id, 0)) * (0.5 + self._score(p)) for p in pool]
        total = sum(weights) or 1.0
        r = self.random_state.random() * total
        parent = pool[-1]
        acc = 0.0
        for p, w in zip(pool, weights):
            acc += w
            if r <= acc:
                parent = p
                break

        # Occasional refine on a fresh top parent when stagnating mildly
        label = ""
        if self.stagnation >= 2 and self.random_state.random() < 0.3:
            fresh_top = [p for p in scored[:3] if self.parent_use_count.get(p.id, 0) < 3]
            if fresh_top:
                parent = self.random_state.choice(fresh_top)
                label = self.REFINE_LABEL

        # --- Context: diverse tiers, exclude parent, prefer under-used ---
        top = [p for p in scored if p.id != parent.id][:2]
        rest = [p for p in scored[2:] if p.id != parent.id]
        self.random_state.shuffle(rest)
        context = (top + rest)[:k]
        # dedupe
        seen, ctx = set(), []
        for p in context:
            if p.id not in seen:
                seen.add(p.id)
                ctx.append(p)

        return {label: parent}, {"": ctx}


# EVOLVE-BLOCK-END