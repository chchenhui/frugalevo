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
    """Score-weighted exploitative search with stagnation-triggered refine/diverge.

    Principles:
    1) Mostly exploit the top score cluster (weighted by score, penalized by
       reuse) since the population is tightly clustered near the best score.
    2) When progress stalls, switch to explicit REFINE (best program) or
       DIVERGE (top-tier program) labels to break the plateau.
    """

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None
        self.random_state = random.Random(getattr(config, "random_seed", None))
        self.best_score: float = float("-inf")
        self.best_program_id: Optional[str] = None
        self.stagnation: int = 0
        self.parent_usage: Dict[str, int] = {}

    # ---------- helpers ----------

    @staticmethod
    def _score(program: EvolvedProgram) -> Optional[float]:
        if not program.metrics:
            return None
        v = program.metrics.get("combined_score")
        if isinstance(v, (int, float)):
            return float(v)
        return None

    def _pick_weighted(self, pool: List[EvolvedProgram]) -> EvolvedProgram:
        """Pick from pool weighted by score, penalizing overused parents."""
        weights = []
        for p in pool:
            s = self._score(p)
            base = max(s, 1.0)
            usage = self.parent_usage.get(p.id, 0)
            weights.append(base * base / (1.0 + 2.0 * usage))
        return self.random_state.choices(pool, weights=weights, k=1)[0]

    # ---------- required methods ----------

    def add(self, program: EvolvedProgram, iteration: Optional[int] = None, **kwargs) -> str:
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        self.programs[program.id] = program

        if program.parent_id:
            self.parent_usage[program.parent_id] = self.parent_usage.get(program.parent_id, 0) + 1

        s = self._score(program)
        if s is not None:
            meaningful = (s - self.best_score) > max(0.01, 0.01 * abs(self.best_score)) if self.best_score != float("-inf") else True
            if s > self.best_score:
                self.best_score = s
                self.best_program_id = program.id
            if meaningful:
                self.stagnation = 0
            else:
                self.stagnation += 1

        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        if self.config.db_path:
            self._save_program(program)

        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to the evolve database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Dict[str, EvolvedProgram], Dict[str, List[EvolvedProgram]]]:
        candidates = list(self.programs.values())
        if not candidates:
            raise ValueError("No candidates available for sampling")

        n_ctx = num_context_programs or 4

        scored = [(p, self._score(p)) for p in candidates]
        scored = [(p, s) for p, s in scored if s is not None]
        if not scored:
            parent = self.random_state.choice(candidates)
            return {"": parent}, {"": [p for p in candidates if p.id != parent.id][:n_ctx]}

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [p for p, _ in scored[:8]]
        mid = [p for p, _ in scored[8:20]] or top

        # --- Plateau handling ---
        if self.stagnation >= 4:
            if self.stagnation >= 9 and self.random_state.random() < 0.4:
                # Fundamentally new direction from a strong (not necessarily best) program
                parent = self._pick_weighted(top)
                return {self.DIVERGE_LABEL: parent}, {}
            # Targeted refinement of the best program (rotate among top-2 to avoid overuse)
            pool = top[:2] if len(top) >= 2 else top
            parent = self._pick_weighted(pool)
            return {self.REFINE_LABEL: parent}, {}

        # --- Default selection: exploit top cluster, occasionally explore ---
        r = self.random_state.random()
        if r < 0.65:
            parent = self._pick_weighted(top)
        elif r < 0.90:
            parent = self._pick_weighted(mid)
        else:
            parent = self.random_state.choice([p for p, _ in scored])

        # --- Context: strong examples + diverse perspectives ---
        context: List[EvolvedProgram] = []
        top_pool = [p for p in top if p.id != parent.id]
        if top_pool:
            context.extend(self.random_state.sample(top_pool, min(2, len(top_pool))))
        rest = [p for p, _ in scored if p.id != parent.id and p not in context]
        if rest and len(context) < n_ctx:
            # prefer diverse (lower-tier) programs for contrast
            lower = [p for p, s in scored if s < scored[len(scored) // 2][1] and p.id != parent.id]
            pool = lower if lower else rest
            self.random_state.shuffle(pool)
            context.extend(pool[: n_ctx - len(context)])
        if len(context) < n_ctx:
            self.random_state.shuffle(rest)
            context.extend([p for p in rest if p not in context][: n_ctx - len(context)])

        return {"": parent}, {"": context[:n_ctx]}


# EVOLVE-BLOCK-END