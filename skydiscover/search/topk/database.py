import logging
from typing import List, Optional, Tuple

from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase

logger = logging.getLogger(__name__)


class TopKDatabase(ProgramDatabase):
    """Database for top-k programs"""

    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.initial_program = None

    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        """Add a program to the database (minimal Top-K)."""
        # Store the initial program
        if iteration == 0 or program.iteration_found == 0:
            self.initial_program = program

        # Store the program
        self.programs[program.id] = program

        # Track last iteration if provided
        if iteration is not None:
            self.last_iteration = max(self.last_iteration, iteration)

        # Save to disk if configured
        if self.config.db_path:
            self._save_program(program)

        # NOTE: no enforcement on population size at all
        # Update the absolute best program tracking
        self._update_best_program(program)

        logger.debug(f"Added program {program.id} to top-k database")
        return program.id

    def sample(
        self, num_context_programs: Optional[int] = 4, **kwargs
    ) -> Tuple[Program, List[Program]]:
        """
        Sample a program and context programs for the next discovery step.

        Top-K sampling strategy:
        - Parent: Top 1 program (best program)
        - Context programs: Next K programs (ranks 2 to K+1)

        Args:
            num_context_programs: Number of context programs for context (defaults to 5)
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (parent, context_programs).
        """
        if not self.programs:
            raise ValueError("Cannot sample: no programs in database")

        # Get top (K+1) programs: top 1 for parent, next K for context programs
        total_needed = num_context_programs + 1
        top_programs = self.get_top_programs(total_needed)

        if not top_programs:
            raise ValueError("Cannot sample: no programs available after filtering")

        if len(top_programs) < 2:
            # Only one program available, use it as both parent and context program
            parent = top_programs[0]
            context_programs = [top_programs[0]]
            logger.debug(
                "Top K search: only 1 program available, using as both parent and context program"
            )
        else:
            # Parent is top 1, context_programs is next K
            parent = top_programs[0]
            context_programs = top_programs[1 : min(len(top_programs), num_context_programs + 1)]
            logger.debug(
                f"Top K search: parent {parent.id} (rank 1), context programs {len(context_programs)} programs (ranks 2-{len(context_programs)+1})"
            )

        return parent, context_programs
