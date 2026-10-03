import asyncio
import errno
import importlib.util
import inspect
import logging
import os
import pickle
import signal
import sys
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from skydiscover.config import EvaluatorConfig
from skydiscover.evaluation.evaluation_result import EvaluationResult
from skydiscover.evaluation.llm_judge import LLMJudge
from skydiscover.utils.async_utils import TaskPool
from skydiscover.utils.metrics import format_metrics

logger = logging.getLogger(__name__)
_EVALUATOR_ENV_LOCK = RLock()
_EVALUATOR_TIMEOUT_KWARG = "skydiscover_timeout_seconds"


class Evaluator:
    """
    Runs the user-provided evaluation function on candidate programs.

    Writes the candidate to a temp file, calls evaluate(program_path), and
    returns an EvaluationResult. Supports optional cascade (multi-stage)
    evaluation and LLM-as-a-judge feedback.
    """

    def __init__(
        self,
        config: EvaluatorConfig,
        llm_judge: Optional[LLMJudge] = None,
        max_concurrent: int = 4,
        env_vars: Optional[Dict[str, str]] = None,
    ):
        if not config.evaluation_file:
            raise ValueError("EvaluatorConfig.evaluation_file must be set")

        self.config = config
        self.evaluation_file = config.evaluation_file
        self.program_suffix = config.file_suffix
        self.is_image_mode = config.is_image_mode
        self.llm_judge = llm_judge
        self.task_pool = TaskPool(max_concurrency=max_concurrent)
        self.env_vars = dict(env_vars or {})
        self._workers = set()
        self._stopping = False

        self._load_evaluation_function()
        logger.debug(f"Initialized evaluator with {self.evaluation_file}")

    # ------------------------------------------------------------------
    # Module loading
    # ------------------------------------------------------------------

    def _load_evaluation_function(self) -> None:
        if not os.path.exists(self.evaluation_file):
            raise ValueError(f"Evaluation file not found: {self.evaluation_file}")

        eval_dir = os.path.dirname(os.path.abspath(self.evaluation_file))
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)

        self._module_name = f"_skydiscover_eval_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(self._module_name, self.evaluation_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {self.evaluation_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[self._module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate"):
            raise AttributeError(f"No evaluate() function in {self.evaluation_file}")

        self.evaluate_function = module.evaluate
        self._eval_module = module
        self._validate_cascade_configuration(module)

    def _validate_cascade_configuration(self, module) -> None:
        if not self.config.cascade_evaluation:
            return
        if not hasattr(module, "evaluate_stage1"):
            logger.warning(
                f"cascade_evaluation is true but {self.evaluation_file} has no evaluate_stage1 — will fall back to direct evaluation"
            )
        elif not hasattr(module, "evaluate_stage2"):
            logger.warning(f"{self.evaluation_file} has evaluate_stage1 but no evaluate_stage2")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate_program(
        self,
        program_solution: str,
        program_id: str = "",
        mode: str = "train",
    ) -> EvaluationResult:
        """Evaluate a program and return scores with optional artifacts.

        Args:
            program_solution: Source code of the candidate program.
            program_id: Optional identifier for logging.
            mode: ``"train"`` or ``"test"``.  Ignored by the Python evaluator
                  (the containerized evaluator passes it to evaluate.sh).
        """
        start_time = time.time()
        label = f" {program_id}" if program_id else ""

        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            if self._stopping:
                return EvaluationResult(metrics={"error": 0.0, "cancelled": True})
            try:
                with tempfile.NamedTemporaryFile(suffix=self.program_suffix, delete=False) as f:
                    f.write(program_solution.encode("utf-8"))
                    temp_path = f.name
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    logger.error("Disk full — cannot create temp file")
                    return EvaluationResult(metrics={"error": 0.0, "disk_space_error": True})
                raise

            sidecar_path = None
            if self.is_image_mode:
                sidecar_path = temp_path + ".image_path"
                try:
                    with open(sidecar_path, "w") as sf:
                        sf.write(program_solution)
                except Exception as e:
                    logger.warning(f"Failed to write image sidecar: {e}")

            try:
                if self.config.cascade_evaluation:
                    result = await self._cascade_evaluate(temp_path)
                else:
                    result = await self._run_stage(self.evaluate_function, temp_path)

                eval_result = self._normalize_result(result)

                if self.llm_judge:
                    llm_result = await self.llm_judge.evaluate(program_solution, program_id)
                    if llm_result:
                        for name, value in llm_result.metrics.items():
                            eval_result.metrics[f"llm_{name}"] = value
                        eval_result.artifacts.update(llm_result.artifacts)

                elapsed = time.time() - start_time
                logger.debug(
                    f"Evaluated program{label} in {elapsed:.2f}s: {format_metrics(eval_result.metrics)}"
                )
                return eval_result

            except asyncio.TimeoutError:
                logger.error(
                    f"Program{label} timed out after {time.time() - start_time:.0f}s (limit: {self.config.timeout}s)"
                )
                return EvaluationResult(metrics={"error": 0.0, "timeout": True})

            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.max_retries + 1} failed{label}: {e}"
                )
                if attempt < self.config.max_retries:
                    await asyncio.sleep(1.0)

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if sidecar_path and os.path.exists(sidecar_path):
                    os.unlink(sidecar_path)

        logger.error(f"All attempts failed{label}: {last_exception}")
        return EvaluationResult(metrics={"error": 0.0})

    async def evaluate_batch(
        self,
        programs: List[Tuple[str, str]],
    ) -> List[EvaluationResult]:
        """Evaluate multiple programs concurrently.

        Concurrency is bounded by ``max_concurrent`` (passed at init,
        default 4).

        Args:
            programs: List of ``(solution, program_id)`` tuples.

        Returns:
            List of EvaluationResult in the same order as *programs*.
        """
        return await self.task_pool.gather(
            coros=[self.evaluate_program] * len(programs),
            args_list=list(programs),
        )

    def close(self) -> None:
        """Remove the dynamically loaded evaluation module from sys.modules."""
        self.request_shutdown()
        sys.modules.pop(getattr(self, "_module_name", None), None)

    def request_shutdown(self) -> None:
        """Stop admission and terminate outstanding evaluator process groups."""
        self._stopping = True
        for worker in tuple(self._workers):
            self._kill_worker(worker)

    @staticmethod
    def _kill_worker(worker) -> None:
        try:
            if os.name == "posix":
                os.killpg(worker.pid, signal.SIGKILL)
            elif worker.returncode is None:
                worker.kill()
        except ProcessLookupError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_stage(self, func, program_path: str) -> Any:
        """Run a stage in a fresh interpreter; timeout actually stops its work."""
        if self._stopping:
            raise RuntimeError("Evaluation shutdown requested")
        with tempfile.TemporaryDirectory(prefix="skydiscover-eval-") as directory:
            result_path = os.path.join(directory, "result.pkl")
            worker = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "skydiscover.evaluation.worker",
                os.path.abspath(self.evaluation_file), func.__name__,
                os.path.abspath(program_path), result_path, str(self.config.timeout),
                env={**os.environ, **self.env_vars,
                     "PYTHONPATH": os.pathsep.join(str(p) for p in sys.path)},
                start_new_session=os.name == "posix",
            )
            self._workers.add(worker)
            try:
                if self._stopping:
                    self._kill_worker(worker)
                await asyncio.wait_for(worker.wait(), timeout=self.config.timeout)
                if self._stopping:
                    raise RuntimeError("Evaluation shutdown requested")
                if worker.returncode != 0:
                    raise RuntimeError(f"Evaluation worker exited with code {worker.returncode}")
                with open(result_path, "rb") as handle:
                    ok, result = pickle.load(handle)
                if not ok:
                    raise RuntimeError(result)
                return result
            finally:
                # Also remove descendants left behind by an otherwise finished stage.
                self._kill_worker(worker)
                await worker.wait()
                self._workers.discard(worker)

    @contextmanager
    def _scoped_env(self):
        if not self.env_vars:
            yield
            return

        with _EVALUATOR_ENV_LOCK:
            old_values = {k: os.environ.get(k) for k in self.env_vars}
            try:
                os.environ.update(self.env_vars)
                yield
            finally:
                for key, old_value in old_values.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value

    def _call_with_env(self, func, program_path: str) -> Any:
        with self._scoped_env():
            # Evaluators that manage their own subprocess can opt into the
            # outer timeout.  Keeping the protocol keyword-only and optional
            # preserves compatibility with existing one-argument evaluators,
            # while allowing the subprocess to terminate before wait_for()
            # gives up on the worker thread.
            try:
                parameter = inspect.signature(func).parameters.get(_EVALUATOR_TIMEOUT_KWARG)
            except (TypeError, ValueError):
                parameter = None
            if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
                return func(
                    program_path,
                    **{_EVALUATOR_TIMEOUT_KWARG: self.config.timeout},
                )
            return func(program_path)

    def _normalize_result(self, result: Any) -> EvaluationResult:
        if isinstance(result, EvaluationResult):
            return result
        if isinstance(result, dict):
            return EvaluationResult.from_dict(result)

        logger.warning(f"Unexpected result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})

    async def _cascade_evaluate(self, program_path: str) -> EvaluationResult:
        """Run cascade evaluation: stage1 → threshold check → stage2 → merge."""
        module = self._eval_module

        if not hasattr(module, "evaluate_stage1"):
            return self._normalize_result(
                await self._run_stage(self.evaluate_function, program_path)
            )

        # Stage 1
        try:
            stage1 = self._normalize_result(
                await self._run_stage(module.evaluate_stage1, program_path)
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 1 timed out ({self.config.timeout}s)")
            return EvaluationResult(
                metrics={"error": 0.0, "timeout": True},
                artifacts={"failure_stage": "stage1"},
            )
        except Exception as e:
            logger.error(f"Stage 1 failed: {e}")
            return EvaluationResult(
                metrics={"error": 0.0},
                artifacts={
                    "failure_stage": "stage1",
                    "stderr": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

        if not self._passes_threshold(stage1.metrics, self.config.cascade_thresholds[0]):
            return stage1

        if not hasattr(module, "evaluate_stage2"):
            return stage1

        # Stage 2
        try:
            stage2 = self._normalize_result(
                await self._run_stage(module.evaluate_stage2, program_path)
            )
        except asyncio.TimeoutError:
            logger.error(f"Stage 2 timed out ({self.config.timeout}s)")
            return EvaluationResult(
                metrics={
                    "error": 0.0,
                    "timeout": True,
                    "validity": 0.0,
                    "combined_score": 0.0,
                },
                artifacts={"failure_stage": "stage2"},
            )
        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            return EvaluationResult(
                metrics={"error": 0.0, "validity": 0.0, "combined_score": 0.0},
                artifacts={"failure_stage": "stage2", "stage2_stderr": str(e)},
            )

        # Some evaluators enforce an inner subprocess timeout and return a
        # timeout metric instead of raising asyncio.TimeoutError.  Treat it as
        # equally non-authoritative before the numeric merge converts booleans
        # to floats.
        if stage2.metrics.get("timeout") is True:
            return EvaluationResult(
                metrics={
                    "error": 0.0,
                    "timeout": True,
                    "validity": 0.0,
                    "combined_score": 0.0,
                },
                artifacts={**stage2.artifacts, "failure_stage": "stage2"},
            )

        # Merge stages
        merged_metrics = {
            k: float(v)
            for k, v in {**stage1.metrics, **stage2.metrics}.items()
            if isinstance(v, (int, float)) and k != "error"
        }
        return EvaluationResult(
            metrics=merged_metrics,
            artifacts={**stage1.artifacts, **stage2.artifacts},
        )

    def _passes_threshold(self, metrics: Dict[str, float], threshold: float) -> bool:
        """Check if metrics pass the threshold (combined_score or average)."""
        if not metrics:
            return False

        if "combined_score" in metrics:
            score = metrics["combined_score"]
            if isinstance(score, (int, float)):
                return float(score) >= threshold

        valid = [
            float(v) for k, v in metrics.items() if k != "error" and isinstance(v, (int, float))
        ]
        return (sum(valid) / len(valid)) >= threshold if valid else False
