"""Private evaluator subprocess entry point (trusted local evaluator protocol)."""
import importlib.util
import inspect
import os
import pickle
import sys
import traceback


def main():
    evaluator_path, stage, program_path, result_path, timeout = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(evaluator_path))
    try:
        spec = importlib.util.spec_from_file_location("_skydiscover_worker_eval", evaluator_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        func = getattr(module, stage)
        name = "skydiscover_timeout_seconds"
        parameter = inspect.signature(func).parameters.get(name)
        kwargs = {name: float(timeout)} if parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY else {}
        result = (True, func(program_path, **kwargs))
    except BaseException:
        result = (False, traceback.format_exc())
    with open(result_path, "wb") as handle:
        pickle.dump(result, handle)
    # Do not let candidate-created threads block interpreter shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
