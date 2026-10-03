"""Run the original EPLB evaluator in a fresh, serial, single-threaded process.

This is process isolation, not a security sandbox. The shared advisory lock
coordinates invocations of this adapter, including different search processes.
"""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def evaluate(program_path, *, skydiscover_timeout_seconds=3600):
    deadline = time.monotonic() + float(skydiscover_timeout_seconds) - 5
    lock_path = os.environ.get('EPLB_LOCK_FILE', f'/tmp/skydiscover-eplb-{os.getuid()}.lock')
    try:
        with open(lock_path, 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Timed out waiting for EPLB evaluation lock')
                    time.sleep(0.1)
            timeout = min(float(os.environ.get('EPLB_EVAL_TIMEOUT', '360')), deadline-time.monotonic())
            if timeout <= 0:
                raise TimeoutError('EPLB evaluation deadline expired')
            with tempfile.TemporaryDirectory(prefix='eplb-eval-') as tmp:
                result_path = Path(tmp) / 'result.json'
                env = os.environ.copy()
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = '1'
                with (Path(tmp)/'worker.log').open('w+') as log:
                    proc = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), str(Path(program_path).resolve()), str(result_path)],
                        stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                    )
                    try:
                        proc.wait(timeout=timeout)
                    finally:
                        # Also clean up candidate descendants before releasing the lock.
                        try:
                            os.killpg(proc.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                    if proc.returncode != 0 or not result_path.exists():
                        log.seek(0, 2)
                        log.seek(max(0, log.tell()-4000))
                        raise RuntimeError(f'EPLB worker failed: {log.read()}')
                return json.loads(result_path.read_text())
    except Exception as exc:
        return {'combined_score': 0.0, 'error': str(exc)}


def worker(program_path, result_path):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = ROOT/'benchmarks/ADRS/eplb/evaluator/evaluator.py'
    spec = importlib.util.spec_from_file_location('eplb_original_evaluator', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.WORKLOAD_PATH = os.environ.get('EPLB_WORKLOAD', str(ROOT/'benchmarks/ADRS/eplb/expert-load.json'))
    result = module.evaluate(program_path)
    Path(result_path).write_text(json.dumps(result))


if __name__ == '__main__':
    worker(*sys.argv[1:])
