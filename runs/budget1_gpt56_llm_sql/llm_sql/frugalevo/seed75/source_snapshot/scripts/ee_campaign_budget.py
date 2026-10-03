"""Conservative cross-process reservations for a bounded experimental campaign.

Reserve each run's entire maximum before it may start any API work. Normal
completion releases unused dollars. Crashed runs keep their full reservation
until explicitly reconciled from authoritative records.
"""

from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path


@contextmanager
def locked_ledger(path):
    path = Path(path)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text())  # Missing/corrupt ledger fails closed.
        yield data
        path.write_text(json.dumps(data, indent=2))


def reserve_run(path, output, limit):
    if not math.isfinite(limit) or not 0 < limit <= 2:
        raise ValueError("Each run must reserve more than zero and at most USD 2")
    key = str(Path(output).resolve())
    with locked_ledger(path) as data:
        if key in data["runs"]:
            raise ValueError("Run already registered; do not launch a duplicate")
        # Previously launched runs can finish without importing this helper.
        for run_path, run in data["runs"].items():
            completion = Path(run_path) / "completed.json"
            if run["status"] == "reserved" and completion.exists():
                charged = float(json.loads(completion.read_text())["charged"])
                if not math.isfinite(charged) or not 0 <= charged <= run["limit"]:
                    raise ValueError("Invalid completion accounting; manual audit required")
                run.update(status="completed", charged=charged)
        committed = sum(r["charged"] if r["status"] == "completed" else r["limit"]
                        for r in data["runs"].values())
        if committed + limit > data["limit"] + 1e-12:
            raise RuntimeError(f"Campaign cap prevents launch: {committed:.6f} committed, {limit:.6f} requested")
        data["runs"][key] = {"status": "reserved", "limit": limit}


def settle_run(path, output, charged):
    key = str(Path(output).resolve())
    with locked_ledger(path) as data:
        run = data["runs"][key]
        if not math.isfinite(charged) or not 0 <= charged <= run["limit"]:
            raise ValueError("Actual charge exceeds reservation; manual audit required")
        run.update(status="completed", charged=charged)
