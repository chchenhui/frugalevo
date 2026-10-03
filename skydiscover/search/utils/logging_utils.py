"""Logging utilities for the search subsystem.

Provides a compact console formatter, a noise-reducing filter, and a
one-call helper to wire up both file and console logging for a
discovery run.
"""

import logging
import os
import time

_QUIET = {"route", "server"}


class _ConsoleFormatter(logging.Formatter):
    """Compact single-line formatter: ``HH:MM:SS [module] message``."""

    def format(self, record):
        ts = self.formatTime(record, "%H:%M:%S")
        name = (
            record.name[len("skydiscover.") :]
            if record.name.startswith("skydiscover.")
            else record.name
        )
        parts = name.split(".")
        short = f"search.{parts[1]}" if parts[0] == "search" and len(parts) >= 3 else parts[-1]
        fmt = (
            f"{ts} {record.levelname} [{short}] "
            if record.levelno >= logging.WARNING
            else f"{ts} [{short}] "
        )
        return fmt + record.getMessage()


class _ConsoleFilter(logging.Filter):
    """Only pass skydiscover messages, suppressing noisy modules below WARNING."""

    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return True
        if not record.name.startswith("skydiscover") or record.name.split(".")[-1] in _QUIET:
            return False
        return True


#: Marks handlers installed by :func:`setup_search_logging`, so re-setup can
#: retire its own without disturbing handlers a caller added.
_OWNED = "_skydiscover_owned"


def setup_search_logging(log_level: str, log_dir: str, name: str) -> None:
    """Configure root logger with a timestamped file handler and a console handler.

    Both handlers respect *log_level*. This prevents third-party HTTP DEBUG
    records (including full LLM request payloads) from flooding INFO logs.
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    console_level = getattr(logging, log_level)
    root.setLevel(console_level)
    # The CLI installs a temporary WARNING level before it has loaded the YAML
    # config.  Logger levels are checked before root handlers, so changing only
    # the root here left every ``skydiscover.*`` INFO record suppressed even
    # when the run config requested INFO.  That hid per-call costs and made
    # native BA-AUC reconstruction impossible.
    logging.getLogger("skydiscover").setLevel(console_level)

    # Retire the file handler from any previous run in this process. Without
    # this, one process running several searches keeps writing every later
    # run's records into every earlier run's file, so the first log becomes a
    # superset of the whole session and per-run analysis silently misreads it.
    # Only handlers this function installed are touched; a caller's own
    # handlers are left alone.
    for handler in list(root.handlers):
        if getattr(handler, _OWNED, False) and isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    log_file = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_file)
    setattr(fh, _OWNED, True)
    fh.setLevel(console_level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root.addHandler(fh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(_ConsoleFormatter())
        ch.addFilter(_ConsoleFilter())
        root.addHandler(ch)

    logging.getLogger(__name__).debug(f"Logging to {log_file}")
