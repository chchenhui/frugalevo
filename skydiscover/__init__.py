"""
SkyDiscover: Self-Improving Framework for LLMs
"""

from skydiscover._version import __version__
from skydiscover.api import (
    DiscoveryResult,
    discover_solution,
    run_discovery,
)
from skydiscover.runner import Runner

__all__ = [
    "Runner",
    "__version__",
    "run_discovery",
    "discover_solution",
    "DiscoveryResult",
]
