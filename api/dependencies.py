"""
dependencies.py — FastAPI dependency providers.

get_agent() returns a process-wide Agent singleton: Agent() loads a joblib
classifier and builds a TF-IDF matrix over ~15.8k retrieval documents at
construction time, which is real, non-trivial work we don't want repeated
per-request. The actual FastAPI override seam is `agent_dependency()` in
api/main.py (which calls this function) -- that's what's declared via
Depends(...) in routes and what tests override, since FastAPI's
dependency_overrides only intercepts things declared that way, not bare
function calls.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent import Agent  # noqa: E402

_agent_instance = None


def get_agent() -> Agent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = Agent()
    return _agent_instance
