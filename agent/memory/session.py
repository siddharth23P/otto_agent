"""Per-run binding of a session's MemoryStore -- how agent/pipeline/tools.py's
`recall_memory` tool (a plain function of one string, TOOL_DISPATCH's shape
-- it cannot see AgentState, a session id, or anything about its caller)
finds out which session's compacted history to search.

Deliberately its own tiny module rather than folded into agent/memory/
wiring.py: wiring.py crosses into agent.pipeline/agent.router (ROUTER,
Task, _call) to build the `summarize` callback a live TieredQueue needs,
and agent/pipeline/tools.py is imported BY agent/pipeline/nodes.py --
tools.py importing wiring.py would close a cycle (nodes.py -> tools.py ->
wiring.py -> nodes.py). This module imports only agent.memory.store, so
tools.py can depend on it with no cycle at all; agent/pipeline/run.py
(which already imports both agent.pipeline.nodes and, transitively via
wiring.py, this module) is what actually calls `bind_store()`, once per
turn, around the graph call -- see its own module docstring.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

from agent.memory.store import MemoryStore

_current: contextvars.ContextVar[MemoryStore | None] = contextvars.ContextVar(
    "otto_current_memory_store", default=None,
)


@contextmanager
def bind_store(store: MemoryStore | None) -> Iterator[None]:
    """Make `store` the one `current_store()` returns for the duration of
    this `with` block (and anything it calls, including across an `await`
    or a background thread that inherits this context -- contextvars, not
    a plain module global, specifically so two turns running concurrently
    never see each other's store). Restores whatever was bound before on
    exit, so a nested bind (there shouldn't be one today, but nothing here
    assumes there won't ever be) unwinds correctly.
    """
    token = _current.set(store)
    try:
        yield
    finally:
        _current.reset(token)


def current_store() -> MemoryStore | None:
    """The MemoryStore bound by the innermost enclosing `bind_store()`, or
    None if nothing bound one -- e.g. agent/eval/'s golden runner, or any
    other caller of agent.pipeline.run that never wired memory in at all.
    None is a normal, expected state here, not an error: callers (tools.py's
    recall_memory) are expected to fail cleanly, not raise, when it happens.
    """
    return _current.get()
