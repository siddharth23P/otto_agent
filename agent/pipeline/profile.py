"""Per-run subtraction from the standing toolbox.

agent/pipeline/toolkit.py ADDS tools for one run. This is the other
direction: a host that knows a standing tool cannot work where it runs takes
it off the menu for the run, so the model is never offered it and never
spends a turn discovering the refusal. The case that forced it is Otto
embedded in an Android app (agent/embed.py): there is no `sys.executable` to
hand `execute_python` a script, no shell worth `execute_bash`, no browser
and no desktop -- seven of the eighteen standing tools, all of which would
otherwise be advertised by `reachable_tools()` and fail on first use.

The same contextvar shape as every other run-scoped fact in this package
(`bind_workspace`, `bind_command_runner`, `bind_extra_tools`), bound on the
thread that consumes the run. Nothing bound is the normal case: an empty set,
and `dispatch_table()`/`reachable_tools()` return exactly what they always
did.

Subtraction only reaches the STANDING tools. A run-scoped tool bound under a
disabled name is still served, because a host that binds `execute_bash` to
its own sandbox means that one, and the disabled set is about the tools that
cannot run here, not about the name.
"""
from __future__ import annotations

import contextvars
from collections.abc import Collection, Iterator
from contextlib import contextmanager

_disabled: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "otto_disabled_tools", default=frozenset(),
)


@contextmanager
def bind_tool_profile(disabled: Collection[str] | None) -> Iterator[None]:
    """Take `disabled` off the standing menu for this block. Replaces rather
    than merges with an outer binding, like `bind_extra_tools`; None or an
    empty collection restores the full toolbox."""
    token = _disabled.set(frozenset(disabled or ()))
    try:
        yield
    finally:
        try:
            _disabled.reset(token)
        except ValueError:
            # Unwound from a different context: a streaming run finalised
            # on another thread (agent/pipeline/tracing.py).
            _disabled.set(frozenset())


def disabled_tools() -> frozenset[str]:
    """The standing tools this run may not reach. Empty is the normal case."""
    return _disabled.get()
