"""Per-run binding of WHERE a tool's command actually runs.

By default it is this process's own machine: agent/pipeline/tools.py's
execute_bash shells out with subprocess, and the file tools touch the real
filesystem through agent/pipeline/workspace.py. That is right for a live
`otto chat` turn, and for the golden eval.

It is wrong for an agentic benchmark. Terminal-Bench (and the container-based
harnesses after it) hand an agent a task that lives entirely inside a Docker
container: the repo, the broken service, the test suite, the files the task's
own grader will look at afterwards are all in there, and none of it exists on
the host. An agent that runs `pytest` on the host is not attempting the task,
it is failing it in a way that looks like a bad answer rather than a wiring
mistake.

So this is the seam: bind a `CommandRunner` and every shell-shaped tool routes
through it instead of subprocess. One seam rather than a container-aware
branch in each tool, because the tools should not know what a container is --
the same reason `summarize` is injected into TieredQueue rather than imported
(agent/memory/queue.py), and the same contextvar shape agent/memory/session.py
and agent/pipeline/workspace.py already use to reach a tool that is a plain
function of one string.

A runner returns `(stdout, stderr, returncode)` rather than a ToolResult so
that this module stays importable by tools.py without a cycle, and so a
harness writing one never has to import the pipeline at all.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Callable, Iterator

#: (command, timeout_seconds) -> (stdout, stderr, returncode). A returncode of
#: -1 conventionally means "timed out", matching what tools.py reports for a
#: subprocess.TimeoutExpired, so a tool loop reads the two the same way.
CommandRunner = Callable[[str, float], tuple[str, str, int]]

_current: contextvars.ContextVar[CommandRunner | None] = contextvars.ContextVar(
    "otto_current_command_runner", default=None,
)


@contextmanager
def bind_command_runner(runner: CommandRunner | None) -> Iterator[None]:
    """Route every shell-shaped tool through `runner` for this block. None
    restores the default (run on this machine), so a harness can bind per
    task and unbind between them without leaking one task's container into
    the next.
    """
    token = _current.set(runner)
    try:
        yield
    finally:
        _current.reset(token)


def current_command_runner() -> CommandRunner | None:
    """The runner bound by the innermost enclosing `bind_command_runner()`, or
    None -- the normal case, meaning "run here", not an error."""
    return _current.get()
