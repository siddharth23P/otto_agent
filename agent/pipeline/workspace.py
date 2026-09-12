"""Per-run binding of a WORKSPACE -- one real directory on disk that survives
across tool calls, so a task can read a file, change it, run the tests, and
read the failure back.

Otto had no such thing until now, and the gap was invisible because nothing
had needed it: agent/pipeline/tools.py's execute_python and execute_bash each
run in their own `tempfile.TemporaryDirectory()`, created and destroyed inside
the single call. That is exactly right for what they were built for -- a node
checking its own arithmetic, or its own snippet, with no way to leave anything
behind. It also means two calls in a row share nothing at all: writing a file
in one and reading it in the next simply does not work, so no amount of
prompting could get this agent through a task that edits a codebase.

Bound the same way agent/memory/session.py binds a MemoryStore, and for the
same reason: a tool in TOOL_DISPATCH is a plain function of one string. It
cannot see AgentState, the run, or its caller, so anything run-scoped has to
reach it through a contextvar rather than an argument.

NOTHING BOUND IS THE NORMAL CASE, and it is what keeps this from widening what
an ordinary `otto chat` turn can do. A chat run binds no workspace, so every
file tool refuses (cleanly, as a failed ToolResult -- the same shape web_search
and rag use for "not available here"), and execute_bash/execute_python keep
their old throwaway-tmpdir behaviour exactly. Only a caller that deliberately
opens a workspace -- agent/eval/'s benchmark harnesses -- gets file access at
all, and only inside the directory it chose.

That confinement is what the file tools' safety argument rests on, and it is
worth being precise about how much it is worth. `resolve_in_workspace()` below
resolves symlinks and rejects anything landing outside the workspace root, so
a path the model invents ("../../.ssh/id_rsa", an absolute path, a symlink
planted by the task itself) cannot escape. What it is NOT: execute_bash runs a
shell IN that directory with no such check, and a shell can obviously write
anywhere the user can. So a bound workspace should be a directory the caller
is willing to lose, and the harnesses that bind one create a fresh temp
directory per instance and delete it afterwards. This is a blast-radius
argument, not a sandbox -- the same honest bar agent/pipeline/tools.py's own
module docstring already sets for execute_python.
"""
from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_current: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "otto_current_workspace", default=None,
)


class OutsideWorkspace(Exception):
    """This path cannot be used: it escaped the workspace root, or the
    filesystem could not evaluate it at all.

    The second case was added after it cost a whole run. On Claw-Eval task C01
    the model put a sentence of prose where a file path goes;
    `resolve_in_workspace` called `Path.exists()` on it and got
    `OSError(ENAMETOOLONG)`, because pathlib swallows only ENOENT, ENOTDIR,
    EBADF and ELOOP. Every file tool catches THIS exception and nothing else,
    so the OSError unwound the entire graph -- discarding a mortgage
    comparison the agent had already computed and verified, after 1096
    seconds. The grader saw a conversation with no assistant messages at all.

    Both cases mean the same thing to a caller ("that is not a path you can
    act on, say so and move on"), and tools.py's own contract is that a tool
    fails cleanly rather than raising, so they share one exception rather than
    making every call site catch two.
    """


@contextmanager
def bind_workspace(path: Path | str | None) -> Iterator[Path | None]:
    """Make `path` the workspace `current_workspace()` returns for this block
    and anything it calls. Created if it doesn't exist; never deleted here --
    whoever opened it owns its lifetime. Restores the previous binding on
    exit, so nesting unwinds correctly.
    """
    resolved = None
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
    token = _current.set(resolved)
    try:
        yield resolved
    finally:
        _current.reset(token)


def current_workspace() -> Path | None:
    """The workspace bound by the innermost enclosing `bind_workspace()`, or
    None -- the normal state for an ordinary chat/eval run, and not an error.
    Callers are expected to fail cleanly on None, never to raise.
    """
    return _current.get()


def resolve_in_workspace(relative: str) -> Path:
    """`relative` as a real path inside the bound workspace.

    Raises OutsideWorkspace if nothing is bound, or if the path escapes the
    root. Two different escapes have to be closed, and closing only one of
    them is a hole: `..` segments are lexical and can appear in a path whose
    directories do not exist yet ("x/../../out.txt"), while a symlink escape
    only shows up once the filesystem is consulted. So the path is normalised
    lexically FIRST, then the deepest ancestor that actually exists is
    resolved for symlinks and the untravelled remainder re-attached -- which
    also means a file being created for the first time (the whole point of a
    write) gets the same check as one already there, where a plain
    `Path.resolve()` on a missing leaf would not.
    """
    root = current_workspace()
    if root is None:
        raise OutsideWorkspace("no workspace is bound for this run")
    root = root.resolve()

    try:
        lexical = Path(os.path.normpath(root / relative.strip()))
        existing = lexical
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        resolved = existing.resolve() / lexical.relative_to(existing)
    except (OSError, ValueError) as exc:
        # A model that wrote prose, a NUL byte, or 4000 characters into a path
        # field. Not an escape attempt and not worth crashing a run over --
        # see this module's OutsideWorkspace docstring for what that cost once.
        raise OutsideWorkspace(
            f"{relative.strip()[:80]!r} is not a usable path: {exc}"
        ) from None

    try:
        resolved.relative_to(root)
    except ValueError:
        raise OutsideWorkspace(f"{relative!r} resolves outside the workspace") from None
    return resolved
