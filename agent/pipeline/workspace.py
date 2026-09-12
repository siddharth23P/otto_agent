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

NOTHING BOUND WAS THE NORMAL CASE until 2026-09-12, and it is no longer (see
"Interactive sessions bind one too", below). It remains the normal case for a
caller that does not ask for one: `current_workspace()` returning None is a
valid state, every file tool refuses cleanly on it (a failed ToolResult -- the
same shape web_search and rag use for "not available here"), and
execute_bash/execute_python keep their throwaway-tmpdir behaviour exactly.

That confinement is what the file tools' safety argument rests on, and it is
worth being precise about how much it is worth. `resolve_in_workspace()` below
resolves symlinks and rejects anything landing outside the workspace root, so
a path the model invents ("../../.ssh/id_rsa", an absolute path, a symlink
planted by the task itself) cannot escape. What it is NOT: execute_bash runs a
shell IN that directory with no such check, and a shell can obviously write
anywhere the user can. This is a blast-radius argument, not a sandbox -- the
same honest bar agent/pipeline/tools.py's own module docstring already sets
for execute_python.

Interactive sessions bind one too (2026-09-12, design call: "we need
filesystem management so we can use it to write code and work on already
implemented codebases", and, asked where a session should get its workspace
from, "current directory by default"). Until then only agent/eval/'s harnesses
bound anything, each a fresh temp directory it created and deleted, so an
`otto chat`/`otto tui` turn asked to edit a codebase could not read a single
file -- observed live as the agent guessing `/workspace/otto_ui.py` and then
`/tmp/otto_ui.py`, both refused, and answering with the previous turn's
greeting. `otto chat`/`otto tui` now bind the directory they were launched
from, `--workspace PATH` picks a different one, and `--no-workspace` is the
old behaviour back.

Which makes the paragraph above load-bearing in a way it was not: the bound
directory is now, by default, a real repository somebody cares about rather
than a scratch dir nobody will miss. The honest statement of what that buys,
asked as an explicit design call ("confine file tools, shell runs free") and
answered deliberately:

* read_file/write_file/edit_file/list_files cannot touch anything outside the
  root. That is enforced here, by `resolve_in_workspace()`, and tested.
* execute_bash can. Its cwd is the root, but a shell reaches whatever the
  person running otto can reach, and no amount of command inspection changes
  that for a determined one. Fencing it was considered and declined: a guard
  good enough to stop `rm -rf ~/other` also refuses `git -C ~/other status`,
  and a tool that refuses legitimate work gets routed around rather than
  obeyed.

So: run otto against a repository you have committed, the same standing advice
as for any tool that can run commands. `workspace_note()` below is the other
half of making this usable -- a model that is not told where it is invents a
path, which is exactly what the observed failure was.
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


#: How many entries of the workspace root `workspace_note()` names. Enough to
#: recognise a repository from ("pyproject.toml, src, tests" is a Python
#: project and the model will act like it), short enough that a directory with
#: four hundred files does not become the prompt.
_NOTE_ENTRIES = 24


def workspace_note() -> str:
    """The block that tells a prompt where its files are. Empty string when
    nothing is bound, so a caller can append it unconditionally -- the same
    shape and the same reason as agent/pipeline/toolkit.py's `render_note()`,
    which every prompt-builder in nodes.py already handles this way.

    Not optional polish. A model that is not told where it is picks a path out
    of the air: the failure that prompted binding a workspace at all had the
    agent try `/workspace/otto_ui.py`, get "no workspace is bound", and then
    try `/tmp/otto_ui.py`. Both were wrong for a reason no error message it
    could see explained. Naming the root and saying paths are relative to it
    is what turns the file tools from present-but-unusable into usable.

    The listing is part of that. "You are in /Users/x/proj" tells a model
    nothing it can act on; "and it contains pyproject.toml, agent/, tests/"
    tells it what kind of project this is and where to look first, which is
    the difference between a first tool call that reads something real and one
    that guesses a filename.
    """
    root = current_workspace()
    if root is None:
        return ""
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
    except OSError:
        # A root that vanished or cannot be listed is not worth failing a run
        # over -- the tools will report it precisely when one is actually used.
        entries = []
    listing = ", ".join(entries[:_NOTE_ENTRIES])
    if len(entries) > _NOTE_ENTRIES:
        listing += f", and {len(entries) - _NOTE_ENTRIES} more"
    lines = [
        f"WORKSPACE: {root}",
        "This is a real directory on disk and it persists between your tool "
        "calls -- write a file in one and the next call sees it. read_file, "
        "write_file, edit_file and list_files take paths RELATIVE to this "
        "root (an absolute path inside it works too); anything resolving "
        "outside it is refused. execute_bash and execute_python run with this "
        "as their working directory.",
        "Read before you write. These are somebody's real files, not a "
        "scratch directory: open what is already there and match it rather "
        "than creating a new file beside it.",
    ]
    if listing:
        lines.insert(1, f"It contains: {listing}")
    return "\n".join(lines)


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
