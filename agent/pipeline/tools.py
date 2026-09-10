"""The agent's tool registry, tiered by reversibility.

Every tool is registered with an explicit tier (READ_ONLY or MUTATING).
Every node in this graph (router excluded -- it only ever dispatches, see
nodes.py) may only ever be handed READ_ONLY tools, via TOOL_DISPATCH below:
a role node's or the evaluator's output has not been judged yet while it is
still iterating, and an irreversible action taken before judgment can't be
undone if the evaluator later rejects that attempt. Same reasoning the
retired swarm pipeline's identical module docstring gave for its workers.

Eleven tools exist (2026-09-10, replacing the swarm pipeline's execute_python
tool as this graph's whole tool box, per the router/planner/solver/
summarizer/finder/evaluator design):

  execute_python -- real. Runs in the bound workspace if there is one,
                     otherwise in a throwaway temp dir as it always did.
  execute_bash    -- real: a generic shell command, same sandboxing bar as
                     execute_python (see its own docstring for the honest
                     limits of that bar), and the same workspace behaviour.
  read_file       -- real, WORKSPACE tier. Reads a file from the bound
  list_files         workspace; lists what is in it. Both READ_ONLY.
  write_file      -- real, WORKSPACE tier. Creates/overwrites a file, and
  edit_file          replaces an exact unique snippet in one. These are the
                     first tools in this repo that change anything on disk
                     the caller didn't hand them, which is why the tier
                     exists -- see WORKSPACE's own note below, and
                     agent/pipeline/workspace.py's module docstring for what
                     the confinement is and is not worth.
  web_search      -- STUBBED. No real search integration yet -- deliberate
                     scope cut (2026-09-10 design call: prove the graph
                     skeleton with Mercury-only routing and stub tools
                     first, add a real search API once that's working).
                     Always fails cleanly with a legible reason, so a
                     caller's tool-loop can fall back to its own knowledge
                     instead of crashing or silently returning nothing.
  rag             -- STUBBED, same reasoning as web_search -- no vector
                     store / knowledge base exists yet to query.
  complete_code   -- real, added later the same day once the graph skeleton
                     above was proven: Mercury's FIM endpoint
                     (Task.CODE_COMPLETE / mercury-edit-2, mapping.py),
                     unused until now because nothing in this graph had
                     called into it. Fills in code given what's already
                     written -- see its own docstring for its CODE: format
                     (an optional "---SUFFIX---" split).
  predict_edit    -- real, same day/reason as complete_code: Mercury's edit
                     endpoint (Task.CODE_EDIT / mercury-edit-2). Genuinely
                     different in kind from every other tool here -- it
                     takes NO instruction, only code (optionally with a
                     `<|cursor|>` marker) and predicts whatever edit comes
                     next from that alone. Good for "what's the obvious
                     next fix/continuation here", useless for "make this
                     specific change" -- see its own docstring.
  recall_memory   -- real, added later the same day, Phase 2 of claude/
                     otto-tiered-memory-design.md: semantic search
                     (agent/memory/retrieval.py's recall()) over THIS
                     session's own compacted-away conversation history --
                     what a caller reaches for when something from earlier
                     in a long conversation got summarized away and the
                     summary alone isn't enough. Takes a plain query
                     string, nothing else. Reads whichever MemoryStore
                     agent/pipeline/run.py bound for this run
                     (agent/memory/session.py's `current_store()`) -- None
                     bound (a caller that never wired memory in at all,
                     e.g. agent/eval/'s golden runner) is not an error,
                     same "fail clean, not crash" shape as web_search/rag.

Both new tools go through the SAME `Router` instance's `.fim()`/
`.code_edit()` (agent/router/router.py) that already served CODE_COMPLETE/
CODE_EDIT before this graph existed -- lazily constructed (`_get_router()`
below) rather than at import time, unlike nodes.py's module-level `ROUTER`:
this module is also imported by agent/eval/runner.py for its offline,
zero-network golden-checker execution (execute_python), and that path must
keep working without INCEPTION_API_KEY set at all. A `ProviderError` from
either call (no key, no viable route, a live API failure) is caught and
turned into a normal failing ToolResult, exactly like web_search/rag's
stub failure -- never a crash that takes down the whole tool loop.

This is NOT a security sandbox -- it is a subprocess with a timeout, not a
seccomp/container jail. It has no network block and no memory/CPU limit
beyond the timeout. That is an acceptable bar for code/commands this
graph's own nodes generate to check their own work, not for untrusted
input. execute_bash widens that gap versus execute_python specifically: an
arbitrary shell command reaches any binary on PATH, not just the Python
interpreter. If a mutating tool (writing a real file, calling a paid API,
sending something) is ever added, it must be registered MUTATING here, and
nothing in a role/evaluator node's reach may call it.
"""
from __future__ import annotations

import base64
import os
import re
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from agent.memory.retrieval import recall
from agent.memory.session import current_store
from agent.pipeline.execution import current_command_runner
from agent.pipeline.workspace import (
    OutsideWorkspace,
    current_workspace,
    resolve_in_workspace,
)
from agent.router.llm_provider.base import ProviderError
from agent.router.router import Router

READ_ONLY = "read_only"
MUTATING = "mutating"
#: Writes, but only ever inside the directory a caller deliberately bound as
#: this run's workspace (agent/pipeline/workspace.py), and never anywhere
#: else. Not READ_ONLY -- it changes real files, and pretending otherwise
#: would make the tier meaningless. Not MUTATING either, in the sense that
#: tier was defined for: what makes an irreversible action unsafe before the
#: evaluator has judged an attempt is that it cannot be taken back, and a
#: write into a workspace the caller created and will delete can be. A run
#: that binds no workspace -- every ordinary chat turn -- cannot reach these
#: at all, so this widens nothing for the case the tier system was protecting.
WORKSPACE = "workspace"

#: Tail length kept from stdout/stderr -- long enough to carry a real
#: traceback, short enough not to blow out a prompt on a runaway print loop.
_TAIL = 4000


@dataclass(frozen=True)
class ToolResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def execute_python(code: str, *, timeout: float = 10.0) -> ToolResult:
    """Run `code` as a standalone script in a fresh process, in the bound
    workspace if there is one and otherwise in a fresh throwaway temp dir.

    Used by every role node/the evaluator as their ACTION/execute_python
    self-check before committing to a FINAL answer or a verdict (nodes.py's
    _tool_loop) -- the same role this had in the retired swarm pipeline,
    just no longer also doubling as evaluate()'s only verification strategy
    (that domain-specific branch is gone; the evaluator now checks things
    for real via this same tool instead of a bespoke code path).
    """
    remote = current_command_runner()
    if remote is not None:
        # base64 rather than a heredoc: a heredoc is only safe until the
        # snippet contains a line equal to the delimiter, and the snippet is
        # model-written text that nothing constrains.
        stdout, stderr, code = remote(
            f"echo {_b64(code)} | base64 -d | python3 -", timeout,
        )
        return ToolResult(
            stdout=stdout[-_TAIL:], stderr=stderr[-_TAIL:],
            returncode=code, timed_out=(code == -1),
        )

    with _run_dir() as tmp, tempfile.TemporaryDirectory() as holder:
        # The script itself lives OUTSIDE the run dir, always. When the run dir
        # is a throwaway temp dir it makes no difference, but when it is a real
        # workspace, dropping a snippet.py into the repo the agent is editing
        # would show up in `git status` and in its own next listing -- a file
        # nobody wrote, indistinguishable from one the task asked for.
        script = Path(holder) / "snippet.py"
        script.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=tmp,
                env=_env_with_pythonpath(tmp),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ToolResult(
                stdout=proc.stdout[-_TAIL:],
                stderr=proc.stderr[-_TAIL:],
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=(exc.stdout or "")[-_TAIL:],
                stderr=((exc.stderr or "") + "\n[timed out]")[-_TAIL:],
                returncode=-1,
                timed_out=True,
            )


def _env_with_pythonpath(run_dir: str) -> dict[str, str]:
    """The child's environment with `run_dir` prepended to PYTHONPATH.

    Needed because the script itself lives outside the run dir (see
    execute_python), and Python seeds sys.path from the SCRIPT's directory,
    not from cwd -- so without this a snippet could no longer `import` a
    module sitting in the workspace next to it, which is most of the reason
    to run Python in a workspace at all.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{run_dir}{os.pathsep}{existing}" if existing else run_dir
    return env


@contextmanager
def _run_dir() -> Iterator[str]:
    """Where execute_bash/execute_python actually run: the bound workspace if
    there is one, otherwise a fresh temp dir deleted on the way out.

    The temp dir is the old behaviour and stays the default, because it is the
    right one for what these two were built for -- a node checking its own
    snippet, leaving nothing behind. A workspace is the opposite case on
    purpose: a task that edits files needs the NEXT command to see what the
    last one did, which a per-call temp dir can never provide.
    """
    workspace = current_workspace()
    if workspace is not None:
        yield str(workspace)
        return
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


def execute_bash(command: str, *, timeout: float = 10.0) -> ToolResult:
    """Run `command` as a shell command, in the bound workspace if there is
    one and otherwise in a fresh throwaway temp dir.

    Same timeout/capture-output shape as execute_python, but `shell=True` over
    the raw command text rather than a Python script -- see the module
    docstring's note on the wider threat-model surface this implies (any
    binary on PATH, not just the Python interpreter), and workspace.py's own
    docstring for why a shell in a workspace is a blast-radius argument
    rather than a sandbox.
    """
    remote = current_command_runner()
    if remote is not None:
        stdout, stderr, code = remote(command, timeout)
        return ToolResult(
            stdout=stdout[-_TAIL:], stderr=stderr[-_TAIL:],
            returncode=code, timed_out=(code == -1),
        )

    with _run_dir() as tmp:
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ToolResult(
                stdout=proc.stdout[-_TAIL:],
                stderr=proc.stderr[-_TAIL:],
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=(exc.stdout or "")[-_TAIL:],
                stderr=((exc.stderr or "") + "\n[timed out]")[-_TAIL:],
                returncode=-1,
                timed_out=True,
            )


#: Longest file listing / read this returns before truncating -- the same
#: "don't blow the caller's context" bar _TAIL sets for command output.
_MAX_LIST_ENTRIES = 400


def _workspace_failure(tool: str, detail: str) -> ToolResult:
    """The refusal every workspace tool gives when there is no workspace to
    act in -- an ordinary `otto chat` turn, or any caller that never opened
    one. A failed ToolResult, never a raise: same "fail clean" shape as
    web_search/rag/recall_memory, so a tool loop reads it as a normal
    unsuccessful call and can say so instead of crashing."""
    return ToolResult(stdout="", stderr=f"{tool}: {detail}", returncode=1)


def _remote_paths_are_the_containers_own(tool: str, path: str) -> ToolResult | None:
    """Reject an empty path for a remote (container) file operation.

    Deliberately the ONLY check in remote mode, where the local
    resolve_in_workspace() confinement does not apply and should not be
    imitated: the container IS the sandbox, its whole filesystem is the task's
    subject, and a task that says "fix /etc/nginx/nginx.conf" means exactly
    that. Confining to a subdirectory there would break real tasks while
    protecting nothing the container boundary does not already protect.
    """
    if not path.strip():
        return _workspace_failure(tool, "the path must not be empty")
    return None


def _b64(text: str) -> str:
    """Text as base64, for shipping into a container through a shell command
    without a quoting story -- content with quotes, backslashes, newlines or a
    line that happens to read `OTTO_EOF` all survive unchanged."""
    return base64.b64encode(text.encode()).decode()


def read_file(body: str) -> ToolResult:
    """Read a file from the bound workspace. CODE: body is the path, alone,
    optionally suffixed `:START-END` for a 1-based inclusive line range
    ("src/app.py:40-80") -- worth having because the whole point of reading
    is to spend context on the part that matters.

    Output is line-numbered, which costs a few characters and buys the two
    things that follow a read: quoting a location back, and knowing which
    lines an edit is actually replacing.
    """
    spec = body.strip()
    line_range = None
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        if head and re.fullmatch(r"\d+-\d+", tail):
            spec, line_range = head, tuple(int(n) for n in tail.split("-"))
    remote = current_command_runner()
    if remote is not None:
        if (bad := _remote_paths_are_the_containers_own("read_file", spec)) is not None:
            return bad
        sed = f"sed -n '{line_range[0]},{line_range[1]}p'" if line_range else "cat"
        start = line_range[0] if line_range else 1
        stdout, stderr, code = remote(
            f"{sed} {shlex.quote(spec)} | nl -ba -v {start} -w6 -s'\t'", 20.0,
        )
        if code != 0:
            return _workspace_failure("read_file", stderr.strip() or f"could not read {spec!r}")
        return ToolResult(stdout=stdout[-_TAIL:], stderr="", returncode=0)

    try:
        path = resolve_in_workspace(spec)
    except OutsideWorkspace as exc:
        return _workspace_failure("read_file", str(exc))
    if not path.is_file():
        return _workspace_failure("read_file", f"{spec!r} is not a file in the workspace")
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return _workspace_failure("read_file", f"could not read {spec!r}: {exc}")

    start, end = (1, len(lines)) if line_range is None else line_range
    start, end = max(1, start), min(len(lines), end)
    numbered = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1))
    return ToolResult(stdout=numbered[-_TAIL:], stderr="", returncode=0)


def write_file(body: str) -> ToolResult:
    """Create or overwrite a file in the bound workspace. CODE: body is the
    path on its OWN FIRST LINE, and everything after that first newline is
    the file's content, verbatim.

    No separator line between the two, deliberately: any delimiter that could
    be typed is a delimiter that can appear in real file content, and a
    write that silently truncates at a `---` inside a Markdown document or a
    Python docstring is a far worse failure than a slightly plainer format.
    Missing parent directories are created -- a model that writes
    "pkg/mod/x.py" means for it to exist.
    """
    head, _, content = body.partition("\n")
    remote = current_command_runner()
    if remote is not None:
        if (bad := _remote_paths_are_the_containers_own("write_file", head)) is not None:
            return bad
        target = shlex.quote(head.strip())
        stdout, stderr, code = remote(
            f"mkdir -p \"$(dirname {target})\" && "
            f"echo {_b64(content)} | base64 -d > {target}", 30.0,
        )
        if code != 0:
            return _workspace_failure("write_file", stderr.strip() or f"could not write {head.strip()!r}")
        return ToolResult(
            stdout=f"wrote {head.strip()} ({len(content.splitlines())} lines)",
            stderr="", returncode=0,
        )

    try:
        path = resolve_in_workspace(head)
    except OutsideWorkspace as exc:
        return _workspace_failure("write_file", str(exc))
    if not head.strip():
        return _workspace_failure("write_file", "first line must be the file path")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    except OSError as exc:
        return _workspace_failure("write_file", f"could not write {head.strip()!r}: {exc}")
    return ToolResult(
        stdout=f"wrote {head.strip()} ({len(content.splitlines())} lines)",
        stderr="", returncode=0,
    )


def edit_file(body: str) -> ToolResult:
    """Replace an exact snippet in a workspace file. CODE: body is

        path/to/file.py
        ---OLD---
        the exact text to replace
        ---NEW---
        what to replace it with

    The old text must appear EXACTLY ONCE. Not zero times (the model is
    editing something it misremembers, and a silently-skipped edit is the
    kind of failure that surfaces later as an inexplicable test result), and
    not several times (which of them was meant is genuinely unknown, and
    guessing is worse than saying so). Both cases come back as a failed
    ToolResult naming the count, which is a thing a tool loop can act on --
    read the file again, quote more surrounding context, retry.

    This exists alongside write_file because rewriting a whole file to change
    three lines is how an agent destroys the parts of it nobody asked about.
    """
    head, _, rest = body.partition("\n")
    remote = current_command_runner()
    if remote is not None:
        return _remote_edit(remote, head, rest)

    # Workspace first, body format second: "there is nowhere to write" is the
    # more fundamental refusal, and reporting a format complaint to a caller
    # that could never have written anything anyway just misdirects it.
    try:
        path = resolve_in_workspace(head)
    except OutsideWorkspace as exc:
        return _workspace_failure("edit_file", str(exc))
    if "---OLD---" not in rest or "---NEW---" not in rest:
        return _workspace_failure(
            "edit_file", "body must be: path, then ---OLD---, then ---NEW---",
        )
    old_part, _, new_part = rest.partition("---NEW---")
    old_text = old_part.split("---OLD---", 1)[1].strip("\n")
    new_text = new_part.strip("\n")
    if not path.is_file():
        return _workspace_failure("edit_file", f"{head.strip()!r} is not a file in the workspace")

    original = path.read_text(errors="replace")
    occurrences = original.count(old_text)
    if occurrences != 1:
        found = "never appears" if occurrences == 0 else f"appears {occurrences} times"
        return _workspace_failure(
            "edit_file",
            f"the ---OLD--- text {found} in {head.strip()} -- it must appear exactly once; "
            "read the file and quote more surrounding lines to make it unique",
        )
    path.write_text(original.replace(old_text, new_text, 1))
    return ToolResult(stdout=f"edited {head.strip()}", stderr="", returncode=0)


#: The exact-once replacement edit_file performs, as a script to run inside a
#: container. Same contract as the local branch -- refuse at zero matches and
#: refuse at several, rather than guessing -- expressed once here so the two
#: modes can't drift into disagreeing about what an edit means. Both texts
#: arrive base64-encoded, so no quoting of the model's content is involved.
_REMOTE_EDIT_SCRIPT = """
import base64, sys
path, old_b64, new_b64 = sys.argv[1], sys.argv[2], sys.argv[3]
old = base64.b64decode(old_b64).decode()
new = base64.b64decode(new_b64).decode()
try:
    original = open(path, errors="replace").read()
except OSError as exc:
    print(f"cannot read {path}: {exc}", file=sys.stderr); sys.exit(2)
count = original.count(old)
if count != 1:
    found = "never appears" if count == 0 else f"appears {count} times"
    print(f"the ---OLD--- text {found} in {path} -- it must appear exactly once; "
          "read the file and quote more surrounding lines to make it unique",
          file=sys.stderr)
    sys.exit(3)
open(path, "w").write(original.replace(old, new, 1))
print(f"edited {path}")
"""


def _remote_edit(remote, head: str, rest: str) -> ToolResult:
    """edit_file against a container, via the bound command runner."""
    if (bad := _remote_paths_are_the_containers_own("edit_file", head)) is not None:
        return bad
    if "---OLD---" not in rest or "---NEW---" not in rest:
        return _workspace_failure(
            "edit_file", "body must be: path, then ---OLD---, then ---NEW---",
        )
    old_part, _, new_part = rest.partition("---NEW---")
    old_text = old_part.split("---OLD---", 1)[1].strip("\n")
    new_text = new_part.strip("\n")
    stdout, stderr, code = remote(
        f"python3 -c {shlex.quote(_REMOTE_EDIT_SCRIPT)} "
        f"{shlex.quote(head.strip())} {_b64(old_text)} {_b64(new_text)}",
        30.0,
    )
    if code != 0:
        return _workspace_failure("edit_file", stderr.strip() or "edit failed")
    return ToolResult(stdout=stdout.strip(), stderr="", returncode=0)


def list_files(body: str) -> ToolResult:
    """List files under a workspace directory, recursively. CODE: body is the
    directory (empty means the workspace root).

    Skips the directories that are always noise and sometimes enormous --
    .git, __pycache__, .venv and friends -- because the first thing an agent
    does in an unfamiliar repo is list it, and a listing dominated by
    thousands of object files teaches it nothing while costing everything.
    """
    spec = body.strip() or "."
    remote = current_command_runner()
    if remote is not None:
        pruned = " ".join(f"-name {shlex.quote(d)} -o" for d in sorted(_LISTING_SKIP))
        stdout, stderr, code = remote(
            f"find {shlex.quote(spec)} \\( {pruned} -false \\) -prune -o -print "
            f"| head -n {_MAX_LIST_ENTRIES}", 30.0,
        )
        if code != 0:
            return _workspace_failure("list_files", stderr.strip() or f"could not list {spec!r}")
        return ToolResult(stdout=stdout[-_TAIL:] or "(empty)", stderr="", returncode=0)

    try:
        root = resolve_in_workspace(spec)
    except OutsideWorkspace as exc:
        return _workspace_failure("list_files", str(exc))
    if not root.is_dir():
        return _workspace_failure("list_files", f"{spec!r} is not a directory in the workspace")

    base = current_workspace()
    entries = []
    for path in sorted(root.rglob("*")):
        if any(part in _LISTING_SKIP for part in path.relative_to(root).parts):
            continue
        entries.append(str(path.relative_to(base)) + ("/" if path.is_dir() else ""))
        if len(entries) > _MAX_LIST_ENTRIES:
            entries.append(f"... (truncated at {_MAX_LIST_ENTRIES} entries)")
            break
    return ToolResult(stdout="\n".join(entries) or "(empty)", stderr="", returncode=0)


#: Directories a listing never descends into -- version-control internals,
#: build/dependency trees, and caches. Never source, so skipping them loses
#: an agent nothing it would have wanted to read.
_LISTING_SKIP = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".eggs", ".idea", ".vscode", "target",
})


def web_search(query: str) -> ToolResult:
    """STUB (see module docstring) -- always fails with a legible reason
    instead of a crash or a silent empty result, so a caller's tool-loop
    (nodes.py's _tool_loop) gets something to react to (fall back to its
    own knowledge, say so in its answer) rather than looking like a tool
    that ran and simply found nothing.
    """
    return ToolResult(
        stdout="",
        stderr=f"web_search is not implemented yet (query={query!r}) -- "
               f"stubbed pending a real search integration",
        returncode=1,
    )


def rag(query: str) -> ToolResult:
    """STUB (see module docstring) -- same reasoning as web_search() above;
    no knowledge base / vector store exists yet to actually query.
    """
    return ToolResult(
        stdout="",
        stderr=f"rag is not implemented yet (query={query!r}) -- "
               f"stubbed pending a real knowledge base",
        returncode=1,
    )


def recall_memory(query: str) -> ToolResult:
    """Semantic search (agent/memory/retrieval.py's recall()) over THIS
    session's own compacted-away conversation history -- kind="history",
    always; there is no "context" (in-turn) memory to search yet (module
    docstring's Phase 2 note). `query` is a plain search string, nothing
    else -- not code, not a shell command.

    Two distinct ways this can come back empty-handed, both reported as a
    normal failing ToolResult rather than a crash (same shape as web_search/
    rag's stub failure, so a caller's tool loop reacts to it the same way):
    no store bound at all for this run (agent/memory/session.py's
    current_store() returns None -- e.g. agent/eval/'s golden runner, which
    never wires memory in), or a store that IS bound but genuinely has
    nothing compacted yet (recall()'s own "nothing to recall yet" message,
    which still comes back as ok=True -- that's a real, if unhelpful,
    answer, not a failure).
    """
    store = current_store()
    if store is None:
        return ToolResult(
            stdout="",
            stderr=f"recall_memory: no session memory is bound for this run (query={query!r})",
            returncode=1,
        )
    try:
        text = recall(store, "history", query)
    except Exception as exc:  # a memory-layer bug must not crash the tool loop
        return ToolResult(stdout="", stderr=f"recall_memory failed: {exc}", returncode=1)
    return ToolResult(stdout=text, stderr="", returncode=0)


#: Lazily constructed, NOT at import time (unlike nodes.py's module-level
#: ROUTER) -- see the module docstring's note on why: this module is also
#: imported for its zero-network golden-checker path (agent/eval/runner.py's
#: execute_python), which must keep working with no INCEPTION_API_KEY set.
_router: Router | None = None


def _get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def complete_code(body: str) -> ToolResult:
    """Fill-in-the-middle via Task.CODE_COMPLETE (Mercury's real FIM
    endpoint, agent/router/mapping.py) -- routed through the same `Router`
    class every chat call goes through, just its `.fim()` method instead of
    `.chat_model()`.

    `body` is the FIM "prefix" -- the code already written, completion
    picks up from its end. To also pin what must follow the completion
    (the FIM "suffix"), put a line containing exactly "---SUFFIX---"
    between the two; most completions (finishing a function to its natural
    end, say) have no suffix at all, which is why that's the plain,
    unmarked case.
    """
    prefix, _, suffix = body.partition("\n---SUFFIX---\n")
    try:
        text = _get_router().fim(prefix, suffix)
    except ProviderError as exc:
        return ToolResult(stdout="", stderr=f"complete_code failed: {exc}", returncode=1)
    return ToolResult(stdout=text, stderr="", returncode=0)


def predict_edit(body: str) -> ToolResult:
    """Next-edit prediction via Task.CODE_EDIT (Mercury's edit endpoint).

    Genuinely different in kind from every other tool here: `body` is not
    an instruction, because this endpoint has none (agent/router/router.py's
    `code_edit()` docstring) -- it predicts whatever edit comes next purely
    from the code itself, optionally with a `<|cursor|>` marker placed
    where attention should focus (placed at the end automatically if you
    don't include one). Good for "what's the obvious next
    fix/continuation here" -- e.g. after execute_python surfaces a
    traceback, handing this the code with the cursor near the failing line.
    Useless for "make this specific change": there is nowhere to say what
    the change should be, so a role that wants an instructed edit should
    just write the new code directly instead of reaching for this tool.
    """
    try:
        text = _get_router().code_edit(body)
    except ProviderError as exc:
        return ToolResult(stdout="", stderr=f"predict_edit failed: {exc}", returncode=1)
    return ToolResult(stdout=text, stderr="", returncode=0)


TOOL_TIERS: dict[str, str] = {
    "execute_python": READ_ONLY,
    "execute_bash": READ_ONLY,
    "read_file": READ_ONLY,
    "list_files": READ_ONLY,
    "write_file": WORKSPACE,
    "edit_file": WORKSPACE,
    "web_search": READ_ONLY,
    "rag": READ_ONLY,
    "complete_code": READ_ONLY,
    "predict_edit": READ_ONLY,
    "recall_memory": READ_ONLY,
}

#: What nodes.py's _tool_loop actually calls, keyed by the tool name an
#: ACTION: line names. Every entry takes one positional string (the CODE:
#: body, whatever that means for the tool -- code, a shell command, a
#: query) and returns a ToolResult, so _tool_loop's formatting/feedback
#: logic never needs to know which tool it just called.
TOOL_DISPATCH: dict[str, Callable[[str], ToolResult]] = {
    "execute_python": execute_python,
    "execute_bash": execute_bash,
    "read_file": read_file,
    "list_files": list_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "web_search": web_search,
    "rag": rag,
    "complete_code": complete_code,
    "predict_edit": predict_edit,
    "recall_memory": recall_memory,
}
