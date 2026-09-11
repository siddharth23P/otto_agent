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
import binascii
import json
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
from agent.pipeline.vision import describe_image, sniff_media_type
from langchain_core.messages import HumanMessage
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
            stdout=_clip(stdout), stderr=_clip(stderr),
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
                stdout=_clip(proc.stdout),
                stderr=_clip(proc.stderr),
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=_clip(_as_text(exc.stdout)),
                stderr=_clip(_as_text(exc.stderr) + "\n[timed out]"),
                returncode=-1,
                timed_out=True,
            )


def _as_text(stream: "str | bytes | None") -> str:
    """A TimeoutExpired's captured output as text.

    subprocess.run(text=True) decodes what it returns normally, but the output
    hung off a TimeoutExpired can still be bytes -- so the timeout branch,
    which is exactly the branch nobody exercises until a real command hangs,
    raised TypeError instead of reporting the timeout. Found when a benchmark
    task ran a command long enough to hit it.
    """
    if stream is None:
        return ""
    return stream.decode(errors="replace") if isinstance(stream, bytes) else stream


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


#: Shell constructs that detach a command from the call that started it.
#: `&` at the end of a line backgrounds it; nohup/setsid/disown outlive the
#: shell entirely.
_DETACHING = re.compile(
    r"(?:^|\s)(?:nohup|setsid|disown)(?:\s|$)"      # explicitly detaching
    r"|(?<!&)&\s*$"                                  # trailing & on the command
    r"|(?<!&)&\s*\n",                                # trailing & on any line
    re.MULTILINE,
)

BACKGROUNDING_REFUSED = (
    "execute_bash: this command backgrounds or detaches part of its work "
    "({why}), so it would return before that work has done anything and you "
    "would be told it succeeded without ever seeing what happened. Run it in "
    "the foreground instead, so its real output comes back to you. If it is "
    "something long-running that never exits on its own, run it with a "
    "timeout or a bounded amount of work (for example `timeout 10 ...`) so it "
    "still returns something you can read."
)


def _detaching_reason(command: str) -> str | None:
    """Why `command` would run detached, or None if it runs to completion.

    Every tool call in this graph is synchronous by design (agent/pipeline/
    nodes.py's _tool_loop runs exactly one, waits for it, and feeds the result
    back), and that is the property the whole loop depends on: the model
    decides what to do next from what the last thing actually printed. A
    backgrounded command breaks it silently -- the call returns instantly with
    an empty stdout and exit 0, which reads as "it worked" for work that has
    not started. So this is refused rather than quietly allowed.
    """
    match = _DETACHING.search(command)
    if not match:
        return None
    token = match.group(0).strip()
    return "nohup/setsid/disown" if token.isalpha() else "a trailing &"


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
    if (why := _detaching_reason(command)) is not None:
        return ToolResult(
            stdout="", stderr=BACKGROUNDING_REFUSED.format(why=why), returncode=1,
        )

    remote = current_command_runner()
    if remote is not None:
        stdout, stderr, code = remote(command, timeout)
        return ToolResult(
            stdout=_clip(stdout), stderr=_clip(stderr),
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
                stdout=_clip(proc.stdout),
                stderr=_clip(proc.stderr),
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=_clip(_as_text(exc.stdout)),
                stderr=_clip(_as_text(exc.stderr) + "\n[timed out]"),
                returncode=-1,
                timed_out=True,
            )


def _clip(text: str) -> str:
    """`text` bounded to _TAIL, keeping BOTH ends when it has to cut.

    It used to keep only the tail, which throws away the most useful part of
    exactly the outputs that overflow: a compiler prints its first and most
    informative error at the top and then cascades, a test run names the
    failure before the summary, and a long `find` says what it is doing before
    it says how it ended. Losing the head means the model is told what
    happened last instead of what went wrong first.
    """
    if len(text) <= _TAIL:
        return text
    head, tail = _TAIL // 2, _TAIL - _TAIL // 2
    cut = len(text) - head - tail
    return f"{text[:head]}\n\n[... {cut} characters omitted ...]\n\n{text[-tail:]}"


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
        return ToolResult(stdout=_clip(stdout), stderr="", returncode=0)

    try:
        path = resolve_in_workspace(spec)
    except OutsideWorkspace as exc:
        return _workspace_failure("read_file", str(exc))
    if not path.is_file():
        return _workspace_failure("read_file", f"{spec!r} is not a file in the workspace")
    # Signpost, not a silent fallback. Reading an image as text returns pages
    # of line-numbered mojibake -- it burns context and teaches the agent
    # nothing, least of all that a tool exists for this. Naming that tool at
    # the moment it is needed is what makes it discoverable, and it costs no
    # network call.
    if (kind := sniff_media_type(path.read_bytes()[:16])) is not None:
        return _workspace_failure(
            "read_file",
            f"{spec!r} is a {kind} image, not text -- use view_image to look at it",
        )
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return _workspace_failure("read_file", f"could not read {spec!r}: {exc}")

    start, end = (1, len(lines)) if line_range is None else line_range
    start, end = max(1, start), min(len(lines), end)
    numbered = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(start, end + 1))
    return ToolResult(stdout=_clip(numbered), stderr="", returncode=0)


def _split_write_body(body: str) -> tuple[str, str]:
    """(path, content) from a write_file body.

    The documented form is the path on the first line and the content after
    it. A JSON object with "path"/"content" keys is also accepted, because
    that is what a model actually reaches for when it has to put two values
    in one string: observed in a Terminal-Bench transcript, where a JSON body
    was taken literally and created a file named `{`. The prompts now spell
    the real format out, so this is a safety net rather than a second
    supported format -- accepting it costs nothing and the alternative is
    writing rubbish to a plausible-looking path.
    """
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and "path" in parsed and "content" in parsed:
            return str(parsed["path"]), str(parsed["content"])
    head, _, content = body.partition("\n")
    return head, content


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
    head, content = _split_write_body(body)
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
        return ToolResult(stdout=_clip(stdout) or "(empty)", stderr="", returncode=0)

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
    return ToolResult(stdout=_clip("\n".join(entries)) or "(empty)", stderr="", returncode=0)


#: Directories a listing never descends into -- version-control internals,
#: build/dependency trees, and caches. Never source, so skipping them loses
#: an agent nothing it would have wanted to read.
_LISTING_SKIP = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".eggs", ".idea", ".vscode", "target",
})


#: Ceiling on an image Otto will ship to a vision model, overridable because
#: the right number depends on the vendor and the wallet rather than on
#: anything this file knows. Over it, the agent is told it can downscale the
#: file itself with a shell command -- deliberately no Pillow dependency, and
#: in a container the file is not on this machine to resize anyway.
MAX_IMAGE_BYTES = int(os.environ.get("OTTO_MAX_IMAGE_BYTES", 8 * 1024 * 1024))

#: Reads the file and base64s it in one command, size-checked first so an
#: enormous file is refused rather than transferred. `base64 < file` rather
#: than `base64 -w0 file`: `-w` is GNU coreutils and missing on BusyBox and
#: BSD, while reading stdin and stripping newlines works everywhere.
_REMOTE_IMAGE_SCRIPT = (
    'sz=$(wc -c < {path}) || exit 2; '
    '[ "$sz" -le {cap} ] || {{ echo "too big: $sz bytes" >&2; exit 3; }}; '
    "base64 < {path} | tr -d '\\n'"
)


def _translated(llm, exc: Exception) -> ProviderError:
    """A vendor SDK exception as one of Otto's own, recording a permanent
    unusability against the right catalogue on the way through.

    The tools that call `.invoke()` directly bypass nodes.py's `_call`, which
    is where this translation normally happens, so they would otherwise let a
    raw `google.genai` or `anthropic` error escape with no vendor attached.
    """
    from agent.router.llm_provider.base import translate_unknown

    provider = getattr(llm, "_otto_provider", "") if llm is not None else ""
    model_id = ""
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None) if llm is not None else None
        if isinstance(value, str):
            model_id = value
            break
    return translate_unknown(exc, provider=provider, model_id=model_id)



def view_image(body: str) -> ToolResult:
    """Look at an image file and answer a question about it.

    CODE: body is the path on the FIRST line, and the question after it:

        fixtures/Canon1.png
        transcribe the top staff bar by bar, with pitch and duration

    The question matters more than it looks. A vision model returns words, and
    those words are all the reasoning model ever sees -- for "reproduce this
    score as SVG" a generic caption is worthless, while a narrow question
    answered twice is close to being able to look. Ask again to narrow; the
    repeat detector in agent/pipeline/nodes.py knows this tool is different and
    keys on the question too, not just the path.

    The image never enters the conversation -- see agent/pipeline/vision.py for
    why that is a deliberate ceiling. One consequence worth knowing: a
    description can be summarised away by memory compaction, and the recovery
    is simply to call this again, since the file is still on disk.
    """
    head, _, question = body.partition("\n")
    path = head.strip()
    if not path:
        return _workspace_failure("view_image", "first line must be the image path")

    remote = current_command_runner()
    if remote is not None:
        command = _REMOTE_IMAGE_SCRIPT.format(path=shlex.quote(path), cap=MAX_IMAGE_BYTES)
        stdout, stderr, code = remote(command, 60.0)
        if code != 0:
            return _workspace_failure("view_image", stderr.strip() or f"could not read {path!r}")
        try:
            # Never _clip this: clipping base64 yields silently corrupt image
            # bytes, which surface as a baffling answer rather than an error.
            data = base64.b64decode(stdout.strip(), validate=True)
        except (ValueError, binascii.Error) as exc:
            return _workspace_failure("view_image", f"{path!r} did not transfer cleanly: {exc}")
    else:
        try:
            resolved = resolve_in_workspace(path)
        except OutsideWorkspace as exc:
            return _workspace_failure("view_image", str(exc))
        if not resolved.is_file():
            return _workspace_failure("view_image", f"{path!r} is not a file in the workspace")
        data = resolved.read_bytes()
        if len(data) > MAX_IMAGE_BYTES:
            return _workspace_failure(
                "view_image",
                f"{path!r} is {len(data)} bytes, over the {MAX_IMAGE_BYTES}-byte limit -- "
                "downscale it first (for example with a shell command) and try again",
            )

    media_type = sniff_media_type(data)
    if media_type is None:
        return _workspace_failure(
            "view_image", f"{path!r} is not an image this can read (checked its first bytes)",
        )

    llm = None
    try:
        # Imported here, not at module level, for the same reason _get_router()
        # is lazy: agent/eval/runner.py imports this module for offline golden
        # checking with no keys set, and that path must keep working.
        from agent.router.mapping import Task

        llm = _get_router().chat_model(Task.VISION)
        answer = describe_image(
            llm, base64.b64encode(data).decode(), media_type, question.strip(),
        )
    except ProviderError as exc:
        # No GEMINI_API_KEY means Task.VISION has no viable route, and that
        # arrives here as a ProviderError. It must degrade like any other
        # failing tool, never take down a benchmark run.
        return _workspace_failure("view_image", f"could not look at {path!r}: {exc}")
    except Exception as exc:
        # This path calls .invoke() directly, so it meets the vendor's own SDK
        # exceptions rather than anything Otto owns -- translated here for the
        # same reason nodes.py's _call translates them, and so a model the
        # vendor says is permanently unusable is remembered rather than picked
        # again by the next capability fallback in this run.
        return _workspace_failure(
            "view_image",
            f"could not look at {path!r}: {_translated(llm, exc)}",
        )

    header = (
        f"view_image: a vision model looked at {path} "
        f"({media_type}, {len(data)} bytes) and reports:"
    )
    return ToolResult(stdout=_clip(f"{header}\n{answer}"), stderr="", returncode=0)


#: Anthropic's server-side web search runs on their infrastructure and returns
#: its results as content blocks in the same response, which is why this fits
#: Otto's tool contract exactly: a query string in, text out, no change to the
#: ACTION/CODE protocol and no search-API key for Otto to hold.
#:
#: Two variants exist and the newer one is NOT a superset: `web_search_20260209`
#: (dynamic filtering) requires Claude 4.6 or later and is rejected with a 400
#: by the cheap tier this routes to, while `web_search_20250305` is accepted
#: everywhere. Verified against the live API rather than assumed. Raise the
#: WEB route's pin past 4.6 and this should move with it.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}


def web_search(query: str) -> ToolResult:
    """Search the live web and return what the search found, with sources.

    CODE: body is the query, in plain words -- not a URL and not a shell
    command. The model may decide a query needs no search (asked for a fact it
    already knows, it will just answer); that is a successful result, not a
    failure.

    Unconfigured is a clean failure, not a crash: with no ANTHROPIC_API_KEY the
    WEB route has no viable candidate and no fallback -- deliberately, since
    falling through to a model with no web access would answer from memory
    while looking like a search -- and that arrives here as a ProviderError.
    """
    if not query.strip():
        return ToolResult(stdout="", stderr="web_search: the query must not be empty", returncode=1)
    try:
        from agent.router.mapping import Task

        llm = _get_router().chat_model(Task.WEB).bind_tools([WEB_SEARCH_TOOL])
        reply = llm.invoke([HumanMessage(
            f"Search the web and answer this, citing the sources you used:\n{query}"
        )])
    except ProviderError as exc:
        return ToolResult(stdout="", stderr=f"web_search failed: {exc}", returncode=1)
    except Exception as exc:  # raw vendor SDK error -- see _translated
        return ToolResult(
            stdout="", stderr=f"web_search failed: {_translated(None, exc)}", returncode=1,
        )

    text = _blocks_to_text(reply.content)
    if not text.strip():
        return ToolResult(
            stdout="", stderr=f"web_search returned nothing for {query!r}", returncode=1,
        )
    return ToolResult(stdout=_clip(text), stderr="", returncode=0)


def _blocks_to_text(content) -> str:
    """The readable text of a reply that may be a plain string or a list of
    blocks. A web-search reply interleaves `server_tool_use` and
    `web_search_tool_result` blocks with the text ones; only the text is worth
    handing back, since the search results are already summarised into it."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


#: One indexed corpus per (workspace, fingerprint). Re-indexing on every query
#: would re-embed the whole tree each time; keying on the fingerprint means an
#: edited file is picked up on the next query and an unchanged tree is not.
_RAG_INDEXES: dict[tuple[str, str], "MemoryStore"] = {}


def rag(query: str) -> ToolResult:
    """Search the CONTENTS of the files in this workspace, semantically.

    CODE: body is a plain question, not a path and not a shell command --
    "where is the retry budget configured", not "grep -r retry".

    Distinct from `recall_memory`, which searches what this CONVERSATION said
    earlier and then compacted away. This searches what is WRITTEN IN THE
    FILES. When you want a specific string, `execute_bash` with grep is faster
    and exact; reach for this when you do not know the word the code uses.

    Requires a bound workspace, and returns a clean failure without one.
    """
    if not query.strip():
        return ToolResult(stdout="", stderr="rag: the query must not be empty", returncode=1)
    root = current_workspace()
    if root is None:
        return ToolResult(
            stdout="",
            stderr="rag: no workspace is bound for this run, so there are no files to search",
            returncode=1,
        )
    try:
        from agent.memory.retrieval import recall
        from agent.memory.store import MemoryStore
        from agent.pipeline.rag import corpus_fingerprint, index_corpus

        key = (str(root), corpus_fingerprint(root))
        store = _RAG_INDEXES.get(key)
        if store is None:
            store = MemoryStore(Path(tempfile.gettempdir()) / f"otto-rag-{key[1]}.db")
            if index_corpus(store, "corpus", root) == 0:
                return ToolResult(
                    stdout="",
                    stderr=f"rag: no indexable text files under {root}",
                    returncode=1,
                )
            _RAG_INDEXES[key] = store
        return ToolResult(stdout=_clip(recall(store, "corpus", query)), stderr="", returncode=0)
    except Exception as exc:  # indexing or embedding trouble must not crash the loop
        return ToolResult(stdout="", stderr=f"rag failed: {exc}", returncode=1)


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
        # purpose="acting": this is asked mid-task, while the agent is doing
        # something, which is the read that measurably wants to be narrow --
        # see agent/memory/retrieval.py's PROCEDURAL_TOP_K.
        text = recall(store, "history", query, purpose="acting")
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
    "view_image": READ_ONLY,
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
    "view_image": view_image,
    "list_files": list_files,
    "write_file": write_file,
    "edit_file": edit_file,
    "web_search": web_search,
    "rag": rag,
    "complete_code": complete_code,
    "predict_edit": predict_edit,
    "recall_memory": recall_memory,
}
