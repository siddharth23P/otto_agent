"""The agent's tool registry, tiered by reversibility.

Every tool is registered with an explicit tier (READ_ONLY or MUTATING).
Every node in this graph (router excluded -- it only ever dispatches, see
nodes.py) may only ever be handed READ_ONLY tools, via TOOL_DISPATCH below:
a role node's or the evaluator's output has not been judged yet while it is
still iterating, and an irreversible action taken before judgment can't be
undone if the evaluator later rejects that attempt. Same reasoning the
retired swarm pipeline's identical module docstring gave for its workers.

Seven tools exist (2026-09-10, replacing the swarm pipeline's execute_python
tool as this graph's whole tool box, per the router/planner/solver/
summarizer/finder/evaluator design):

  execute_python -- real, unchanged from before.
  execute_bash    -- real, new: a generic shell command, same sandboxing
                     bar as execute_python (see its own docstring for the
                     honest limits of that bar).
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

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.memory.retrieval import recall
from agent.memory.session import current_store
from agent.router.llm_provider.base import ProviderError
from agent.router.router import Router

READ_ONLY = "read_only"
MUTATING = "mutating"

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
    """Run `code` as a standalone script in a fresh temp dir, fresh process.

    Used by every role node/the evaluator as their ACTION/execute_python
    self-check before committing to a FINAL answer or a verdict (nodes.py's
    _tool_loop) -- the same role this had in the retired swarm pipeline,
    just no longer also doubling as evaluate()'s only verification strategy
    (that domain-specific branch is gone; the evaluator now checks things
    for real via this same tool instead of a bespoke code path).
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
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


def execute_bash(command: str, *, timeout: float = 10.0) -> ToolResult:
    """Run `command` as a shell command in a fresh temp dir, fresh process.

    Same fresh-tmp-dir/timeout/capture-output shape as execute_python, but
    `shell=True` over the raw command text rather than a Python script --
    see the module docstring's note on the wider threat-model surface this
    implies (any binary on PATH, not just the Python interpreter).
    """
    with tempfile.TemporaryDirectory() as tmp:
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
    "web_search": web_search,
    "rag": rag,
    "complete_code": complete_code,
    "predict_edit": predict_edit,
    "recall_memory": recall_memory,
}
