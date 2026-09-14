"""Otto as a library: the surface an embedding host is allowed to depend on.

`otto tui` and `otto chat` are one kind of host, a terminal. An Android app
with Otto's Python inside it (the phone helper) is another, and it has none of
what the terminal front ends assume: no repository `.env`, no `HOME`, no
shell for `execute_bash`, no Rich console to render into. What it has is a
thread it can run a turn on, a callback it can receive events through, and a
set of tools of its own to bind.

This module is that contract, and it is deliberately narrow. The pipeline's
internals (agent/pipeline/nodes.py, the graph, the prompts) are not the API;
these names are:

    API_VERSION                          bumped on any incompatible change here
    configure(home, env_file=, environ=) where state and keys live -- FIRST
    set_key / key_status / ready / doctor / version
    Runtime.open_session / list_sessions / delete_session / transcript
    SessionHandle.run / answer / cancel / close

and, beside them, agent/phone/ (the phone tools) and agent/pipeline/toolkit.py's
`ExtraTool`. A host pins the otto release it was tested against and reads
`API_VERSION` before trusting anything else.

ORDER MATTERS ONCE. `configure()` sets OTTO_HOME and OTTO_ENV_FILE
(agent/config/home.py), which the state modules read when they are imported;
it therefore has to run before `agent.memory`, `agent.router` or
`agent.pipeline` are imported, and this module imports nothing from them at
module level so that a host importing `agent.embed` first has not already lost.
`Runtime` is what triggers the heavy import, on purpose after Setup: the
pipeline builds its router at import, and a setup screen should not pay for
that just to ask which keys are missing.

EVENTS, NOT WIDGETS. `SessionHandle.run` blocks on the calling thread and
reports through a callback with plain dicts -- JSON-serialisable, so a Kotlin
or Swift host can hand them across a bridge unchanged:

    {"type": "progress", "kind", "text", "calls", "elapsed", "partial", "detail"}
    {"type": "board",    "node", "lines": [...], "output": str | None}
    {"type": "ask",      "thread_id", "question", "choices": [...]}
    {"type": "final",    "text", "usage": {...}, "trace_id"}
    {"type": "error",    "code": "provider" | "cancelled" | "failed", "message"}

An `ask` pauses the run inside `run()` until `answer()` arrives from another
thread (the UI's), exactly as agent/cli/tui.py's worker thread blocks on its
modal; `cancel()` releases that wait too, so a cancelled run never hangs on a
question nobody will answer.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agent.config import home as _home

if TYPE_CHECKING:  # pragma: no cover
    from agent.pipeline.toolkit import ExtraTool

logger = logging.getLogger(__name__)

#: The version of THIS contract. A host compares it with the highest it
#: understands and refuses loudly above that, rather than guessing.
API_VERSION = 1

#: The one key Otto cannot run without, and the three that each unlock seats.
#: Static rather than read off the provider classes so that asking which keys
#: are missing never imports a vendor SDK (tests/test_embed.py keeps the two
#: in step).
KEY_VARS: tuple[str, ...] = (
    "INCEPTION_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
)

Events = Callable[[dict[str, Any]], None]

_configured: dict[str, Any] = {}
_STATE_MODULES = ("agent.memory.store", "agent.memory.sessions", "agent.router.outcomes",
                  "agent.config.envfile")


def configure(home: str | os.PathLike, *, env_file: str | os.PathLike | None = None,
              environ: dict[str, str] | None = None) -> Path:
    """Say where Otto's state lives, and where its keys come from.

    `home` becomes OTTO_HOME (created if missing). `env_file` becomes
    OTTO_ENV_FILE and is loaded; `set_key` writes to it. `environ` is the
    alternative for a host that keeps keys somewhere safer than a file (an
    app's keystore): they are put into the process environment and never
    written anywhere by Otto, and `set_key` then updates the environment
    only. Both may be given.

    KEYS ARE PROCESS-WIDE. Either way they end up in `os.environ`, which is
    where the router and the vendor SDKs read them, and the environment is
    the process: a second Runtime or a second server connection in the same
    process sees every configured key. One process is one person's Otto;
    a host that serves several people runs several processes.

    Idempotent for the same home; a second call with a different one is a
    mistake and raises. Called after a state module was already imported it
    still sets the variables but logs that the imported module resolved its
    paths before this ran -- the contract in agent/config/home.py.
    """
    root = Path(home).expanduser().resolve()
    if _configured and _configured["home"] != root:
        raise RuntimeError(f"otto is already configured for {_configured['home']}, not {root}")
    try:
        before = _home.otto_home().resolve()
    except RuntimeError:
        before = None
    root.mkdir(parents=True, exist_ok=True)
    os.environ[_home.HOME_ENV] = str(root)
    env_path: Path | None = None
    if env_file is not None:
        env_path = Path(env_file).expanduser().resolve()
        os.environ[_home.ENV_FILE_ENV] = str(env_path)
        from dotenv import load_dotenv

        if env_path.is_file():
            load_dotenv(env_path)
    if environ:
        for name, value in environ.items():
            if value:
                os.environ[name] = value
    if not _configured and before != root:
        late = [m for m in _STATE_MODULES if m in sys.modules]
        if late:
            logger.warning("configure() ran after %s were imported; their paths were "
                           "resolved from the environment at that time", ", ".join(late))
    _configured.update(home=root, env_file=env_path)
    from agent.router import overrides

    overrides.apply_at_startup()
    return root


def _require_configured() -> None:
    if not _configured:
        raise RuntimeError("call agent.embed.configure(home) first")


def key_status() -> dict[str, str]:
    """Each vendor key, masked or "not set". Never the value."""
    from agent.config.envfile import masked

    return {name: masked(os.environ.get(name)) for name in KEY_VARS}


def set_key(name: str, value: str) -> str:
    """Set or clear one key, then make every live router and the embedding
    backend see it. Returns the masked value, the only form to display."""
    _require_configured()
    from agent.config import envfile

    if not envfile.KEY_RE.match(name or ""):
        raise ValueError(f"{name!r} is not a valid environment variable name")
    env_path = _configured.get("env_file")
    if env_path is not None:
        shown = envfile.set_value(name, value, path=env_path)
    else:
        value = (value or "").strip()
        if value:
            os.environ[name] = value
        else:
            os.environ.pop(name, None)
        shown = envfile.masked(value or None)
    from agent.memory import embeddings
    from agent.router.reload import reload_everything

    reload_everything()
    embeddings.reset_backend()
    return shown


def ready() -> bool:
    """Whether the one provider Otto cannot run without is configured."""
    from agent.router.router import Router

    return Router().ready()


def doctor() -> list[dict[str, Any]]:
    """One row per provider, as plain data: a real call to each vendor."""
    from agent.router.llm_provider import health_report

    return [
        {"provider": r.provider, "status": r.status.value, "models": r.model_count or 0,
         "detail": r.detail or ""}
        for r in health_report()
    ]


def version() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version as _version

    try:
        otto = _version("otto-cli-agent")
    except PackageNotFoundError:
        otto = "dev"
    return {"otto": otto, "api": API_VERSION,
            "python": ".".join(str(p) for p in sys.version_info[:3])}


# --------------------------------------------------------------------------
# Sessions and turns
# --------------------------------------------------------------------------

#: How often a streamed partial answer is forwarded. A diffusing route emits
#: whole-reply refinements many times a second; a UI redrawing on each is
#: what makes a phone stutter.
PARTIAL_INTERVAL_S = 0.25


class Runtime:
    """The heavy half: constructing one imports the pipeline."""

    def __init__(self) -> None:
        _require_configured()
        from agent.cli.context import AppContext

        self._ctx = AppContext()

    def open_session(self, ref: str | None = None) -> "SessionHandle":
        """A fresh session, or a saved one by id, unique prefix or "last"
        (LookupError with the reason, as `otto sessions` prints it)."""
        from agent.cli.shell import Session

        session = Session(ctx=self._ctx, workspace=None)
        if ref is not None:
            session.load(ref)
        return SessionHandle(session)

    def list_sessions(self, limit: int | None = 20) -> list[dict[str, Any]]:
        from agent.memory import sessions as index

        return [_session_row(info) for info in index.list_sessions(limit=limit)]

    def delete_session(self, session_id: str) -> bool:
        """Close the handle first if it is open: SQLite on Windows will not
        delete a file a connection still holds."""
        from agent.memory import sessions as index

        return index.delete(session_id)

    def transcript(self, ref: str) -> dict[str, Any]:
        """What a resumed session would show: the compacted earlier part as
        text, and the recent turns as messages."""
        from langchain_core.messages import HumanMessage

        handle = self.open_session(ref)
        try:
            earlier, messages = handle._session.transcript()
            return {
                "id": handle.id, "title": handle.title, "turns": handle.turns,
                "earlier": earlier,
                "messages": [{"role": "you" if isinstance(m, HumanMessage) else "otto",
                              "text": str(m.content)} for m in messages],
            }
        finally:
            handle.close()


def _session_row(info) -> dict[str, Any]:
    from agent.memory.sessions import describe_age

    return {"id": info.id, "short_id": info.short_id, "title": info.label,
            "workspace": info.workspace, "turns": info.turns,
            "created_at": info.created_at, "last_active_at": info.last_active_at,
            "age": describe_age(info.last_active_at)}


class SessionHandle:
    """One conversation. `run` blocks; `answer` and `cancel` come from any
    other thread."""

    def __init__(self, session) -> None:
        from agent.pipeline.usage import UsageLedger

        self._session = session
        self.usage = UsageLedger()
        self._lock = threading.Lock()
        self._running = False
        self._cancel: threading.Event | None = None
        self._pending_thread: str | None = None
        self._answer_ready = threading.Event()
        self._answer: str | None = None

    @property
    def id(self) -> str:
        return self._session.session_id

    @property
    def title(self) -> str:
        return self._session.title

    @property
    def turns(self) -> int:
        return self._session.turn

    @property
    def running(self) -> bool:
        return self._running

    def run(self, text: str, *, events: Events, tools: Sequence["ExtraTool"] = (),
            guidance: str = "", disabled_tools: Collection[str] = (),
            cancel: threading.Event | None = None) -> None:
        """One turn, to completion, on this thread.

        Every run-scoped binding happens here, on the thread that consumes
        the stream, because that is the only thread where a contextvar is
        visible to the graph. `tools` and `guidance` go to
        agent/pipeline/toolkit.py, `disabled_tools` to agent/pipeline/profile.py.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("a turn is already running in this session")
            self._running = True
            self._cancel = cancel or threading.Event()
            self._pending_thread = None
            self._answer_ready.clear()
        try:
            self._drive(text, events, tools, guidance, disabled_tools)
        finally:
            with self._lock:
                self._running = False
                self._pending_thread = None

    def answer(self, thread_id: str, text: str) -> bool:
        """Answer the question a running turn is waiting on. False when no
        question with that thread id is pending."""
        with self._lock:
            if self._pending_thread != thread_id:
                return False
            self._answer = text
            self._answer_ready.set()
            return True

    def cancel(self) -> None:
        """Stop the running turn within one model call, or at once if it is
        waiting on an answer."""
        with self._lock:
            if self._cancel is not None:
                self._cancel.set()
            self._answer_ready.set()

    def close(self) -> None:
        self._session.close()

    # -- the loop ----------------------------------------------------------

    def _drive(self, text: str, events: Events, tools, guidance: str, disabled) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from agent.pipeline import run as pipeline
        from agent.pipeline.profile import bind_tool_profile
        from agent.pipeline.progress import Cancelled, bind_progress
        from agent.pipeline.toolkit import bind_extra_tools
        from agent.router.llm_provider.base import AuthError, ProviderError

        last_partial = 0.0

        def sink(update) -> None:
            nonlocal last_partial
            if update.kind == "partial":
                now = time.monotonic()
                if now - last_partial < PARTIAL_INTERVAL_S:
                    return
                last_partial = now
            events({"type": "progress", "kind": update.kind, "text": update.text,
                    "calls": update.calls, "elapsed": round(update.elapsed, 2),
                    "partial": update.partial, "detail": _plain(update.detail)})

        human = HumanMessage(text)
        history, memory_context = self._session.history_for_graph()
        session_id = self._session.session_id
        final = None
        try:
            with bind_progress(sink, cancel=self._cancel), \
                 bind_extra_tools(list(tools), guidance=guidance), \
                 bind_tool_profile(disabled):
                stream = pipeline.run_pipeline_stream(
                    text, session_id=session_id, history=history,
                    memory_context=memory_context, workspace=None, usage=self.usage,
                )
                while stream is not None:
                    next_stream = None
                    for update in stream:
                        if "__ask__" in update:
                            ask = update["__ask__"]
                            reply = self._wait_for_answer(ask, events)
                            # Unwind the suspended generator on THIS thread,
                            # where its contextvar tokens are valid
                            # (agent/cli/chat.py says why).
                            stream.close()
                            next_stream = pipeline.resume_pipeline_stream(
                                reply, thread_id=ask["thread_id"], session_id=session_id,
                                workspace=None, usage=self.usage,
                            )
                            break
                        if "__final__" in update:
                            final = update["__final__"]
                            self._session.trace_id = update.get("__trace_id__")
                            continue
                        node, delta = next(iter(update.items()))
                        if isinstance(delta, dict):
                            events({"type": "board", "node": str(node),
                                    "lines": [str(line) for line in delta.get("board", []) or []],
                                    "output": delta.get("output") or None})
                    stream = next_stream
        except Cancelled:
            events({"type": "error", "code": "cancelled", "message": "stopped"})
            return
        except (AuthError, ProviderError) as exc:
            events({"type": "error", "code": "provider", "message": str(exc)})
            return
        except Exception as exc:  # a provider's own exception must not kill the host
            logger.exception("turn failed")
            events({"type": "error", "code": "failed", "message": f"{type(exc).__name__}: {exc}"})
            return

        raw = ((final or {}).get("final_output") or "").strip()
        self._session.record_turn(human, AIMessage(raw) if raw else None)
        events({"type": "final", "text": raw, "usage": self.usage.snapshot(),
                "trace_id": self._session.trace_id})

    def _wait_for_answer(self, ask: dict, events: Events) -> str:
        from agent.pipeline.progress import Cancelled

        with self._lock:
            self._pending_thread = ask["thread_id"]
            self._answer = None
            self._answer_ready.clear()
        events({"type": "ask", "thread_id": ask["thread_id"],
                "question": ask.get("question", ""), "choices": list(ask.get("choices") or [])})
        self._answer_ready.wait()
        with self._lock:
            self._pending_thread = None
            if self._cancel is not None and self._cancel.is_set():
                raise Cancelled("stopped while waiting for an answer")
            return self._answer or ""


def _plain(detail) -> dict[str, Any] | None:
    """A progress detail as JSON-safe data."""
    if not detail:
        return None
    return {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
            for k, v in dict(detail).items()}
