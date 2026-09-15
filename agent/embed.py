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
    setup_status / probe / doctor_report / models
    routing / routing_options / set_pin / clear_pin
    lessons / delete_lesson / clear_lessons / notes / note / delete_note
    Runtime.open_session / list_sessions / delete_session / transcript /
            rename_session / export_session / import_session
    SessionHandle.run / answer / cancel / rename / usage_report / close

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
        kind "phase" with detail {"phone": bool} says where a turn with the
        phone tools runs: "working on your phone" or "answering here"
    {"type": "board",    "node", "lines": [...], "output": str | None}
    {"type": "ask",      "thread_id", "question", "choices": [...]}
    {"type": "final",    "text", "usage": {...}, "trace_id", "turn", "title", "turns",
                         "phone", "document"}
    {"type": "error",    "code": "provider" | "cancelled" | "failed", "message", "turn"}

`turn` is what this turn alone spent, {"tokens", "calls", "cost"} (cost None
when a model that answered has no rate); `usage` stays the session's running
total. `phone` is where the turn ran (None: it had no phone tools). `document`
is None, or the Markdown document a research turn wrote:
{"path" (workspace-relative), "format": "md", "markdown" (at most
DOCUMENT_MAX_BYTES), "truncated", "files": ["document.md", "document.docx", ...]}.

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
from collections.abc import Callable, Collection, Mapping, Sequence
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

#: The standing tools that start a subprocess. A subprocess inherits the
#: whole environment, keys included, so a host that injected keys through
#: `configure(environ=...)` gets these switched off unless it says otherwise
#: (`SessionHandle.run`'s `disabled_tools`).
SUBPROCESS_TOOLS: tuple[str, ...] = ("execute_bash", "execute_python")

#: Whether a turn that was handed the phone tools asks a model first if it
#: needs the phone at all (agent/pipeline/nodes.py `needs_phone`). A module
#: switch so a test suite can keep its turns free of that call
#: (tests/conftest.py); a host says per turn with `run(phone=...)`.
DECIDE_PHONE = True


class _Auto:
    def __repr__(self) -> str:
        return "AUTO"


#: `run(workspace=AUTO)`: no workspace on the phone, a directory of the
#: session's own under `<OTTO_HOME>/workspaces/` everywhere else.
AUTO: Any = _Auto()

_configured: dict[str, Any] = {}
_STATE_MODULES = ("agent.memory.store", "agent.memory.sessions", "agent.router.outcomes",
                  "agent.config.envfile", "agent.cli.output")


def configure(home: str | os.PathLike, *, env_file: str | os.PathLike | None = None,
              environ: dict[str, str] | None = None, strict: bool = False) -> Path:
    """Say where Otto's state lives, and where its keys come from.

    `home` becomes OTTO_HOME (created if missing). `env_file` becomes
    OTTO_ENV_FILE and is loaded; `set_key` writes to it. `environ` is the
    alternative for a host that keeps keys somewhere safer than a file (an
    app's keystore): they are put into the process environment and never
    written anywhere by Otto, and `set_key` then updates the environment
    only. Both may be given. Naming either source also clears the vendor
    keys the process already had, so the keys in play are the ones the
    caller supplied; `configure(home)` alone leaves the environment as it is.

    KEYS ARE PROCESS-WIDE. Either way they end up in `os.environ`, which is
    where the router and the vendor SDKs read them, and the environment is
    the process: a second Runtime or a second server connection in the same
    process sees every configured key. One process is one person's Otto;
    a host that serves several people runs several processes.

    Idempotent for the same home, env file included: a repeat call that omits
    `env_file` keeps the one already configured. A second call with a
    different home is a mistake and raises. Called after a state module was
    already imported it still sets the variables but that module resolved
    its paths before this ran (the contract in agent/config/home.py): with
    `strict` that raises, otherwise it is logged and `late_imports()` names
    the modules, so a host can assert on it.

    OTTO_OUTPUT_DIR, where the export tools write files a person opens, is
    set to `<home>/output` unless the host set it already: an embedded
    process has no working directory worth writing into.
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
    os.environ.setdefault("OTTO_OUTPUT_DIR", str(root / "output"))
    env_path: Path | None = None
    if env_file is not None or environ:
        # The caller named where keys come from, so the keys are those and
        # not whatever the shell happened to export: an ambient key that won
        # over a host's keystore would be a key the host never provided.
        for name in KEY_VARS:
            os.environ.pop(name, None)
    if env_file is not None:
        env_path = Path(env_file).expanduser().resolve()
        os.environ[_home.ENV_FILE_ENV] = str(env_path)
        from dotenv import load_dotenv

        if env_path.is_file():
            load_dotenv(env_path, override=True)
    if environ:
        for name, value in environ.items():
            if value:
                os.environ[name] = value
    late: list[str] = []
    if not _configured and before != root:
        late = [m for m in _STATE_MODULES if m in sys.modules]
        if late and strict:
            raise RuntimeError(f"configure() ran after {', '.join(late)} were imported; "
                               "call it before importing agent.*")
        if late:
            logger.warning("configure() ran after %s were imported; their paths were "
                           "resolved from the environment at that time", ", ".join(late))
    if env_path is None:
        env_path = _configured.get("env_file")  # sticky: a repeat call without it keeps the file
    _configured.update(home=root, env_file=env_path, late=late or _configured.get("late", []),
                       keys_from_host=bool(environ) or _configured.get("keys_from_host", False))
    from agent.router import overrides

    overrides.apply_at_startup()
    return root


def late_imports() -> list[str]:
    """The state modules that were imported before configure() ran, and so
    resolved their paths without it. Empty is the healthy answer."""
    return list(_configured.get("late") or [])


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


def setup_status() -> dict[str, Any]:
    """What a setup screen opens on, as data: whether Otto can run, every
    vendor key masked, one row per vendor (agent/router/setup.py
    `vendor_rows`: names, masks and presence, never a value) and the
    version."""
    from dataclasses import asdict

    from agent.router import setup as router_setup

    return {"ready": ready(), "keys": key_status(),
            "vendors": [asdict(row) for row in router_setup.vendor_rows()], "version": version()}


def probe(name: str) -> dict[str, Any]:
    """One vendor, checked with a real call, and the models it lists.
    ValueError for a name that is not a configured vendor row."""
    from agent.router import setup as router_setup

    if name not in {row.name for row in router_setup.vendor_rows()}:
        raise ValueError(f"{str(name)[:40]!r} is not a vendor otto knows")
    result = router_setup.probe(name)
    report = result.report
    return {"name": name, "ok": result.ok, "status": report.status.value, "detail": report.detail or "",
            "model_count": report.model_count or len(result.models),
            "models": [_model_row(m) for m in result.models]}


def doctor_report() -> dict[str, Any]:
    """`otto doctor` as data: every provider checked with a real call
    (`doctor`), and the conclusion the TUI prints under it -- whether the
    required provider is configured and which others are."""
    from agent.router.router import Router

    router = Router()
    return {"providers": doctor(), "ready": router.ready(), "required": Router.REQUIRED,
            "also_configured": [p for p in router.usable() if p != Router.REQUIRED]}


def models() -> list[dict[str, Any]]:
    """Every model every configured vendor lists, as `otto models --json`
    shows them plus the spec a pin is written with. Network."""
    from agent.router.llm_provider import all_models

    return [_model_row(m) for m in sorted(all_models(None), key=lambda m: (m.provider, m.id))]


def routing() -> list[dict[str, Any]]:
    """One row per task, as the TUI's mapping tab and sidebar read it: the
    pin routes.json holds (None: the shipped route), the shipped head
    (a model spec, or the vendor a query is scoped to), the vendor a task is
    bound to whatever the pin and why, and the seat a phone turn binds over
    it (agent/phone PHONE_SEATS)."""
    from agent.phone import PHONE_SEATS
    from agent.router import overrides
    from agent.router.mapping import Task

    pins = overrides.pins()
    rows = []
    for task in Task:
        head = overrides.shipped(task)[0]
        bound = overrides.PROVIDER_ONLY.get(task)
        rows.append({"task": task.value, "pin": pins.get(task), "default": head.spec or head.provider_name,
                     "provider_only": {"provider": bound[0], "reason": bound[1]} if bound else None,
                     "phone_seat": PHONE_SEATS.get(task.value)})
    return rows


def _task(name: Any):
    from agent.router.mapping import Task

    try:
        return Task(str(name))
    except ValueError:
        raise ValueError(f"{str(name)[:40]!r} is not a task; one of {', '.join(t.value for t in Task)}") from None


def routing_options(task: str) -> dict[str, Any]:
    """The pins one task could take -- agent/router/setup.py `pin_options`
    over every model the configured vendors list, exactly the TUI's picker.
    Network. ValueError for a name that is not a task."""
    from agent.router import overrides
    from agent.router.llm_provider import all_models
    from agent.router.mapping import TASK_ROUTES
    from agent.router.setup import pin_options

    seat = _task(task)
    found = sorted(all_models(None), key=lambda m: (m.provider, m.id))
    return {"task": seat.value, "pin": overrides.pins().get(seat),
            "options": [{"label": label, "spec": spec} for label, spec in pin_options(seat, found, TASK_ROUTES)]}


def set_pin(task: str, spec: str) -> list[str]:
    """Pin `spec` on `task` in routes.json and make every live router see it.
    agent/router/overrides.py's PinError (a ValueError) for a pin that can be
    seen to be wrong; returns the problems re-applying the table reported."""
    from agent.router import overrides
    from agent.router.reload import reload_everything

    overrides.set_pin(_task(task), str(spec).strip())
    return reload_everything()


def clear_pin(task: str) -> list[str]:
    from agent.router import overrides
    from agent.router.reload import reload_everything

    overrides.clear_pin(_task(task))
    return reload_everything()


def _checked_kind(kind: Any) -> str:
    from agent.memory import lessons as L

    if not L.valid_kind(kind):
        raise ValueError("kind is lesson, phone_lesson or app_note:<package>")
    return kind


def lessons(kind: str) -> dict[str, Any]:
    """What the bank holds under `kind` (agent/memory/lessons.py
    `list_kind`): {"kind", "lessons": [{"lesson_id", "cue", "action",
    "outcome", "text"}]}. ValueError for a kind a host may not name."""
    from agent.memory import lessons as L

    kind = _checked_kind(kind)
    with L.bank_session():
        return {"kind": kind, "lessons": L.list_kind(kind)}


def delete_lesson(kind: str, lesson_id: str) -> bool:
    from agent.memory import lessons as L

    kind = _checked_kind(kind)
    if not L.valid_lesson_id(lesson_id):
        raise ValueError("a lesson id is 64 hex characters")
    with L.bank_session():
        return L.delete_lesson(kind, lesson_id)


def clear_lessons(kind: str) -> int:
    """Delete everything under `kind`; how many there were. Not undoable."""
    from agent.memory import lessons as L

    kind = _checked_kind(kind)
    with L.bank_session(), L.bind_kind(kind):
        return L.clear_bank()


def notes() -> list[dict[str, Any]]:
    """One row per app with notes, shipped or learned: {"package", "seeded",
    "learned"} (how many learned notes)."""
    from agent.memory import lessons as L
    from agent.phone import notes as N

    shipped = set(N.seeded_packages())
    with L.bank_session():
        learned = {kind[len(L.APP_NOTE_PREFIX):]: len(L.list_kind(kind)) for kind in L.note_kinds()}
    return [{"package": package, "seeded": package in shipped, "learned": learned.get(package, 0)}
            for package in sorted(shipped | set(learned))]


def note(package: str) -> dict[str, Any]:
    """One app's notes: the shipped text, the learned rows (deletable by id)
    and the lines a phone run is actually shown (agent/phone/notes.py
    `notes_for`: seeded first, learned ones not already said, capped)."""
    from agent.memory import lessons as L
    from agent.phone import notes as N

    if not N.valid_package(package):
        raise ValueError("package is an Android package name")
    with L.bank_session():
        return {"package": package, "seeded": N.seeded(package),
                "learned": L.list_kind(N.note_kind(package)), "shown": N.notes_for(package)}


def delete_note(package: str, lesson_id: str) -> bool:
    """Forget one learned note. Shipped notes are package data and stay."""
    from agent.phone import notes as N

    if not N.valid_package(package):
        raise ValueError("package is an Android package name")
    return delete_lesson(N.note_kind(package), lesson_id)


def _model_row(m) -> dict[str, Any]:
    return {"spec": m.spec, "provider": m.provider, "id": m.id, "display_name": m.display_name,
            "capabilities": sorted(c.value for c in m.capabilities),
            "context_window": m.context_window, "max_output_tokens": m.max_output_tokens}


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

#: The most of a research document a `final` event carries. A report runs to
#: tens of kilobytes; this bounds the frame a runaway one makes.
DOCUMENT_MAX_BYTES = 2 * 1024 * 1024
#: The files a research run may leave beside document.md (agent/pipeline/
#: research.py FORMATS), listed on the event so a host can offer them.
DOCUMENT_FORMATS: tuple[str, ...] = ("md", "docx", "pdf", "xlsx")

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
        (LookupError with the reason, as `otto sessions` prints it).

        `ref` must look like one: "last" or lower-case hex. Anything else
        raises agent/memory/sessions.py's InvalidSessionId (a LookupError and
        a ValueError) before it can reach a query or a file name."""
        from agent.cli.shell import Session
        from agent.memory.sessions import check_ref

        if ref is not None:
            check_ref(ref)
        session = Session(ctx=self._ctx, workspace=None)
        if ref is not None:
            session.load(ref)
        return SessionHandle(session)

    def list_sessions(self, limit: int | None = 20) -> list[dict[str, Any]]:
        from agent.memory import sessions as index

        return [_session_row(info) for info in index.list_sessions(limit=limit)]

    def delete_session(self, session_id: str) -> bool:
        """Close the handle first if it is open: SQLite on Windows will not
        delete a file a connection still holds.

        A full id only -- no prefix, no "last": deleting is not a place to
        guess. InvalidSessionId for anything else."""
        from agent.memory import sessions as index

        return index.delete(index.check_id(session_id))

    def rename_session(self, session_id: str, title: str) -> str:
        """Rename a saved session that is not open here; returns the title as
        stored. Only one that exists: a host must not be able to create index
        rows for ids it made up. An open one renames through its handle."""
        from agent.memory import sessions as index

        index.check_id(session_id)
        if not " ".join(str(title or "").split()):
            raise ValueError("a title needs some text")
        if index.get(session_id) is None:
            raise LookupError(f"no session {session_id[:8]!r}")
        return index.rename(session_id, str(title)).title

    def export_session(self, session_id: str) -> dict[str, Any]:
        """{"filename", "data"}: a name to save it under and the export itself
        (agent/memory/sessions.py `export_payload`), without the workspace --
        a path on this computer is nobody else's business. LookupError for a
        session that was never saved."""
        from agent.memory import sessions as index

        info = index.get(index.check_id(session_id))
        if info is None:
            raise LookupError(f"no session {session_id[:8]!r} to export")
        return {"filename": index.export_filename(info),
                "data": index.export_payload(session_id, include_workspace=False)}

    def import_session(self, data: Any, *, keep_workspace: bool = False) -> dict[str, Any]:
        """A session from `export_session` data, as a `list_sessions` row.
        Keeps the exported id when nothing here has it; drops the workspace
        unless told otherwise. ValueError for data that is not an export."""
        from agent.memory import sessions as index

        return _session_row(index.import_payload(data, keep_workspace=keep_workspace))

    def transcript(self, ref: str) -> dict[str, Any]:
        """What a resumed session would show: the compacted earlier part as
        text, and the recent turns as messages."""
        from langchain_core.messages import HumanMessage

        handle = self.open_session(ref)  # checks `ref`
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
        #: Where the last turn ran: True on the phone, False off it, None for
        #: a turn that had no phone tools to decide about.
        self.phone: bool | None = None
        #: Tokens each turn of this handle spent, oldest first (the TUI's
        #: sparkline), and the last turn's totals.
        self.turn_tokens: list[int] = []
        self.last_turn: dict[str, Any] | None = None

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
            guidance: str = "", disabled_tools: Collection[str] | None = None,
            cancel: threading.Event | None = None,
            seats: Mapping[str, str] | None = None,
            phone: bool | None = None,
            off_phone_disabled_tools: Collection[str] | None = None,
            workspace: "str | os.PathLike | None" = AUTO) -> None:
        """One turn, to completion, on this thread.

        Every run-scoped binding happens here, on the thread that consumes
        the stream, because that is the only thread where a contextvar is
        visible to the graph. `tools` and `guidance` go to
        agent/pipeline/toolkit.py, `disabled_tools` to agent/pipeline/profile.py,
        `seats` to agent/router/overrides.py's bind_seats.

        `seats` left as None means agent/phone's PHONE_SEATS when `tools`
        include `phone_screen` and none otherwise, so a host that binds the
        phone tools gets the phone's fast judge without having to know about
        routing. Pass an explicit mapping (`{}` included) to decide.

        `disabled_tools` left as None means SUBPROCESS_TOOLS when the keys
        came in through `configure(environ=...)` and nothing otherwise: a
        subprocess inherits the environment, keys included, and a host that
        kept its keys out of every file should not find them in a shell the
        model runs. Pass an explicit collection (`()` included) to decide.

        WHETHER THE PHONE IS NEEDED is decided once, before anything is bound,
        when `tools` include the phone's (`phone_*`): `phone=None` asks
        agent/pipeline/nodes.py's `needs_phone` (one cheap call, failing
        toward yes; skipped when DECIDE_PHONE is off, which keeps the phone),
        True or False says. A `progress{kind: "phase"}` event reports it. On
        the phone the turn binds exactly what the arguments above say, with
        no workspace. Off it, the phone tools and `guidance` are left out,
        routing is routes.json's (no seats), `off_phone_disabled_tools`
        replaces `disabled_tools` (None: SUBPROCESS_TOOLS for keys from the
        host, else nothing -- a person at `otto tui` has the shell too), and
        the workspace defaults to the session's own directory, so a document
        has somewhere to be written. The decision holds for the whole turn:
        a question, its answer and every rejection run in the same bindings.

        `workspace` AUTO is that default; None is no workspace; a path is that
        directory, on or off the phone.
        """
        if disabled_tools is None:
            disabled_tools = SUBPROCESS_TOOLS if _configured.get("keys_from_host") else ()
        if off_phone_disabled_tools is None:
            off_phone_disabled_tools = SUBPROCESS_TOOLS if _configured.get("keys_from_host") else ()
        with self._lock:
            if self._running:
                raise RuntimeError("a turn is already running in this session")
            self._running = True
            self._cancel = cancel or threading.Event()
            self._pending_thread = None
            self._answer_ready.clear()
        try:
            self._drive(text, events, tools, guidance, disabled_tools, seats, phone=phone,
                        off_phone_disabled=off_phone_disabled_tools, workspace=workspace)
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

    def usage_report(self) -> dict[str, Any]:
        """What the TUI's sidebar shows about this session's spend, as data:
        the running ledger, tokens per turn, the last turn's totals, the
        title and how many turns it has."""
        return {"usage": self.usage.snapshot(), "turn_tokens": list(self.turn_tokens),
                "turn": self.last_turn, "title": self.title, "turns": self.turns}

    def rename(self, title: str) -> str:
        """Name the session, as `/rename` does; returns the title as stored
        (whitespace collapsed). A blank title is a ValueError: clearing a
        name is not something any front end offers."""
        if not " ".join(str(title or "").split()):
            raise ValueError("a title needs some text")
        self._session.rename(str(title))
        return self.title

    def close(self) -> None:
        # A phone turn distils its lesson after its answer is sent
        # (agent/pipeline/nodes.py wait_for_learning). Give it a bounded
        # moment to land before the stores it writes to close under it. Only
        # if the pipeline was ever loaded: closing must not import it.
        nodes = sys.modules.get("agent.pipeline.nodes")
        if nodes is not None:
            try:
                nodes.wait_for_learning(10)
            except Exception:  # noqa: BLE001 -- closing must not raise
                logger.warning("waiting for a background lesson failed", exc_info=True)
        self._session.close()

    # -- the loop ----------------------------------------------------------

    def _drive(self, text: str, events: Events, tools, guidance: str, disabled,
               seats: Mapping[str, str] | None = None, *, phone: bool | None = None,
               off_phone_disabled: Collection[str] = (), workspace=AUTO) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        from agent.pipeline import run as pipeline
        from agent.pipeline.profile import bind_tool_profile
        from agent.pipeline.progress import Cancelled, bind_progress
        from agent.pipeline.toolkit import bind_extra_tools
        from agent.pipeline.usage import bind_usage
        from agent.router.llm_provider.base import AuthError, ProviderError
        from agent.router.overrides import bind_seats

        started = time.monotonic()
        mark = self._ledger_mark()
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
        has_phone = any(_is_phone_tool(t) for t in tools)
        self.phone = True if has_phone else None
        try:
            if has_phone and (phone is not None or DECIDE_PHONE):
                if phone is None:
                    from agent.pipeline import nodes

                    # Before the bindings: they are contextvars, which do not
                    # carry from one LangGraph node to the next, so the choice
                    # has to be made out here and held for the whole turn.
                    # Counted in the session's ledger and stoppable, but with
                    # no sink: its streamed "PHONE: no" is not an answer.
                    with bind_progress(None, cancel=self._cancel), bind_usage(self.usage):
                        self.phone = bool(nodes.needs_phone(text, history))
                else:
                    self.phone = bool(phone)
                events({"type": "progress", "kind": "phase",
                        "text": "working on your phone" if self.phone else "answering here",
                        "calls": 0, "elapsed": round(time.monotonic() - started, 2),
                        "partial": "", "detail": {"phone": self.phone}})
            if has_phone and not self.phone:
                tools = [t for t in tools if not _is_phone_tool(t)]
                guidance, disabled, seats = "", off_phone_disabled, {}
            else:
                seats = _seats_for(tools, seats)
            workspace = self._workspace_for(bool(self.phone), workspace)
            with bind_progress(sink, cancel=self._cancel), \
                 bind_extra_tools(list(tools), guidance=guidance), \
                 bind_tool_profile(disabled), \
                 bind_seats(seats):
                # One binding covers the resume below too: it is opened on
                # this thread inside the same block, and LangGraph copies
                # this context into whatever worker runs a node.
                stream = pipeline.run_pipeline_stream(
                    text, session_id=session_id, history=history,
                    memory_context=memory_context, workspace=workspace, usage=self.usage,
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
                                workspace=workspace, usage=self.usage,
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
            events({"type": "error", "code": "cancelled", "message": "stopped",
                    "turn": self._close_turn(mark)})
            return
        except (AuthError, ProviderError) as exc:
            events({"type": "error", "code": "provider", "message": str(exc),
                    "turn": self._close_turn(mark)})
            return
        except Exception as exc:  # a provider's own exception must not kill the host
            logger.exception("turn failed")
            events({"type": "error", "code": "failed", "message": f"{type(exc).__name__}: {exc}",
                    "turn": self._close_turn(mark)})
            return

        raw = ((final or {}).get("final_output") or "").strip()
        # The history keeps the short report a research turn answers with;
        # the document itself rides beside it on the event, never in memory.
        self._session.record_turn(human, AIMessage(raw) if raw else None)
        events({"type": "final", "text": raw, "usage": self.usage.snapshot(),
                "trace_id": self._session.trace_id, "turn": self._close_turn(mark),
                "title": self.title, "turns": self.turns, "phone": self.phone,
                "document": _document(workspace, (final or {}).get("document_path"))})

    def _ledger_mark(self) -> dict[str, tuple[int, int, int, int, int]]:
        """Where each model's counters stood, so a turn's own share can be
        taken off the session ledger afterwards (`_turn_totals`)."""
        return {name: (m.calls, m.input_tokens, m.output_tokens, m.cached_input_tokens,
                       m.cache_write_tokens)
                for name, m in list(self.usage.by_model.items())}

    def _close_turn(self, mark: Mapping[str, tuple[int, int, int, int, int]]) -> dict[str, Any]:
        turn = self._turn_totals(mark)
        self.last_turn = turn
        self.turn_tokens.append(turn["tokens"])
        return turn

    def _turn_totals(self, mark: Mapping[str, tuple[int, int, int, int, int]]) -> dict[str, Any]:
        """This turn's tokens, calls and dollars: the ledger minus `mark`,
        priced per model the way agent/pipeline/usage.py prices the whole --
        cost None when a model that reported tokens has no rate, rather than
        a short total passed off as the total."""
        from agent.pipeline.usage import ModelUsage

        tokens = calls = 0
        cost: float | None = 0.0
        for name, m in list(self.usage.by_model.items()):
            c0, i0, o0, r0, w0 = mark.get(name, (0, 0, 0, 0, 0))
            share = ModelUsage(model=name, calls=m.calls - c0, input_tokens=m.input_tokens - i0,
                               output_tokens=m.output_tokens - o0, cached_input_tokens=m.cached_input_tokens - r0,
                               cache_write_tokens=m.cache_write_tokens - w0, reported=m.reported)
            if share.calls <= 0 and share.total_tokens <= 0:
                continue
            calls += share.calls
            tokens += share.total_tokens
            if share.reported and share.total_tokens:
                priced = share.cost
                cost = None if priced is None or cost is None else cost + priced
        return {"tokens": tokens, "calls": calls, "cost": cost}

    def _workspace_for(self, on_phone: bool, workspace) -> str | None:
        """The directory a turn's file tools and research documents get.
        AUTO: none on the phone (a phone run has no files, and its prompt and
        call budget were measured without them), else the session's own
        `<OTTO_HOME>/workspaces/<session_id>`, created by the run when it
        binds it -- one per session, so a follow-up finds the document the
        turn before wrote."""
        if workspace is not AUTO:
            return None if workspace is None else str(workspace)
        if on_phone:
            return None
        return str(_home.otto_home() / "workspaces" / self._session.session_id)

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


def _document(workspace: str | None, relative: str | None) -> dict[str, Any] | None:
    """The document a research turn wrote, for the `final` event, or None.

    `relative` comes out of the graph's state, so it is treated as untrusted:
    it must resolve (symlinks included) to a Markdown file inside the turn's
    own workspace, or nothing is read."""
    if not workspace or not relative:
        return None
    try:
        root = Path(workspace).resolve()
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or path.suffix != ".md" or not path.is_file():
            return None
        with path.open("rb") as fh:
            data = fh.read(DOCUMENT_MAX_BYTES + 1)
    except (OSError, ValueError):
        return None
    files = [f"{path.stem}.{fmt}" for fmt in DOCUMENT_FORMATS if path.with_suffix(f".{fmt}").is_file()]
    return {"path": path.relative_to(root).as_posix(), "format": "md",
            "markdown": data[:DOCUMENT_MAX_BYTES].decode("utf-8", errors="ignore"),
            "truncated": len(data) > DOCUMENT_MAX_BYTES, "files": files}


def _is_phone_tool(tool) -> bool:
    """One of agent/phone/tools.py's: every one is named phone_*."""
    return str(getattr(tool, "name", "") or "").startswith("phone_")


def _seats_for(tools: Sequence["ExtraTool"], seats: Mapping[str, str] | None) -> Mapping[str, str]:
    """The seats a turn binds: the caller's when given, the phone's when the
    turn carries the phone tools, none otherwise -- so a coding turn's routing
    is exactly what routes.json says."""
    if seats is not None:
        return seats
    if any(getattr(t, "name", None) == "phone_screen" for t in tools):
        from agent.phone import PHONE_SEATS

        return PHONE_SEATS
    return {}


def _plain(detail) -> dict[str, Any] | None:
    """A progress detail as JSON-safe data."""
    if not detail:
        return None
    return {str(k): (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
            for k, v in dict(detail).items()}
