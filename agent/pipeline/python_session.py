"""Per-run binding of ONE PERSISTENT PYTHON INTERPRETER, so what a snippet
builds is still there for the next one (issue #1).

Every execute_python call used to be a standalone script in a fresh process
(agent/pipeline/tools.py). An agent that loaded a dataset, built an index or
constructed an object in one call had to rebuild it in the next, from the
transcript or through disk: work compounded as text, never as live state.
Prime Agent (arXiv:2608.23552) names the persistent interpreter as the layer
that makes long-horizon work compositional at all, alongside the disk-backed
history and recursive subagents Otto already had.

Bound the same way agent/pipeline/execution.py binds a command runner and
agent/pipeline/workspace.py a directory, for the same reason: a tool in
TOOL_DISPATCH is a plain function of one string and cannot be handed a
session as an argument. NOTHING BOUND stays a valid state -- `current_python_
session()` returning None means execute_python runs a fresh process per call
exactly as before, which is what agent/eval/runner.py's golden checker still
gets (it runs outside any run) and what every existing test asserts on.

What the fresh-process model gave for free was that nothing survived a call,
and each thing it wiped now has to be handled on purpose. In order of how
much they matter:

* ISOLATION. A session is a contextvar bound per run and closed when the
  run's block exits, never a module global: two runs in two threads, or two
  golden-eval tasks back to back, never share an interpreter, and a builtin
  one task poisoned is gone with that task's process. A delegated subagent
  (nodes.py's `_delegate`) gets its OWN fresh session through
  `fresh_python_session()`, torn down when it returns -- explicitly, not by
  contextvar inheritance. The child runs with none of the parent's
  conversation, so state it could not know exists is not state it should
  reach, and a child that wedged the interpreter must not cost the parent
  its remaining calls.

* FRAMING. Results come back on a channel the snippet's own output never
  touches (the shim's docstring has the mechanics), and every call is framed
  with a uuid the parent minted. A frame with a wrong or stale id is
  dropped, so a snippet that prints something frame-shaped -- by accident or
  because it executed content it read from somewhere -- is printing, not
  speaking for the next call.

* A CALL THAT WILL NOT END. `subprocess.run(timeout=)` killed the whole
  process, which was total and simple. Here the in-flight call is
  interrupted first (SIGINT; CTRL_BREAK on Windows, best-effort) so only it
  stops and the state survives; if nothing comes back within a grace period
  -- a C call that never checks for signals, a snippet that ignores them --
  the process is killed and replaced, and the result SAYS so (RESET_NOTE in
  stderr, next to the "[timed out]" the old path wrote). A model that is not
  told its variables are gone would keep using them. The returncode/
  timed_out contract is the old one either way: -1 and True.

* CEILINGS. A fresh process bounded memory by one call; a session
  accumulates by design. RLIMIT_AS on the child (Linux; macOS ignores it and
  Windows has no equivalent) turns a runaway allocation into a MemoryError
  inside the snippet rather than an OOM kill of anything. OTTO_PYTHON_
  SESSION_MEMORY_MB sets it, 0 disables it. A max-calls and an idle-timeout
  eviction exist too and default to off: a run is already bounded by its
  model-call budget and its session dies with it, so a harness that wants
  a ceiling can ask for one without every chat turn paying for it.

* LIFETIME. The private directory (capture files, and the snippet's cwd when
  no workspace is bound) is created with the process and deleted with it;
  `python_session()` is opened INSIDE the workspace binding in run.py, so a
  session is always closed before a harness's scratch directory goes away
  under it. The process is started lazily on the first call: a greeting
  that never runs Python never pays for an interpreter.

* THE CONTAINER PATH. With a CommandRunner bound (agent/pipeline/
  execution.py), execute_python keeps its one-shot `python3 -` command --
  the snippet must land in the container, and a host-side session would be
  wrong, not merely different. `default_python_session()` therefore offers
  nothing for such a run, so the prompt claims no persistence there; a
  session bound alongside a runner by hand is not used and the tool result
  says so. Real persistence inside a container is a runner-side REPL, a
  separate piece of work.

Not a security sandbox, as tools.py already says of the fresh-process path:
a snippet that redefines a builtin, installs a signal handler or swallows
KeyboardInterrupt has done so for the rest of the run. That was true of a
single call before; it is now true of the run, which is the trade the
feature makes and the reason the reset paths above are loud.

OTTO_PYTHON_SESSION=0 switches the whole feature off -- the Claw-Eval
comparison the issue asks for needs both arms runnable from one checkout.
"""
from __future__ import annotations

import base64
import contextvars
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from agent.pipeline.execution import current_command_runner
from agent.pipeline.workspace import current_workspace

logger = logging.getLogger(__name__)

PYTHON_SESSION_ENV = "OTTO_PYTHON_SESSION"
MEMORY_LIMIT_ENV = "OTTO_PYTHON_SESSION_MEMORY_MB"
#: Address-space ceiling for the interpreter when the environment does not
#: say. Generous on purpose: RLIMIT_AS counts reserved virtual memory, and a
#: numerical library reserves far more than it touches.
DEFAULT_MEMORY_LIMIT_MB = 8192
#: How long after the interrupt the in-flight call gets to report back before
#: the process is killed and replaced.
INTERRUPT_GRACE_S = 3.0
#: How long the child gets to say it is ready. It is one interpreter start,
#: which is well under a second; a loaded CI runner is why this is not one.
BOOT_TIMEOUT_S = 60.0

_OFF = {"0", "false", "no", "off"}

#: What a tool result says when the interpreter had to be replaced. Read by
#: the model, so it names the consequence rather than the mechanism.
RESET_NOTE = (
    "[python session restarted: every variable, import and object defined in "
    "earlier calls is gone; define again what you need]"
)

#: What a tool result says when a session is bound but a command runner is
#: too, and the runner won.
NO_PERSISTENCE_NOTE = (
    "[state does not persist between execute_python calls inside a container: "
    "each call is a fresh script]"
)

#: Appended to a MemoryError under the ceiling. The old fresh-process path
#: had no ceiling, so a snippet that used to work can now fail here, and a
#: bare MemoryError does not say why or what to do about it.
MEMORY_CEILING_NOTE = (
    "[the python session's address space is capped at {mb} MB; set "
    f"{MEMORY_LIMIT_ENV} to raise it, or to 0 for no cap]"
)

#: What a tool result says when the interpreter could not be started and the
#: call ran in a fresh process instead. Distinct from the container note:
#: nothing here is about where the code ran, only that it did not persist.
UNAVAILABLE_NOTE = (
    "[the persistent interpreter could not start; this call ran in a fresh "
    "process and nothing from it persists]"
)

_SHIM = Path(__file__).with_name("_python_session_shim.py")


@dataclass(frozen=True)
class SessionResult:
    """What one call in a session produced. `reset` is True on the first
    result the caller sees after the interpreter was replaced -- the call
    that hung and was killed, the call in which the process died, or the
    first call after an eviction -- and False otherwise."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False
    reset: bool = False


@runtime_checkable
class PythonSession(Protocol):
    """What tools.py needs from a session. `run` never raises for anything a
    snippet did; `fresh` is a new session with the same settings and none of
    the state, for a delegated subagent."""

    def run(self, code: str, timeout: float) -> SessionResult: ...

    def close(self) -> None: ...

    def fresh(self) -> "PythonSession": ...


class LocalPythonSession:
    """One long-lived interpreter on this machine, started on first use.

    `cwd` is where snippets run: the bound workspace at spawn time when None
    and one is bound, otherwise a directory private to this session. See the
    module docstring for the ceilings and the hang policy.
    """

    def __init__(
        self,
        cwd: Path | str | None = None,
        *,
        memory_limit_bytes: int | None = None,
        max_calls: int | None = None,
        idle_timeout_s: float | None = None,
        interrupt_grace_s: float = INTERRUPT_GRACE_S,
    ) -> None:
        self._cwd = Path(cwd) if cwd is not None else None
        self._memory_limit = (
            _memory_limit_from_env() if memory_limit_bytes is None else memory_limit_bytes
        )
        self._max_calls = max_calls
        self._idle_timeout = idle_timeout_s
        self._grace = interrupt_grace_s
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._frames: queue.Queue | None = None
        self._reader: threading.Thread | None = None
        self._private: Path | None = None
        self._boot_log: Path | None = None
        self._calls_in_process = 0
        self._has_state = False
        self._last_used = 0.0

    # -- what the outside can see ------------------------------------------

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def private_dir(self) -> Path | None:
        return self._private

    @property
    def memory_limit_bytes(self) -> int:
        """The address-space ceiling on the interpreter, 0 for none. Only
        Linux enforces it -- see the module docstring."""
        return self._memory_limit or 0

    def fresh(self) -> "LocalPythonSession":
        return LocalPythonSession(
            self._cwd, memory_limit_bytes=self._memory_limit, max_calls=self._max_calls,
            idle_timeout_s=self._idle_timeout, interrupt_grace_s=self._grace,
        )

    # -- the call ----------------------------------------------------------

    def run(self, code: str, timeout: float) -> SessionResult:
        with self._lock:
            # `lost` is "the caller had state and it is gone": an eviction, or
            # a process that died on its own since the last call. The first
            # spawn of a session loses nothing.
            lost = self._evict_if_due()
            if not self.alive:
                lost = lost or self._has_state
                self._discard()
                self._spawn()
            request_id = uuid.uuid4().hex
            try:
                self._send({"id": request_id, "code": code})
            except (OSError, ValueError):
                # The pipe is gone: the child died between calls. One
                # respawn; a child that cannot take its first request either
                # is an interpreter that does not work here, which is what
                # SessionUnavailable means and what tools.py falls back on.
                lost = True
                self._discard()
                self._spawn()
                try:
                    self._send({"id": request_id, "code": code})
                except (OSError, ValueError) as exc:
                    self._discard()
                    raise SessionUnavailable(
                        f"the interpreter started but would not take a request: {exc}"
                    ) from exc
            self._calls_in_process += 1
            result = self._await(request_id, timeout)
            self._last_used = time.monotonic()
            self._has_state = self.alive
            if lost and not result.reset:
                result = replace(result, reset=True)
            return result

    def close(self) -> None:
        with self._lock:
            self._discard()

    # -- process lifetime --------------------------------------------------

    def _spawn(self) -> None:
        self._private = Path(tempfile.mkdtemp(prefix="otto-pysession-"))
        cwd = self._cwd or current_workspace()
        if cwd is None:
            cwd = self._private / "cwd"
            cwd.mkdir()
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{cwd}{os.pathsep}{existing}" if existing else str(cwd)
        self._boot_log = self._private / "boot.err"
        popen_kwargs: dict = {}
        if os.name == "posix":
            # Its own process group: a Ctrl-C at the terminal running otto
            # must not land in a snippet, and killing the group later takes
            # any subprocess a snippet left behind with it.
            popen_kwargs["start_new_session"] = True
        else:
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        with open(self._boot_log, "wb") as boot_err:
            self._proc = subprocess.Popen(
                [sys.executable, str(_SHIM), str(self._private), str(self._memory_limit or 0)],
                cwd=str(cwd), env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=boot_err,
                **popen_kwargs,
            )
        self._frames = queue.Queue()
        self._reader = threading.Thread(
            target=_pump, args=(self._proc.stdout, self._frames), daemon=True,
            name=f"otto-pysession-{self._proc.pid}",
        )
        self._reader.start()
        self._calls_in_process = 0
        self._last_used = time.monotonic()
        try:
            frame = self._frames.get(timeout=BOOT_TIMEOUT_S)
        except queue.Empty:
            frame = None
        if not (isinstance(frame, dict) and frame.get("ready")):
            problem = self._boot_problem()
            self._discard()
            raise SessionUnavailable(problem)

    def _boot_problem(self) -> str:
        text = ""
        if self._boot_log is not None:
            try:
                text = self._boot_log.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
        return text[-2000:] or "the interpreter did not start"

    def _discard(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            _kill(proc)
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass
        if self._reader is not None:
            self._reader.join(timeout=5.0)
            self._reader = None
        self._frames = None
        if self._private is not None:
            shutil.rmtree(self._private, ignore_errors=True)
            self._private = None

    def _evict_if_due(self) -> bool:
        """Kill the process if a ceiling says so. True when a caller who had
        state has just lost it."""
        if not self.alive:
            return False
        due = (
            (self._max_calls is not None and self._calls_in_process >= self._max_calls)
            or (self._idle_timeout is not None
                and time.monotonic() - self._last_used > self._idle_timeout)
        )
        if due:
            self._discard()
        return due

    # -- the wire ----------------------------------------------------------

    def _send(self, request: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("no interpreter to send to")  # spawn first
        code = base64.b64encode(request["code"].encode("utf-8", errors="replace"))
        self._proc.stdin.write(request["id"].encode("ascii") + b" " + code + b"\n")
        self._proc.stdin.flush()

    def _await(self, request_id: str, timeout: float) -> SessionResult:
        frame = self._frame_for(request_id, timeout)
        if isinstance(frame, dict):
            return _as_result(frame)
        if frame is _DIED:
            return self._died()

        # Nothing within the timeout: interrupt just this call, and give the
        # child a grace period to say it stopped.
        interrupted = self._interrupt()
        frame = self._frame_for(request_id, self._grace) if interrupted else None
        if isinstance(frame, dict):
            # The wall clock is the contract, whatever the child reports.
            return SessionResult(
                frame.get("stdout", ""), frame.get("stderr", ""), -1, True, False,
            )
        if frame is _DIED:
            return self._died(timed_out=True)

        # Could not be interrupted: kill it, keep what it managed to print.
        partial_out, partial_err = self._partial_capture()
        self._discard()
        return SessionResult(partial_out, partial_err, -1, True, True)

    def _frame_for(self, request_id: str, timeout: float):
        """The frame answering `request_id`, `_DIED` on EOF, or None on
        timeout. A frame with another id is a stale answer to a call that
        already timed out, or a forgery -- dropped either way."""
        assert self._frames is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                frame = self._frames.get(timeout=remaining)
            except queue.Empty:
                return None
            if frame is None:
                return _DIED
            if isinstance(frame, dict) and frame.get("id") == request_id:
                return frame

    def _died(self, *, timed_out: bool = False) -> SessionResult:
        proc = self._proc
        code = -1
        if proc is not None:
            try:
                code = proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        partial_out, partial_err = self._partial_capture()
        self._discard()
        note = f"[python session died (exit {code})]"
        stderr = (partial_err + "\n" + note).strip()
        return SessionResult(partial_out, stderr, -1 if timed_out else (code or 1),
                             timed_out, True)

    def _partial_capture(self) -> tuple[str, str]:
        """What a call that never reported had printed so far -- the capture
        files are in the private dir, and a dead child's are readable."""
        if self._private is None:
            return "", ""
        texts = []
        for name in ("stdout", "stderr"):
            try:
                texts.append((self._private / name).read_text(encoding="utf-8", errors="replace"))
            except OSError:
                texts.append("")
        return texts[0], texts[1]

    def _interrupt(self) -> bool:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        try:
            if os.name == "posix":
                # The whole group, as a terminal Ctrl-C would: a snippet
                # blocked on a subprocess it started (a test run, a build)
                # has that subprocess interrupted too, not just itself.
                os.killpg(proc.pid, signal.SIGINT)
            else:
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            return True
        except OSError:
            return False


class SessionUnavailable(RuntimeError):
    """The interpreter could not be started. tools.py falls back to the
    fresh-process path on it and says so; it is never a crashed run."""


_DIED = object()


def _pump(stream, frames: queue.Queue) -> None:
    """Reader thread: every line from the child's result channel onto the
    queue as a parsed frame, then None at EOF. Unparseable lines are noise
    (on Windows a subprocess of a snippet can, in principle, still reach the
    original handle) and are dropped."""
    try:
        for raw in stream:
            frame = _parse_frame(raw)
            if frame is not None:
                frames.put(frame)
    except (OSError, ValueError):
        pass
    finally:
        frames.put(None)


def _parse_frame(raw: bytes) -> dict | None:
    """One line of the shim's protocol (its module docstring) as a dict, or
    None for anything else."""
    parts = raw.rstrip(b"\r\n").split(b" ")
    if parts == [b"READY"]:
        return {"ready": True}
    if len(parts) != 6 or parts[0] != b"R":
        return None
    try:
        return {
            "id": parts[1].decode("ascii"),
            "returncode": int(parts[2]),
            "timed_out": parts[3] == b"1",
            "stdout": base64.b64decode(parts[4], validate=True).decode("utf-8", errors="replace"),
            "stderr": base64.b64decode(parts[5], validate=True).decode("utf-8", errors="replace"),
        }
    except ValueError:  # binascii.Error is one
        return None


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=1.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        logger.warning("python session %s did not die when killed", proc.pid)


def _as_result(frame: dict) -> SessionResult:
    return SessionResult(
        stdout=str(frame.get("stdout", "")),
        stderr=str(frame.get("stderr", "")),
        returncode=int(frame.get("returncode", 1)),
        timed_out=bool(frame.get("timed_out", False)),
    )


def _memory_limit_from_env() -> int:
    raw = os.environ.get(MEMORY_LIMIT_ENV, "").strip()
    if raw:
        try:
            megabytes = int(float(raw))
        except ValueError:
            logger.warning("%s=%r is not a number; ignoring it", MEMORY_LIMIT_ENV, raw)
        else:
            return max(megabytes, 0) * 1024 * 1024
    return DEFAULT_MEMORY_LIMIT_MB * 1024 * 1024


# --------------------------------------------------------------------------
# The binding
# --------------------------------------------------------------------------

_current: contextvars.ContextVar[PythonSession | None] = contextvars.ContextVar(
    "otto_current_python_session", default=None,
)


@contextmanager
def bind_python_session(session: PythonSession | None) -> Iterator[PythonSession | None]:
    """Make `session` what `current_python_session()` returns for this block
    and anything it calls. Does not close it -- whoever made it owns it (see
    `python_session()` for the usual pairing). Restores the previous binding
    on exit, so nesting unwinds correctly."""
    previous = _current.get()
    token = _current.set(session)
    try:
        yield session
    finally:
        try:
            _current.reset(token)
        except ValueError:
            # Unwound from a different context: a streaming run finalised
            # on another thread (agent/pipeline/tracing.py).
            _current.set(previous)


def current_python_session() -> PythonSession | None:
    """The session bound by the innermost enclosing binding, or None -- the
    fresh-process-per-call behaviour, and not an error."""
    return _current.get()


@contextmanager
def python_session(session: PythonSession | None = None, **settings) -> Iterator[PythonSession]:
    """Open a session for this block: bind it, and close it on the way out.
    A new LocalPythonSession with `settings` unless one is given."""
    live = session if session is not None else LocalPythonSession(**settings)
    try:
        with bind_python_session(live):
            yield live
    finally:
        live.close()


@contextmanager
def fresh_python_session() -> Iterator[PythonSession | None]:
    """A new session of the bound one's kind for this block, closed after;
    nothing at all when nothing is bound. What a delegated subagent runs in
    (module docstring, ISOLATION)."""
    parent = current_python_session()
    if parent is None:
        yield None
        return
    with python_session(parent.fresh()) as child:
        yield child


def default_python_session() -> LocalPythonSession | None:
    """What a run should bind: a lazy local session, or None when the
    feature is switched off or the run's commands go to a container."""
    if os.environ.get(PYTHON_SESSION_ENV, "").strip().lower() in _OFF:
        return None
    if current_command_runner() is not None:
        return None
    return LocalPythonSession()


def run_session_binding():
    """`python_session()` for a run, or nothing -- agent/pipeline/run.py's
    counterpart to its `_workspace_binding`."""
    live = default_python_session()
    return python_session(live) if live is not None else nullcontext()


#: Told to the model once per run, next to the workspace note. Constant text
#: on purpose: nodes.py rebuilds the prompt on a resume and the stored
#: transcript assumes it is byte-identical.
_NOTE = (
    "PYTHON SESSION: execute_python runs every snippet in ONE interpreter that "
    "persists for this run. Variables, imports, functions and loaded data "
    "survive from one call to the next, so build on what an earlier call "
    "defined instead of recomputing it or writing it to disk. An exception or "
    "sys.exit() ends only that call. A call that runs too long is interrupted; "
    "if the interpreter has to be restarted the result says so, and everything "
    "defined earlier is gone. Each result shows only what that call printed."
)


def python_session_note() -> str:
    """The prompt block that says state persists. Empty when nothing is
    bound, so a caller appends it unconditionally -- the same shape as
    workspace_note() and render_note()."""
    return _NOTE if current_python_session() is not None else ""
