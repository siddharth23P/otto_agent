"""The server: one connection, one runtime, any number of sessions.

Built on `websockets` (the `[serve]` extra) and agent/embed.py. Each turn runs
`SessionHandle.run` on a worker thread, exactly as the TUI does; its events
are forwarded to the client as they happen, and its phone tools reach the
client through agent/server/proxy.py. Everything a person could do in the
app -- start a turn, answer a question, stop, list and resume sessions -- is
one message type (agent/server/protocol.py).

Authentication is a shared token from `hello`, compared in constant time.
The server binds loopback by default; anything wider is the operator's call
(`otto serve --host 0.0.0.0` behind a network they trust). What the token
grants is a turn, and a turn that is not on the phone runs like `otto tui`:
shell and Python on this computer included, unless `no_exec` (`--no-exec`)
takes the subprocess tools away. The phone reaches a loopback server through
`adb reverse tcp:8765 tcp:8765`.

THREE LANES for a client's messages, so nothing quick waits on anything slow:
  inline      ping, answer, cancel, device_result -- answered on the event
              loop at once. A device_result in particular must never queue
              behind a request: a turn's worker thread is blocked on it.
  ordered     turn and sessions, one at a time in the order sent, so an
              `open` is registered before the `turn` that names its session.
  concurrent  everything else (settings, catalogues, memory), each its own
              task, since a doctor or a model catalogue can take tens of
              seconds of network.
Every blocking call a request makes runs on the server's ops pool
(`MAX_OP_WORKERS`), never on the loop and never on the turn pool.
"""
from __future__ import annotations

import asyncio
import functools
import hmac
import ipaddress
import logging
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Any

from agent.server import protocol
from agent.server.proxy import SocketPhone

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    pass


class InvalidSession(Exception):
    """A session id or ref from the client that is not one otto mints."""


def _sid(value: Any, *, ref: bool = False, required: bool = True) -> str:
    """The client's `session_id` (or `ref`, which may also be a hex prefix or
    "last") once it is known to be one, else InvalidSession -- the error
    frame `invalid_session`. Every id a client sends passes through here
    before it reaches a handle, a query or a file name: a session id used to
    go straight into `sessions.delete`, which made it a path. Empty is
    returned as "" when `required` is False, for the messages where no id
    has always meant "none"."""
    from agent.memory.sessions import valid_id, valid_ref

    text = value if isinstance(value, str) else ("" if value is None else str(value))
    if not text and not required:
        return ""
    if (valid_ref if ref else valid_id)(text):
        return text
    raise InvalidSession(f"{text[:80]!r} is not a session id" + (", prefix or 'last'" if ref else ""))


def _peer_is_loopback(websocket) -> bool:
    """Whether the client is on this computer (`adb reverse` included, which
    arrives from 127.0.0.1). Unknown is no."""
    try:
        host = str(websocket.remote_address[0]).split("%", 1)[0]
        ip = ipaddress.ip_address(host)
    except (AttributeError, IndexError, TypeError, ValueError):
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return ip.is_loopback or bool(mapped is not None and mapped.is_loopback)


#: How many sessions one connection may hold open. A phone uses one or two;
#: the cap bounds what a client holding the token can spend, which is the
#: one thing the shared-token trust model does not otherwise limit.
MAX_SESSIONS_PER_CONNECTION = 8

#: Turns across every connection share one pool of this size, not Python's
#: process-wide default executor: a server bound wider than loopback with
#: many clients queues turns here, visibly, instead of starving whatever
#: else the process runs in that default pool.
MAX_TURN_WORKERS = 8

#: Requests (reading sessions, catalogues, settings) share a smaller pool of
#: their own, so eight running turns cannot starve a session list, and a slow
#: catalogue cannot take a turn's thread.
MAX_OP_WORKERS = 4

#: `turn.phone`: decide per turn (agent/embed.py DECIDE_PHONE), or say.
PHONE_MODES: dict[str, bool | None] = {"auto": None, "on": True, "off": False}

#: The lanes (module docstring). Message types not named here are inline.
ORDERED = frozenset({"turn", "sessions"})
CONCURRENT: frozenset[str] = frozenset()


class Connection:
    """State for one client socket."""

    def __init__(self, websocket, token: str, loop: asyncio.AbstractEventLoop,
                 executor: ThreadPoolExecutor | None = None, *, no_exec: bool = False,
                 ops_executor: ThreadPoolExecutor | None = None) -> None:
        self.ws = websocket
        self.token = token
        #: The operator's --no-exec: no turn on this server starts a subprocess.
        self.no_exec = no_exec
        self.loop = loop
        self.executor = executor
        self.ops_executor = ops_executor
        #: Whether the client is on this computer. Settings that change what
        #: the whole installation does are only taken from a local client.
        self.local = _peer_is_loopback(websocket)
        self.phone: SocketPhone | None = None
        self.capabilities: set[str] = set()
        self.handles: dict[str, Any] = {}
        self.turns: dict[str, asyncio.Future] = {}
        self._runtime = None
        self._runtime_lock = asyncio.Lock()
        self._opening = 0
        self._queue: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()

    # -- plumbing ----------------------------------------------------------

    async def send(self, frame: str) -> None:
        await self.ws.send(frame)

    async def reply(self, kind: str, rid: str | int | None, **fields: Any) -> None:
        await self.send(protocol.reply(kind, rid, **fields))

    async def send_error(self, code: str, message: str, rid: str | int | None = None) -> None:
        await self.send(protocol.reply("error", rid, code=code, message=message))

    async def blocking(self, fn, *args, **kwargs):
        """`fn(*args, **kwargs)` on the ops pool, awaited."""
        return await self.loop.run_in_executor(self.ops_executor, functools.partial(fn, *args, **kwargs))

    def runtime(self):
        if self._runtime is None:
            from agent import embed

            self._runtime = embed.Runtime()
        return self._runtime

    async def runtime_async(self):
        """The runtime, built off the loop: the first one imports the whole
        pipeline, which takes seconds a ping should not wait for."""
        if self._runtime is None:
            async with self._runtime_lock:
                if self._runtime is None:
                    from agent import embed

                    self._runtime = await self.blocking(embed.Runtime)
        return self._runtime

    async def handle_for(self, session_id: str | None, ref: str | None = None):
        if session_id and session_id in self.handles:
            return self.handles[session_id]
        if len(self.handles) + self._opening >= MAX_SESSIONS_PER_CONNECTION:
            raise LookupError(f"this connection already holds {MAX_SESSIONS_PER_CONNECTION} sessions; "
                              "close one first")
        self._opening += 1
        try:
            runtime = await self.runtime_async()
            handle = await self.blocking(runtime.open_session, ref or session_id)
        finally:
            self._opening -= 1
        existing = self.handles.get(handle.id)
        if existing is not None:
            # A prefix or "last" named a session this connection already
            # holds: keep the one that may be running, drop the second.
            await self.blocking(handle.close)
            return existing
        self.handles[handle.id] = handle
        return handle

    def busy(self, session_id: str) -> bool:
        """Whether a turn is running, or about to, in that session. The
        future is checked as well as the handle: a turn handed to the pool a
        moment ago has not set `running` yet."""
        handle = self.handles.get(session_id)
        future = self.turns.get(session_id)
        return bool((handle is not None and handle.running) or (future is not None and not future.done()))

    def any_busy(self) -> bool:
        return any(self.busy(sid) for sid in set(self.handles) | set(self.turns))

    # -- the conversation --------------------------------------------------

    async def hello(self, raw: str | bytes) -> None:
        message = protocol.decode(raw)
        if message.get("type") != "hello":
            raise ProtocolError("the first frame must be hello")
        version = message.get("protocol_version")
        if not isinstance(version, int) or version < protocol.MIN_PROTOCOL:
            raise ProtocolError(f"protocol_version {version!r} is older than {protocol.MIN_PROTOCOL}; update the app")
        if version > protocol.PROTOCOL_VERSION:
            raise ProtocolError(f"protocol_version {version} is newer than this otto ({protocol.PROTOCOL_VERSION}); "
                                "update otto (pipx upgrade otto-cli-agent)")
        offered = str(message.get("token") or "")
        if not hmac.compare_digest(offered.encode(), self.token.encode()):
            raise ProtocolError("bad token")
        self.capabilities = {str(c) for c in (message.get("capabilities") or [])}
        if "phone" in self.capabilities:
            self.phone = SocketPhone(self.send, self.loop)
        from agent import embed

        info = embed.version()
        await self.send(protocol.encode("hello_ok", otto_version=info["otto"], api_version=info["api"],
                                        protocol_version=protocol.PROTOCOL_VERSION,
                                        min_protocol=protocol.MIN_PROTOCOL,
                                        features=list(protocol.FEATURES)))

    async def dispatch(self, raw: str | bytes) -> None:
        """Route one frame to its lane. Never raises for a bad or failing
        request -- the client gets an error carrying its id -- only when the
        socket itself is gone."""
        message = protocol.decode(raw)
        kind = message["type"]
        rid = None
        if kind not in ("invalid", "device_result"):  # a device_result's id is the device call's
            try:
                rid = protocol.request_id(message)
            except ValueError as exc:
                await self.send_error("invalid", str(exc))
                return
        if kind in ORDERED:
            if self._queue is None:
                self._queue = asyncio.Queue()
                self._worker = asyncio.ensure_future(self._drain())
            await self._queue.put((message, rid))
        elif kind in CONCURRENT:
            task = asyncio.ensure_future(self._guarded(message, rid))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        else:
            await self._guarded(message, rid)

    async def _drain(self) -> None:
        while True:
            message, rid = await self._queue.get()
            await self._guarded(message, rid)

    async def _guarded(self, message: dict, rid: str | int | None) -> None:
        try:
            await self._dispatch(message, rid)
        except asyncio.CancelledError:
            raise
        except InvalidSession as exc:
            await self._quietly(self.send_error("invalid_session", str(exc), rid))
        except Exception as exc:  # one bad message must not drop the connection
            logger.exception("serve: message failed")
            await self._quietly(self.send_error("failed", f"{type(exc).__name__}: {exc}", rid))

    async def _quietly(self, sending) -> None:
        try:
            await sending
        except Exception:
            logger.debug("serve: could not send an error; the client has gone", exc_info=True)

    async def _dispatch(self, message: dict, rid: str | int | None) -> None:
        kind = message["type"]
        if kind == "invalid":
            await self.send_error("invalid", message["reason"])
        elif kind == "ping":
            await self.reply("pong", rid)
        elif kind == "device_result":
            if self.phone is None or not self.phone.resolve(message):
                await self.send_error("unexpected", "no device_call is waiting for that id")
        elif kind == "turn":
            await self.start_turn(message, rid)
        elif kind == "answer":
            handle = self.handles.get(_sid(message.get("session_id"), required=False))
            if handle is None or not handle.answer(str(message.get("thread_id") or ""), str(message.get("text") or "")):
                await self.send_error("no_question", "no question is waiting for that answer", rid)
        elif kind == "cancel":
            handle = self.handles.get(_sid(message.get("session_id"), required=False))
            if handle is not None:
                handle.cancel()
        elif kind == "sessions":
            await self.sessions(message, rid)
        else:
            await self.send_error("unknown", f"unknown message type {kind!r}", rid)

    async def start_turn(self, message: dict, rid: str | int | None = None) -> None:
        text = str(message.get("text") or "").strip()
        if not text:
            await self.send_error("empty", "turn needs text", rid)
            return
        mode = message.get("phone")
        mode = "auto" if mode is None else mode
        if not isinstance(mode, str) or mode not in PHONE_MODES:
            await self.send_error("invalid", "turn.phone is one of auto, on, off", rid)
            return
        if mode == "on" and self.phone is None:
            await self.send_error("no_phone", "this connection did not offer the phone capability", rid)
            return
        session_ref = _sid(message.get("session_id"), required=False) or None
        try:
            handle = await self.handle_for(session_ref)
        except LookupError as exc:
            await self.send_error("no_session", str(exc), rid)
            return
        if self.busy(handle.id):
            await self.send_error("busy", "a turn is already running in that session", rid)
            return
        session_id = handle.id
        from agent.pipeline.budget import default_budget

        # The ceiling run.py binds when nobody bound one, which is this
        # server's case: a client's "12 of 40 calls" needs the 40.
        await self.reply("event", rid, session_id=session_id,
                         event={"type": "started", "session_id": session_id,
                                "budget_max": default_budget().max_model_calls})

        def events(event: dict) -> None:
            frame = protocol.encode("event", session_id=session_id, event=event)
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.send(frame)))

        from agent.embed import SUBPROCESS_TOOLS

        off_phone = SUBPROCESS_TOOLS if self.no_exec else ()
        tools, guidance, disabled, seats = [], "", off_phone, {}
        if self.phone is not None:
            from agent.phone import (PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, PHONE_SEATS,
                                     JsonBackend, phone_tools)

            tools = phone_tools(JsonBackend(self.phone))
            guidance, disabled, seats = PHONE_GUIDANCE, PHONE_DISABLED_STANDING_TOOLS, PHONE_SEATS

        def run() -> None:
            handle.run(text, events=events, tools=tools, guidance=guidance, disabled_tools=disabled,
                       seats=seats, phone=PHONE_MODES[mode], off_phone_disabled_tools=off_phone)

        self.turns[session_id] = self.loop.run_in_executor(self.executor, run)

    async def sessions(self, message: dict, rid: str | int | None = None) -> None:
        op = str(message.get("op") or "list")
        try:
            if op == "list":
                limit = message.get("limit")
                try:
                    limit = max(1, min(int(limit), 500)) if limit else 20
                except (TypeError, ValueError):
                    await self.send_error("invalid", "limit is a number", rid)
                    return
                runtime = await self.runtime_async()
                rows = await self.blocking(runtime.list_sessions, limit=limit)
                await self.reply("sessions_result", rid, op=op, sessions=rows)
            elif op == "open":
                handle = await self.handle_for(None, _sid(message.get("ref"), ref=True, required=False) or None)
                await self.reply("sessions_result", rid, op=op, session_id=handle.id,
                                 title=handle.title, turns=handle.turns)
            elif op == "transcript":
                ref = _sid(message.get("ref") or "last", ref=True)
                runtime = await self.runtime_async()
                data = await self.blocking(runtime.transcript, ref)
                if rid is not None:
                    data = {"session_id": data.pop("id"), **data}
                await self.reply("sessions_result", rid, op=op, **data)
            elif op == "delete":
                sid = _sid(message.get("session_id"))
                if self.busy(sid):
                    await self.send_error("busy", "that session is running a turn; stop it first", rid)
                    return
                handle = self.handles.pop(sid, None)
                if handle is not None:
                    await self.blocking(handle.close)
                runtime = await self.runtime_async()
                deleted = await self.blocking(runtime.delete_session, sid)
                await self.reply("sessions_result", rid, op=op, session_id=sid, deleted=deleted)
            else:
                await self.send_error("unknown", f"unknown sessions op {op!r}", rid)
        except LookupError as exc:
            await self.send_error("no_session", str(exc), rid)

    async def close(self) -> None:
        """Best-effort teardown: nothing here may raise, but everything that
        goes wrong is logged as a warning so a field problem can be read."""
        if self.phone is not None:
            self.phone.fail_all("the client disconnected")
        pending = [t for t in (self._worker, *self._tasks) if t is not None]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for handle in self.handles.values():
            try:
                handle.cancel()
            except Exception:
                logger.warning("serve: cancel on close failed for %s", handle.id, exc_info=True)
        for session_id, future in list(self.turns.items()):
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=10)
            except Exception:
                logger.warning("serve: turn %s did not finish within the close window", session_id)
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                logger.warning("serve: close failed for %s", handle.id, exc_info=True)


class OttoServer:
    def __init__(self, token: str, *, allowed_origins: tuple[str, ...] = (),
                 max_workers: int = MAX_TURN_WORKERS, no_exec: bool = False) -> None:
        self.token = token
        self.no_exec = no_exec
        self.allowed_origins = tuple(o.rstrip("/").lower() for o in allowed_origins)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="otto-turn")
        self.ops_executor = ThreadPoolExecutor(max_workers=MAX_OP_WORKERS, thread_name_prefix="otto-ops")

    def check_origin(self, connection, request):
        """Refuse a browser. A native client sends no Origin header; a page
        in a browser always does, and a page has no business on this port
        unless `allowed_origins` names its origin. The token would stop it
        anyway; this stops it before the token is ever tried."""
        origin = request.headers.get("Origin")
        if origin is None or origin.rstrip("/").lower() in self.allowed_origins:
            return None
        logger.warning("serve: refused a connection from origin %s", origin)
        return connection.respond(HTTPStatus.FORBIDDEN, "origin not allowed\n")

    async def handler(self, websocket) -> None:
        loop = asyncio.get_running_loop()
        connection = Connection(websocket, self.token, loop, self.executor, no_exec=self.no_exec,
                                ops_executor=self.ops_executor)
        try:
            try:
                first = await asyncio.wait_for(websocket.recv(), timeout=15)
                await connection.hello(first)
            except ProtocolError as exc:
                await connection.send_error("hello", str(exc))
                return
            except asyncio.TimeoutError:
                logger.debug("serve: a client connected and sent no hello within 15s")
                return
            except Exception:
                logger.debug("serve: the hello exchange failed", exc_info=True)
                return
            async for raw in websocket:
                try:
                    await connection.dispatch(raw)
                except Exception:  # only a send on a socket that has gone
                    logger.debug("serve: could not answer a message", exc_info=True)
        finally:
            await connection.close()

    async def run(self, host: str, port: int, *, ready: asyncio.Event | None = None) -> None:
        from websockets.asyncio.server import serve

        try:
            async with serve(self.handler, host, port, max_size=protocol.MAX_FRAME_BYTES,
                             process_request=self.check_origin) as server:
                self.sockets = server.sockets
                if ready is not None:
                    ready.set()
                await server.serve_forever()
        finally:
            # The pools belong to this server: a test that builds several
            # servers in one process must not keep their threads around.
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.ops_executor.shutdown(wait=False, cancel_futures=True)


def bound_port(server: OttoServer) -> int:
    for sock in getattr(server, "sockets", ()) or ():
        return sock.getsockname()[1]
    return 0
