"""The server: one connection, one runtime, any number of sessions.

Built on `websockets` (the `[serve]` extra) and agent/embed.py. Each turn runs
`SessionHandle.run` on a worker thread, exactly as the TUI does; its events
are forwarded to the client as they happen, and its phone tools reach the
client through agent/server/proxy.py. Everything a person could do in the
app -- start a turn, answer a question, stop, list and resume sessions -- is
one message type (agent/server/protocol.py).

Authentication is a shared token from `hello`, compared in constant time.
The server binds loopback by default; anything wider is the operator's call
(`otto serve --host 0.0.0.0` behind a network they trust).
"""
from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any

from agent.server import protocol
from agent.server.proxy import SocketPhone

logger = logging.getLogger(__name__)


class ProtocolError(Exception):
    pass


#: How many sessions one connection may hold open. A phone uses one or two;
#: the cap bounds what a client holding the token can spend, which is the
#: one thing the shared-token trust model does not otherwise limit.
MAX_SESSIONS_PER_CONNECTION = 8


class Connection:
    """State for one client socket."""

    def __init__(self, websocket, token: str, loop: asyncio.AbstractEventLoop) -> None:
        self.ws = websocket
        self.token = token
        self.loop = loop
        self.phone: SocketPhone | None = None
        self.capabilities: set[str] = set()
        self.handles: dict[str, Any] = {}
        self.turns: dict[str, asyncio.Future] = {}
        self._runtime = None

    # -- plumbing ----------------------------------------------------------

    async def send(self, frame: str) -> None:
        await self.ws.send(frame)

    async def send_error(self, code: str, message: str) -> None:
        await self.send(protocol.encode("error", code=code, message=message))

    def runtime(self):
        if self._runtime is None:
            from agent import embed

            self._runtime = embed.Runtime()
        return self._runtime

    def handle_for(self, session_id: str | None, ref: str | None = None):
        if session_id and session_id in self.handles:
            return self.handles[session_id]
        if len(self.handles) >= MAX_SESSIONS_PER_CONNECTION:
            raise LookupError(f"this connection already holds {MAX_SESSIONS_PER_CONNECTION} sessions; "
                              "delete one first")
        handle = self.runtime().open_session(ref or session_id)
        self.handles[handle.id] = handle
        return handle

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
                                        min_protocol=protocol.MIN_PROTOCOL))

    async def dispatch(self, raw: str | bytes) -> None:
        message = protocol.decode(raw)
        kind = message["type"]
        if kind == "invalid":
            await self.send_error("invalid", message["reason"])
        elif kind == "ping":
            await self.send(protocol.encode("pong"))
        elif kind == "device_result":
            if self.phone is None or not self.phone.resolve(message):
                await self.send_error("unexpected", "no device_call is waiting for that id")
        elif kind == "turn":
            await self.start_turn(message)
        elif kind == "answer":
            handle = self.handles.get(str(message.get("session_id") or ""))
            if handle is None or not handle.answer(str(message.get("thread_id") or ""), str(message.get("text") or "")):
                await self.send_error("no_question", "no question is waiting for that answer")
        elif kind == "cancel":
            handle = self.handles.get(str(message.get("session_id") or ""))
            if handle is not None:
                handle.cancel()
        elif kind == "sessions":
            await self.sessions(message)
        else:
            await self.send_error("unknown", f"unknown message type {kind!r}")

    async def start_turn(self, message: dict) -> None:
        text = str(message.get("text") or "").strip()
        if not text:
            await self.send_error("empty", "turn needs text")
            return
        try:
            handle = self.handle_for(message.get("session_id"))
        except LookupError as exc:
            await self.send_error("no_session", str(exc))
            return
        if handle.running:
            await self.send_error("busy", "a turn is already running in that session")
            return
        session_id = handle.id
        await self.send(protocol.encode("event", session_id=session_id,
                                        event={"type": "started", "session_id": session_id}))

        def events(event: dict) -> None:
            frame = protocol.encode("event", session_id=session_id, event=event)
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self.send(frame)))

        tools, guidance, disabled = [], "", ()
        if self.phone is not None:
            from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, JsonBackend, phone_tools

            tools = phone_tools(JsonBackend(self.phone))
            guidance, disabled = PHONE_GUIDANCE, PHONE_DISABLED_STANDING_TOOLS

        def run() -> None:
            handle.run(text, events=events, tools=tools, guidance=guidance, disabled_tools=disabled)

        self.turns[session_id] = self.loop.run_in_executor(None, run)

    async def sessions(self, message: dict) -> None:
        op = str(message.get("op") or "list")
        runtime = self.runtime()
        try:
            if op == "list":
                limit = message.get("limit")
                rows = runtime.list_sessions(limit=int(limit) if limit else 20)
                await self.send(protocol.encode("sessions_result", op=op, sessions=rows))
            elif op == "open":
                handle = self.handle_for(None, str(message.get("ref") or "") or None)
                await self.send(protocol.encode("sessions_result", op=op, session_id=handle.id,
                                                title=handle.title, turns=handle.turns))
            elif op == "transcript":
                await self.send(protocol.encode("sessions_result", op=op,
                                                **runtime.transcript(str(message.get("ref") or "last"))))
            elif op == "delete":
                sid = str(message.get("session_id") or "")
                handle = self.handles.pop(sid, None)
                if handle is not None:
                    handle.close()
                await self.send(protocol.encode("sessions_result", op=op, session_id=sid,
                                                deleted=runtime.delete_session(sid)))
            else:
                await self.send_error("unknown", f"unknown sessions op {op!r}")
        except LookupError as exc:
            await self.send_error("no_session", str(exc))

    async def close(self) -> None:
        """Best-effort teardown: nothing here may raise, but everything that
        goes wrong is logged at debug level so a field problem can be read."""
        if self.phone is not None:
            self.phone.fail_all("the client disconnected")
        for handle in self.handles.values():
            try:
                handle.cancel()
            except Exception:
                logger.debug("serve: cancel on close failed for %s", handle.id, exc_info=True)
        for session_id, future in list(self.turns.items()):
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=10)
            except Exception:
                logger.debug("serve: turn %s did not finish within the close window", session_id, exc_info=True)
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                logger.debug("serve: close failed for %s", handle.id, exc_info=True)


class OttoServer:
    def __init__(self, token: str) -> None:
        self.token = token

    async def handler(self, websocket) -> None:
        loop = asyncio.get_running_loop()
        connection = Connection(websocket, self.token, loop)
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
                except Exception as exc:  # one bad message must not drop the connection
                    logger.exception("serve: message failed")
                    await connection.send_error("failed", f"{type(exc).__name__}: {exc}")
        finally:
            await connection.close()

    async def run(self, host: str, port: int, *, ready: asyncio.Event | None = None) -> None:
        from websockets.asyncio.server import serve

        async with serve(self.handler, host, port, max_size=protocol.MAX_FRAME_BYTES) as server:
            self.sockets = server.sockets
            if ready is not None:
                ready.set()
            await server.serve_forever()


def bound_port(server: OttoServer) -> int:
    for sock in getattr(server, "sockets", ()) or ():
        return sock.getsockname()[1]
    return 0
