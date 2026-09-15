"""A PhoneBackend whose phone is on the other end of the socket.

Built as the `raw` object agent/phone/backend.py's JsonBackend wraps: each
method sends one `device_call` and blocks the calling thread -- the turn's
worker thread, never the event loop -- until the matching `device_result`
arrives or the timeout passes. The envelope the phone answers with is the
same one the in-process bridge returns, so JsonBackend needs no second shape.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import itertools
import threading
from typing import Any, Callable, Coroutine

from agent.phone.backend import PhoneError
from agent.server.protocol import DEVICE_CALL_TIMEOUT_S, INSTALL_TIMEOUT_S, encode

Sender = Callable[[str], Coroutine[Any, Any, None]]


class SocketPhone:
    def __init__(self, send: Sender, loop: asyncio.AbstractEventLoop) -> None:
        self._send = send
        self._loop = loop
        self._ids = itertools.count(1)
        self._pending: dict[int, concurrent.futures.Future] = {}
        self._lock = threading.Lock()

    # -- called from the event loop when a device_result arrives ------------

    def resolve(self, message: dict) -> bool:
        with self._lock:
            future = self._pending.pop(int(message.get("id", -1)), None)
        if future is None:
            return False
        if not future.done():
            future.set_result(message)
        return True

    def fail_all(self, reason: str) -> None:
        with self._lock:
            pending, self._pending = list(self._pending.values()), {}
        for future in pending:
            if not future.done():
                future.set_result({"ok": False, "error": {"code": "failed", "message": reason}})

    # -- called from the turn thread ---------------------------------------

    def _call(self, method: str, *args: Any, timeout: float = DEVICE_CALL_TIMEOUT_S) -> dict:
        call_id = next(self._ids)
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._lock:
            self._pending[call_id] = future
        frame = encode("device_call", id=call_id, method=method, args=list(args), timeout=timeout)
        try:
            asyncio.run_coroutine_threadsafe(self._send(frame), self._loop).result(timeout=5.0)
        except Exception as exc:
            with self._lock:
                self._pending.pop(call_id, None)
            raise PhoneError(f"could not reach the phone: {exc}") from exc
        try:
            reply = future.result(timeout=timeout + 5.0)
        except concurrent.futures.TimeoutError:
            with self._lock:
                self._pending.pop(call_id, None)
            raise PhoneError(f"the phone did not answer {method} within {int(timeout)}s", code="timeout")
        # The whole envelope: JsonBackend reads ok/data/error itself.
        return {"ok": bool(reply.get("ok")), "data": reply.get("data"), "error": reply.get("error")}

    def tree(self):
        return self._call("tree")

    def foreground(self):
        return self._call("foreground")

    def tap(self, x, y):
        return self._call("tap", x, y)

    def tap_node(self, snapshot_id, node, long, commit):
        return self._call("tap_node", snapshot_id, node, long, commit)

    def type_text(self, text, node):
        return self._call("type_text", text, node)

    def press(self, key):
        return self._call("press", key)

    def swipe(self, direction, *point):
        # (direction) or (direction, x, y): JsonBackend sends the start point
        # only when there is one, and the phone app takes either shape.
        return self._call("swipe", direction, *point)

    def scroll(self, direction, node):
        return self._call("scroll", direction, node)

    def screenshot(self):
        return self._call("screenshot")

    def apps(self):
        return self._call("apps")

    def launch(self, package):
        return self._call("launch", package)

    def open_settings(self, page, package):
        return self._call("open_settings", page, package)

    def install(self, package, query):
        return self._call("install", package, query, timeout=INSTALL_TIMEOUT_S)
