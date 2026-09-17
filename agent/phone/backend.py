"""What a phone must provide, and the adapter that turns a JSON-speaking
bridge into it.

The vocabulary is agent/pipeline/native.py's `Device` (launch, texts,
click_text, tap, type_text, press, screenshot) grown to what a real phone
session needs: the accessibility tree as data rather than a list of strings,
the app in front, settings pages, the app store. Every method returns plain
Python or raises `PhoneError`, so the tools in agent/phone/tools.py have one
failure shape to translate into a ToolResult.

THE GUARD LIVES ON THE PHONE. A backend refuses an action inside a payment
app or on a payment screen by raising `PhoneError(code="guard",
handover=True)`. The Python side (agent/phone/guard.py) pre-checks with the
same rules so a refused call costs no round trip, but the backend's verdict
is the one that counts: it sees the real window flags and the real package,
and it cannot be talked out of it by anything the model writes.

`JsonBackend` is what a Kotlin bridge or a socket proxy becomes: an object
whose methods take positional arguments and return JSON strings of
`{"ok": true, "data": ...}` or `{"ok": false, "error": {"code", "message",
"handover"}}` (a screenshot may return raw bytes or base64). The argument
order of each method is the contract, written once here.
"""
from __future__ import annotations

import base64
import json
from typing import Any, Protocol, runtime_checkable

#: Error codes a backend may use. `guard` is the money guard handing the
#: phone to the person; `refused` is the guard declining one action (a pay
#: button on an ordinary page) and nothing more; `stale` means the snapshot
#: a node index came from is no longer on screen; `unsupported` is a
#: capability the device lacks (no accessibility service running, no
#: screenshot permission); `timeout` a gesture or capture that never
#: completed.
ERROR_CODES = ("guard", "refused", "stale", "unsupported", "invalid", "failed", "timeout")


class PhoneError(Exception):
    """A step the phone could not or would not do, with the reason a person
    can act on. `handover=True` means the phone has stopped acting for the
    model and is waiting for the person."""

    def __init__(self, message: str, *, code: str = "failed", handover: bool = False):
        super().__init__(message)
        self.code = code if code in ERROR_CODES else "failed"
        self.handover = handover


@runtime_checkable
class PhoneBackend(Protocol):
    """Every method returns a dict (documented per method) or raises
    PhoneError. Snapshots are agent/phone/digest.py's shape."""

    def tree(self) -> dict: ...                       # a snapshot of the screen
    def foreground(self) -> dict: ...                 # {"package", "label"}
    def tap(self, x: int, y: int) -> dict: ...        # {"done", "after": snapshot}
    def tap_node(self, snapshot_id: str, node: int, *, long: bool = False,
                 commit: bool = False) -> dict: ...   # {"done", "after"}
    def type_text(self, text: str, node: int | None = None) -> dict: ...
    def press(self, key: str) -> dict: ...            # back|home|recents|enter
    def swipe(self, direction: str, x: int | None = None,
              y: int | None = None) -> dict: ...  # the way the finger moves, from x,y
    def scroll(self, direction: str, node: int | None = None) -> dict: ...
    def screenshot(self) -> bytes: ...                # PNG or JPEG bytes
    def apps(self) -> dict: ...                       # {"apps": [{"label", "package"}]}
    def launch(self, package: str) -> dict: ...       # {"package", "label", "after"}
    def open_settings(self, page: str, package: str = "") -> dict: ...  # {"page", "after"}
    def install(self, package: str = "", query: str = "") -> dict: ...  # {"state", "after"}


class JsonBackend:
    """A PhoneBackend over an object whose methods return JSON strings.

    `raw` may be a Chaquopy Java class (static methods), a socket proxy, or a
    Python fake; each method is looked up by name and called with positional
    arguments in the order below. Optional arguments are passed explicitly
    (`-1` for "no node", `""` for "no package"), because a Java static method
    has no defaults.
    """

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def _call(self, method: str, *args: Any) -> Any:
        fn = getattr(self._raw, method, None)
        if fn is None:
            raise PhoneError(f"the phone bridge has no {method}()", code="unsupported")
        try:
            reply = fn(*args)
        except PhoneError:
            raise
        except Exception as exc:  # a bridge exception is a failed step, not a dead loop
            raise PhoneError(f"{method} failed on the phone: {exc}") from exc
        return _unwrap(method, reply)

    def tree(self) -> dict:
        return self._call("tree")

    def foreground(self) -> dict:
        return self._call("foreground")

    def tap(self, x: int, y: int) -> dict:
        return self._call("tap", int(x), int(y))

    def tap_node(self, snapshot_id: str, node: int, *, long: bool = False, commit: bool = False) -> dict:
        return self._call("tap_node", str(snapshot_id), int(node), bool(long), bool(commit))

    def type_text(self, text: str, node: int | None = None) -> dict:
        return self._call("type_text", str(text), -1 if node is None else int(node))

    def press(self, key: str) -> dict:
        return self._call("press", str(key))

    def swipe(self, direction: str, x: int | None = None, y: int | None = None) -> dict:
        # The start point is sent only when there is one: an app built before
        # it existed has a one-argument swipe, and a bridge that has both
        # takes (direction, x, y).
        if x is None or y is None:
            return self._call("swipe", str(direction))
        return self._call("swipe", str(direction), int(x), int(y))

    def scroll(self, direction: str, node: int | None = None) -> dict:
        return self._call("scroll", str(direction), -1 if node is None else int(node))

    def screenshot(self) -> bytes:
        reply = self._call("screenshot")
        if isinstance(reply, (bytes, bytearray)):
            return bytes(reply)
        if isinstance(reply, dict):
            encoded = reply.get("png_b64") or reply.get("image_b64") or ""
            try:
                data = base64.b64decode("".join(str(encoded).split()), validate=True)
            except (ValueError, TypeError) as exc:
                raise PhoneError(f"the screenshot did not transfer cleanly: {exc}") from exc
            if data:
                return data
        raise PhoneError("the phone returned no image", code="unsupported")

    def apps(self) -> dict:
        return self._call("apps")

    def launch(self, package: str) -> dict:
        return self._call("launch", str(package))

    def open_settings(self, page: str, package: str = "") -> dict:
        return self._call("open_settings", str(page), str(package or ""))

    def install(self, package: str = "", query: str = "") -> dict:
        return self._call("install", str(package or ""), str(query or ""))

    # Optional, not part of PhoneBackend: the phone's own action registry (the Android app's
    # actions/ActionCatalog.kt). A backend without it simply offers no phone_action tool.

    def actions(self) -> list[dict]:
        """[{"name", "summary", "effect": read|change|confirm, "params": [{"name", "type", ...}]}]."""
        reply = self._call("actions")
        return list(reply.get("actions") or []) if isinstance(reply, dict) else []

    def run_action(self, name: str, args: dict) -> dict:
        """{"done", "handed_over"?, "data"?}"""
        return self._call("run_action", str(name), json.dumps(args or {}))


def _unwrap(method: str, reply: Any) -> Any:
    """The `data` of an ok envelope, a PhoneError for an error envelope, and
    bytes or a dict passed through for a bridge that already speaks Python."""
    if isinstance(reply, (bytes, bytearray)):
        return bytes(reply)
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except ValueError as exc:
            raise PhoneError(f"{method} returned something that is not JSON: {reply[:120]!r}") from exc
    if not isinstance(reply, dict):
        raise PhoneError(f"{method} returned {type(reply).__name__}, not an object")
    if "ok" not in reply:
        return reply  # a Python fake handing back data directly
    if reply.get("ok"):
        data = reply.get("data")
        return data if isinstance(data, dict) else {"value": data}
    error = reply.get("error") or {}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    raise PhoneError(str(error.get("message") or f"{method} failed"),
                     code=str(error.get("code") or "failed"), handover=bool(error.get("handover")))
