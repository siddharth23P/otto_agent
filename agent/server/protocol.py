"""The messages between `otto serve` and a client, in one place.

One JSON object per WebSocket text frame, every object a `type`. Versioned by
`PROTOCOL_VERSION`; the server accepts any client from `MIN_PROTOCOL` up, so
an app pinned to an older otto keeps working across one bump.

Protocol 2 (2026-09-15) adds, without changing a protocol-1 exchange:
  - `hello_ok.features`, the names of what this server does beyond protocol 1
    (`FEATURES`), sent to every client, so an app gates each screen on a name
    rather than on a version number.
  - an optional request `id` (a string of at most 64 characters, or an
    integer) on any client message but `hello` and `device_result`, echoed as
    `id` on the `*_result`, `pong` or `error` it causes, and on the `started`
    event frame of a turn. A message without one is answered byte for byte as
    in protocol 1. Because a protocol-1 `sessions_result{op: transcript}`
    already used `id` for the session, a transcript asked for WITH a request
    id carries the session's id as `session_id` instead.

client -> server
    hello           {protocol_version, token, device?, capabilities?: ["phone"]}
    turn            {session_id?, text, phone?}      start a turn (one at a time per session);
                                                     phone: auto (default) | on | off
    answer          {session_id, thread_id, text}    answer an `ask`
    cancel          {session_id}
    device_result   {id, ok, data?, error?}          the reply to a `device_call`
    sessions        {op: list|open|delete|transcript, ref?, session_id?, limit?}
                    {op: close|rename|export|usage, session_id, title?}   protocol 2
                    {op: import, data}                                   protocol 2
    ping            {}

server -> client
    hello_ok        {otto_version, api_version, protocol_version, min_protocol, features}
    event           {session_id, event}              an agent/embed.py event dict, after
                                                     {type: started, session_id, budget_max}
    device_call     {id, method, args, timeout}      run a PhoneBackend method on the phone
    sessions_result {op, ...}
                    close  {session_id, closed}      rename {session_id, title}
                    export {session_id, filename, data}   import {session_id, title, turns}
                    usage  {session_id, usage, turn_tokens, turn, title, turns}
    error           {code, message}                  codes include invalid_session, busy,
                                                     no_session, invalid, no_phone
    pong            {}
"""
from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 2
MIN_PROTOCOL = 1

#: `hello_ok.features`. A client checks for a name before offering what it
#: unlocks; a server older than a name simply does not send it.
FEATURES: tuple[str, ...] = (
    "ids", "turn.phone",
    "sessions.close", "sessions.rename", "sessions.export", "sessions.import", "sessions.usage",
)

#: The longest string request id echoed back.
MAX_REQUEST_ID_CHARS = 64

#: How long a device call may take before the tool reports a timeout. A
#: Play Store install waits on the network; everything else is sub-second.
DEVICE_CALL_TIMEOUT_S = 30.0
INSTALL_TIMEOUT_S = 180.0

#: Frames larger than this are refused: a screenshot is at most a few MB.
MAX_FRAME_BYTES = 12 * 1024 * 1024


def encode(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False)


def reply(kind: str, request_id: str | int | None = None, **fields: Any) -> str:
    """`encode`, with the request's id when it had one -- and exactly
    `encode` when it did not, which is what keeps protocol 1 unchanged."""
    if request_id is None:
        return encode(kind, **fields)
    return encode(kind, id=request_id, **fields)


def request_id(message: dict[str, Any]) -> str | int | None:
    """The message's request id, None when it has none, ValueError when it
    has one that is not a short string or an integer (a bool is neither,
    whatever Python thinks)."""
    rid = message.get("id")
    if rid is None:
        return None
    if isinstance(rid, int) and not isinstance(rid, bool) and abs(rid) < 2 ** 53:
        return rid
    if isinstance(rid, str) and 0 < len(rid) <= MAX_REQUEST_ID_CHARS:
        return rid
    raise ValueError(f"a request id is a string of 1-{MAX_REQUEST_ID_CHARS} characters or an integer")


def decode(raw: str | bytes) -> dict[str, Any]:
    """The object in a frame, or {"type": "invalid", "reason": ...}."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return {"type": "invalid", "reason": f"not JSON: {exc}"}
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        return {"type": "invalid", "reason": "a frame is a JSON object with a string 'type'"}
    return data
