"""The messages between `otto serve` and a client, in one place.

One JSON object per WebSocket text frame, every object a `type`. Versioned by
`PROTOCOL_VERSION`; the server accepts any client from `MIN_PROTOCOL` up, so
an app pinned to an older otto keeps working across one bump.

client -> server
    hello           {protocol_version, token, device?, capabilities?: ["phone"]}
    turn            {session_id?, text, phone?}      start a turn (one at a time per session);
                                                     phone: auto (default) | on | off
    answer          {session_id, thread_id, text}    answer an `ask`
    cancel          {session_id}
    device_result   {id, ok, data?, error?}          the reply to a `device_call`
    sessions        {op: list|open|delete|transcript, ref?, session_id?, limit?}
    ping            {}

server -> client
    hello_ok        {otto_version, api_version, protocol_version, min_protocol}
    event           {session_id, event}              an agent/embed.py event dict, after
                                                     {type: started, session_id, budget_max}
    device_call     {id, method, args, timeout}      run a PhoneBackend method on the phone
    sessions_result {op, ...}
    error           {code, message}                  codes include invalid_session, busy,
                                                     no_session, invalid, no_phone
    pong            {}
"""
from __future__ import annotations

import json
from typing import Any

PROTOCOL_VERSION = 1
MIN_PROTOCOL = 1

#: How long a device call may take before the tool reports a timeout. A
#: Play Store install waits on the network; everything else is sub-second.
DEVICE_CALL_TIMEOUT_S = 30.0
INSTALL_TIMEOUT_S = 180.0

#: Frames larger than this are refused: a screenshot is at most a few MB.
MAX_FRAME_BYTES = 12 * 1024 * 1024


def encode(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, ensure_ascii=False)


def decode(raw: str | bytes) -> dict[str, Any]:
    """The object in a frame, or {"type": "invalid", "reason": ...}."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return {"type": "invalid", "reason": f"not JSON: {exc}"}
    if not isinstance(data, dict) or not isinstance(data.get("type"), str):
        return {"type": "invalid", "reason": "a frame is a JSON object with a string 'type'"}
    return data
