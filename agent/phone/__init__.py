"""The phone as a place Otto works: tools over a device a host drives.

The device-agnostic half. What a phone can do is a Protocol
(`agent.phone.backend.PhoneBackend`); a host -- the Android app, a WebSocket
proxy, a test fake -- supplies the implementation, and `agent.phone.tools.
phone_tools(backend)` turns it into the run-scoped tools the agent loop can
call (agent/pipeline/toolkit.py). The screen digest, the money guard's rules
and the guidance the model reads all live here, so they are tested by this
suite and versioned with agent/embed.py's API.
"""
from agent.phone.backend import JsonBackend, PhoneBackend, PhoneError
from agent.phone.tools import (
    PHONE_DISABLED_STANDING_TOOLS,
    PHONE_GUIDANCE,
    phone_tools,
)

#: The seats a phone run binds (agent/router/overrides.py bind_seats). Only
#: the judge moves: a phone turn's evaluator is a check of a screen the agent
#: already read, while a person holds the phone waiting, and an installation
#: pin on a large reasoning model (routes.json pins opus there) makes it the
#: slowest seat of the turn. Its prompt, tools and rejections are unchanged;
#: a coding run never binds this, so it keeps whatever routes.json says.
PHONE_SEATS: dict[str, str] = {"evaluate": "gemini:gemini-3.8-flash"}

__all__ = [
    "JsonBackend", "PhoneBackend", "PhoneError",
    "PHONE_DISABLED_STANDING_TOOLS", "PHONE_GUIDANCE", "PHONE_SEATS", "phone_tools",
]
