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

__all__ = [
    "JsonBackend", "PhoneBackend", "PhoneError",
    "PHONE_DISABLED_STANDING_TOOLS", "PHONE_GUIDANCE", "phone_tools",
]
