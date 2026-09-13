"""Langfuse when it can be reached, and silence when it cannot.

Every pipeline entry point traces a run to Langfuse through a LangChain
callback handler and one observation span. The SDK assumes its host is up:
with `LANGFUSE_BASE_URL` pointing at a collector that is not running, the
exporter thread retries every batch and prints a connection error to stderr
every couple of seconds for the life of the process. In the REPL that is the
whole screen; in the TUI it is a crash log after quitting. A tracing backend
must never be louder than the run it is tracing.

So the host is probed ONCE per process, with a plain TCP connect and a short
timeout, before any client exists. Unreachable means one warning, no client,
no handler, no span, and `LANGFUSE_TRACING_ENABLED=false` in the environment
so anything else in the process that calls `get_client()` later is quiet too.
No keys configured means the same silence without the warning: nothing was
asked for.

`closing_in_foreign_context` is the other half of the same quit-time crash.
A streaming run is a generator that holds contextvars and an OpenTelemetry
context across its `yield`s. When the front end exits mid-turn, the generator
is finalised by whichever thread the garbage collector happens to run in,
and a token created in one context cannot be reset in another: Python raises
`ValueError: ... was created in a different Context`, which the interpreter
then prints as "Exception ignored in: <generator ...>". The binders in this
package guard their own resets; this predicate lets the generators recognise
the one the SDK raises and finish quietly instead.
"""
from __future__ import annotations

import logging
import os
import socket
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://cloud.langfuse.com"
#: A TCP connect, not an HTTP round trip: the question is only whether
#: anything is listening, and a second of startup latency is the most a
#: check nobody asked for may cost.
PROBE_TIMEOUT_S = 1.0

_FALSE = {"false", "0", "no", "off"}
_state: dict[str, bool | None] = {"ready": None}


def configured_base_url() -> str:
    return (
        os.environ.get("LANGFUSE_BASE_URL")
        or os.environ.get("LANGFUSE_HOST")
        or DEFAULT_BASE_URL
    )


def _keys_present() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")) and bool(os.environ.get("LANGFUSE_SECRET_KEY"))


def _probe(url: str, timeout: float = PROBE_TIMEOUT_S) -> bool:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError:
        return False
    return True


def _decide() -> bool:
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "").strip().lower() in _FALSE:
        return False
    if not _keys_present():
        return False
    url = configured_base_url()
    if _probe(url):
        return True
    # setdefault, not set: a person who wrote the variable keeps their value.
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    logger.warning(
        "Langfuse at %s is unreachable; tracing is off for this process. "
        "Start it, fix LANGFUSE_BASE_URL, or unset the LANGFUSE_* keys to "
        "silence this.",
        url,
    )
    return False


def langfuse_ready() -> bool:
    """Whether this process traces to Langfuse. Decided once, on first ask."""
    if _state["ready"] is None:
        _state["ready"] = _decide()
    return bool(_state["ready"])


def reset_for_tests() -> None:
    _state["ready"] = None


class NullSpan:
    """What a run holds instead of a Langfuse observation when tracing is off.

    Accepts every call `_score` and the entry points make, and has no trace
    id, so `__trace_id__` is None rather than a value nobody can look up.
    """

    trace_id = None

    def update(self, **kwargs) -> None:
        pass

    def score_trace(self, **kwargs) -> None:
        pass


def callback_handler():
    """A LangChain callback handler for the graph, or None when tracing is off."""
    if not langfuse_ready():
        return None
    from langfuse.langchain import CallbackHandler

    return CallbackHandler()


@contextmanager
def observe_run(*, session_id: str, name: str, input: str, tags: list[str]) -> Iterator[object]:
    """The observation span a run is scored on, or a `NullSpan`."""
    if not langfuse_ready():
        yield NullSpan()
        return
    from langfuse import get_client, propagate_attributes

    client = get_client()
    with propagate_attributes(trace_name="otto:pipeline", session_id=session_id, tags=tags):
        with client.start_as_current_observation(name=name, as_type="agent", input=input) as span:
            yield span


def closing_in_foreign_context(exc: BaseException) -> bool:
    """True for the ValueError a context token raises when reset from a
    context other than the one that created it: the signature of a generator
    finalised on another thread, and nothing a run can act on."""
    return isinstance(exc, ValueError) and "different Context" in str(exc)
