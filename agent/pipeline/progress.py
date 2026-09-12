"""What the run is doing right now, for whoever is watching it.

The graph streams one update per NODE. `agent` is a node, and inside it a
run spends two to twenty model calls and every tool call it makes -- so
between START and the first thing a front end can show, a person watches an
unchanging screen for as long as the whole turn takes. Measured on the
twenty golden items: 828 seconds of wall time, 96% of it inside model
requests, and not one graph update in the middle of any of it. The longest
item ran 131 seconds with nothing on screen.

This is the seam that fixes that, built the way every other run-scoped fact
in this package is built (`bind_budget`, `bind_workspace`,
`bind_command_runner`, `bind_extra_tools`): a contextvar holding a sink,
bound by whoever is watching, absent everywhere else. Nothing bound is the
normal case and makes `report()` a dict build and a function call that
returns immediately -- eval harnesses and tests bind nothing and see no
change at all.

Deliberately a callback and not another stream: a stream would have to be
threaded back through LangGraph's own channels, and the thing being
reported is not state. Nothing here is ever read back by the graph, nothing
is persisted, and a sink that raises is swallowed -- a broken progress
display must not be able to kill a run that is otherwise going fine.

Cancellation rides along for the same reason, and in the same place. A
front end that wants to stop a run sets the Event it bound; `_call` checks
it before spending, and raises Cancelled. It is cooperative on purpose: the
worker is a real thread doing real network I/O, there is no safe way to
kill it from outside, and the longest a cancel can take to land is one
model call. What it buys is the difference between "wait out the 28-minute
run you started by accident" and "press escape".
"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator


class Cancelled(Exception):
    """Raised inside a run whose watcher asked it to stop.

    A distinct type so a front end can tell "you stopped this" apart from
    "this broke", and print accordingly.
    """


@dataclass(frozen=True)
class Progress:
    """One thing that just happened, or is about to.

    `kind` is what happened; `text` is the short human phrase for it; the
    rest is whatever that kind carries. A consumer should render `text` and
    read the fields it knows, never assume a field is there -- kinds are
    expected to be added, and a front end written against today's set must
    keep working when they are.
    """

    kind: str
    text: str = ""
    #: Model requests spent by this run so far, when a budget is bound.
    calls: int = 0
    #: Seconds since the watcher started watching.
    elapsed: float = 0.0
    #: The reply so far, on a `partial`. Every frame is the WHOLE reply, not
    #: the next slice: a diffusing route's chunks are successive refinements
    #: of one answer, and a consumer that appends them renders garbage.
    partial: str = ""
    detail: dict[str, Any] | None = None


Sink = Callable[[Progress], None]

_sink: contextvars.ContextVar[Sink | None] = contextvars.ContextVar(
    "otto_progress_sink", default=None,
)
_cancel: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "otto_progress_cancel", default=None,
)
_started: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "otto_progress_started", default=None,
)


@contextmanager
def bind_progress(sink: Sink | None, *,
                  cancel: threading.Event | None = None) -> Iterator[None]:
    """Watch the run inside this block.

    `sink` is called from whichever thread the run is on -- a UI toolkit's
    widget tree is almost never safe to touch from there, so a sink that
    draws must marshal (Textual's `call_from_thread`, and see
    agent/cli/tui.py).
    """
    tokens = (_sink.set(sink), _cancel.set(cancel), _started.set(time.monotonic()))
    try:
        yield
    finally:
        _sink.reset(tokens[0])
        _cancel.reset(tokens[1])
        _started.reset(tokens[2])


def watching() -> bool:
    """Whether anything is listening. Worth checking before building a
    `detail` dict that is expensive to assemble."""
    return _sink.get() is not None


def check_cancelled() -> None:
    """Raise if the watcher has asked this run to stop. Cheap enough to call
    before every model request, which is exactly where it is called."""
    event = _cancel.get()
    if event is not None and event.is_set():
        raise Cancelled("stopped")


def report(kind: str, text: str = "", **fields: Any) -> None:
    """Tell the watcher what is happening. A no-op when nobody is watching.

    Never raises. A front end whose renderer has a bug would otherwise take
    the run down with it, and the run is the part that matters.
    """
    sink = _sink.get()
    if sink is None:
        return
    started = _started.get()
    try:
        sink(Progress(
            kind=kind, text=text,
            elapsed=0.0 if started is None else time.monotonic() - started,
            **fields,
        ))
    except Exception:                       # a broken display, not a broken run
        pass
