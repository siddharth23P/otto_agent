"""Per-run binding of how much a run may SPEND, counted in model calls.

Otto had no budget of any kind until now. The only stop was LangGraph's
`recursion_limit` (agent/pipeline/nodes.py's `_RECURSION_SAFETY_NET`), which
counts super-steps rather than money, and `MAX_TOOL_ITERATIONS`, which caps one
node rather than one run. Neither is a ceiling anybody chose.

Two harnesses already built the right shape and bound it to the wrong thing.
agent/eval/terminal_bench.py and agent/eval/claw_bench.py each implement a
two-stage deadline -- at 0.8 of the budget every tool refuses and says to wrap
up, past the end it stops -- and both check it inside a TOOL. Tools cost
nothing: measured across four Claw-Eval tasks, every tool call a task made
totalled 0.1 to 0.4 seconds, while the run took 119 to 946. All of the rest was
model time, which a tool-level deadline cannot see. C01 ran 1096 seconds against
a 900-second budget without the deadline ever firing.

So the check moves to the one place every model call passes through:
`_call`. That includes its retry paths -- the empty-stream retry and the
diffusion max_tokens doubling -- because those are real HTTP requests that cost
real money, and a budget that did not count them would undercount by up to 3x
exactly when a run is going badly.

TWO STAGES, and the first is the one that produces an answer. Past `wrap_up_at`
the loop is told, once, to stop exploring and answer now; past `hard_at` it
stops. A run killed mid-command reports nothing, where one told to wrap up
writes down what it found -- which is why `check()` returns text rather than
raising, and why exhaustion is a `stop` the caller handles rather than an
exception that unwinds the graph.

NOTHING BOUND IS A VALID STATE and it is what an ordinary `otto chat` turn
uses unless OTTO_MAX_MODEL_CALLS says otherwise. Same contextvar shape as
agent/pipeline/execution.py, workspace.py and toolkit.py, for the same reason:
`_call` is reached through a graph and cannot see the run it belongs to.
"""
from __future__ import annotations

import contextvars
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Literal

#: The default ceiling on model calls for one run, when nothing binds a budget
#: and nothing sets OTTO_MAX_MODEL_CALLS. Provisional: it is a number chosen to
#: be well clear of the observed working range rather than one measured off a
#: distribution. The measurement to replace it is a Claw-Eval batch run with a
#: deliberately high cap, reading `model_calls` at p95.
#:
#: There is deliberately NO wall-clock default. A benchmark harness knows its
#: own deadline and binds one; a chat turn with a person waiting should not be
#: killed by a clock, and call count is the thing that maps to spend.
DEFAULT_MAX_MODEL_CALLS = 120

#: Where the first stage starts, as a fraction of the budget. Carried over from
#: the two harnesses that already measured this working.
WRAP_UP_FRACTION = 0.8

#: Said once, when the first stage opens. Deliberately the wording both
#: harnesses arrived at independently, because it is what actually turns a
#: run that is out of time into a run that produced something.
WRAP_UP_NOTE = (
    "Time is nearly up. Stop exploring, make sure any change you made is "
    "actually written, and give your FINAL answer now."
)

Phase = Literal["ok", "wrap_up", "spent"]


@dataclass
class Budget:
    """What this run may spend, and how much of it is gone.

    Mutable on purpose: `spend()` is called from `_call`, several graph nodes
    deep, with no way to hand a new value back up. The contextvar holds one
    instance for the run and every node shares it.
    """

    max_model_calls: int | None = None
    #: `time.monotonic()` deadlines, or None for "no clock on this run".
    hard_at: float | None = None
    wrap_up_at: float | None = None
    calls: int = field(default=0)
    #: So the wrap-up note is said once rather than on every iteration, which
    #: would be both noise and a growing prompt.
    warned: bool = field(default=False)

    @classmethod
    def of(
        cls,
        seconds: float | None = None,
        *,
        max_model_calls: int | None = None,
        wrap_up_fraction: float = WRAP_UP_FRACTION,
        margin: float = 10.0,
    ) -> "Budget":
        """A budget of `seconds` from now, `max_model_calls`, or both.

        `margin` ends the run slightly early so it finishes by answering rather
        than by being killed -- the same reasoning, and roughly the same number,
        as the two harnesses use.
        """
        hard_at = wrap_up_at = None
        if seconds is not None:
            usable = max(float(seconds) - margin, 1.0)
            now = time.monotonic()
            hard_at = now + usable
            wrap_up_at = now + usable * wrap_up_fraction
        return cls(
            max_model_calls=max_model_calls,
            hard_at=hard_at,
            wrap_up_at=wrap_up_at,
        )

    @classmethod
    def until(cls, monotonic_deadline: float, **kwargs) -> "Budget":
        """A budget sharing a clock a caller already has -- what a harness with
        its own `Deadline` binds, so the two cannot disagree about when the
        task ends."""
        return cls.of(max(monotonic_deadline - time.monotonic(), 1.0), **kwargs)

    def spend(self) -> None:
        """Record one model request. Called from `_call`, once per HTTP
        request rather than once per logical call, so a retry storm is
        visible to the ceiling that is meant to stop it."""
        self.calls += 1

    def phase(self) -> Phase:
        """Where this run is: working, wrapping up, or done."""
        if self.max_model_calls is not None and self.calls >= self.max_model_calls:
            return "spent"
        if self.hard_at is not None and time.monotonic() >= self.hard_at:
            return "spent"
        if self.max_model_calls is not None:
            if self.calls >= self.max_model_calls * WRAP_UP_FRACTION:
                return "wrap_up"
        if self.wrap_up_at is not None and time.monotonic() >= self.wrap_up_at:
            return "wrap_up"
        return "ok"

    def wrap_up_once(self) -> str | None:
        """The note to append, the first time the run enters its last stretch.
        None afterwards, and None while there is budget left."""
        if self.warned or self.phase() != "wrap_up":
            return None
        self.warned = True
        return WRAP_UP_NOTE

    def spent(self) -> bool:
        return self.phase() == "spent"


_current: contextvars.ContextVar[Budget | None] = contextvars.ContextVar(
    "otto_current_budget", default=None,
)


@contextmanager
def bind_budget(budget: Budget | None) -> Iterator[Budget | None]:
    """Make `budget` the ceiling for this block and anything it calls.

    None unbinds, so a harness can bind per task without one task's spent
    budget leaking into the next. A resumed run must bind a FRESH budget: the
    time a person spent answering an `ask_user` question is not the agent's,
    and an absolute deadline carried across would arrive already spent.
    """
    token = _current.set(budget)
    try:
        yield budget
    finally:
        _current.reset(token)


def current_budget() -> Budget | None:
    """The budget bound by the innermost `bind_budget()`, or None."""
    return _current.get()


def default_budget() -> Budget:
    """The ceiling a run gets when nobody bound one. Env-overridable, because
    the number above is provisional and somebody measuring should not have to
    edit the source to try another."""
    raw = os.environ.get("OTTO_MAX_MODEL_CALLS", "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_MAX_MODEL_CALLS
    except ValueError:
        limit = DEFAULT_MAX_MODEL_CALLS
    return Budget(max_model_calls=max(limit, 1))
