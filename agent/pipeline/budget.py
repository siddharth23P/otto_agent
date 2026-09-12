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

NOTHING BOUND IS A VALID STATE for a direct caller of `_call`, but a real run
always has one: agent/pipeline/run.py binds `current_budget() or
default_budget()` around every entry point, so a harness's own budget wins and
an ordinary `otto chat` turn still gets the OTTO_MAX_MODEL_CALLS ceiling. This
docstring previously claimed that already happened, and it did not -- the
function was imported and never called, which made the env var dead and left an
interactive turn able to spend without limit. Same contextvar shape as
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

#: The least a turn of a multi-turn task can be given and still produce an
#: answer at all.
#:
#: A turn is not just the loop: it pays for the criteria call before the loop
#: and the judgment after it, so a share of three calls is spent before any
#: work happens. These are floors, not targets -- a turn that needs less
#: returns the rest to the turns after it.
#:
#: Both numbers come from a run that went wrong. Rationing 480 seconds across
#: C04's nine turns gave each one 53 seconds, and the grader's verdict was
#: "the provided conversation only contains user messages and lacks any
#: responses from the assistant" -- score 0.32 against 0.43 for the
#: unrationed run it was meant to improve. A share too small to answer with is
#: worse than no rationing at all, because one long answer beats none.
MIN_TURN_CALLS = 8
MIN_TURN_SECONDS = 90.0

#: Said once, when the first stage opens. Deliberately the wording both
#: harnesses arrived at independently, because it is what actually turns a
#: run that is out of time into a run that produced something.
WRAP_UP_NOTE = (
    "Time is nearly up. Stop exploring, make sure any change you made is "
    "actually written, and give your FINAL answer now."
)

#: How much of a run's budget is reconnaissance, before any of it is spent
#: committing to an approach.
#:
#: Habits 1 and 3 in the agent prompt already say to find out what state the
#: system is in before concluding, and to look wide before looking narrow.
#: That is the right instinct stated as guidance the model may or may not
#: follow. The failure it targets has a name -- premature exploitation,
#: committing to training-time priors before learning what the environment
#: actually allows -- and making exploration a phase with its own budget,
#: spent BEFORE execution, was worth +6.3 to +11.7 points.
#:
#: Small, because the same work reports that naive exploration HURT. A fifth
#: of the budget is enough to read the code a change touches and see what is
#: actually running; more than that is the exploration becoming the task.
#:
#: Zero disables it, and that is the control the measurement needs.
RECON_FRACTION = float(os.environ.get("OTTO_RECON_FRACTION", "0.2"))

#: Said once, when the reconnaissance stretch ends. Not a prohibition -- the
#: agent can still look at things afterwards -- but the point at which looking
#: stops being the job.
RECON_NOTE = (
    "You have looked around enough. From here, work from what you have "
    "found rather than gathering more: make the change, run the thing, and "
    "check it. Look something up again only when a specific question blocks "
    "you, not to be thorough."
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
    #: Ceilings for the CURRENT turn of a multi-turn task, set by
    #: `begin_turn`. None on an ordinary single-turn run, which is every
    #: `otto chat` turn and most benchmark tasks.
    turn_max_calls: int | None = field(default=None)
    turn_hard_at: float | None = field(default=None)
    #: Where the current turn began, so its wrap-up point is a fraction of
    #: THIS turn's share rather than of the whole run -- otherwise a later
    #: turn would be told to wrap up the moment it started.
    turn_start_calls: int = field(default=0)
    turn_wrap_at: float | None = field(default=None)
    #: So the end-of-reconnaissance note is said once, like the wrap-up one.
    recon_warned: bool = field(default=False)

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

    def begin_turn(self, turns_left: int) -> None:
        """Ration what is left across this turn and the ones still to come.

        NOTHING CALLS THIS. It was written for issue #12, measured, and found
        to make the task it was written for WORSE. It stays as the mechanism a
        working version would build on, and as the record of why the obvious
        version does not work.

        The premise was sound: a multi-turn task is many runs sharing one
        budget, and nothing stopped the first run taking all of it. C03 and
        C04 each reached ~930 seconds and ~46 model calls and then stopped
        answering, with the graders saying so outright.

        The measurement, on C04, at the budget the baseline had:

            no rationing        46 calls   23 actions   933s   score 0.43
            rationed, 480s      12 calls    7 actions   353s   score 0.32
            rationed + floors   14 calls    4 actions   520s   score 0.32
            rationed, 933s      16 calls    9 actions   635s   score 0.32

        Worse on every axis, and the grader's reason names the mechanism:
        "the transcript only contains the user's messages". A turn whose share
        runs out returns NO answer, so rationing converted one mediocre answer
        into nine empty turns -- and left 300 seconds of the budget unspent
        while doing it.

        What a working version needs first is for a turn to finish by
        ANSWERING when its share ends, the way the run-level budget already
        does through `wrap_up_once`. Rationing on top of a turn that can end
        with nothing is rationing into a hole.

        An equal share rather than anything cleverer. The alternative is
        guessing which turn deserves more, and a wrong guess starves exactly
        the turn that mattered -- where an equal share at least leaves every
        turn able to answer.

        The ceilings are absolute rather than relative so `phase()` stays a
        comparison: a turn ends when `calls` reaches the number this set.
        Calling it again for the next turn recomputes from what is actually
        left, so a turn that finished early hands its unused share forward.
        """
        # Each turn gets its own wrap-up note. Said once per RUN, a later turn
        # would never be told to finish.
        self.warned = False
        self.turn_start_calls = self.calls
        turns_left = self._affordable_turns(max(1, int(turns_left)))
        if self.max_model_calls is not None:
            remaining = max(self.max_model_calls - self.calls, 0)
            self.turn_max_calls = self.calls + max(
                MIN_TURN_CALLS, remaining // turns_left,
            )
        if self.hard_at is not None:
            now = time.monotonic()
            share = max(
                MIN_TURN_SECONDS, max(self.hard_at - now, 0.0) / turns_left,
            )
            self.turn_hard_at = now + share
            self.turn_wrap_at = now + share * WRAP_UP_FRACTION

    def _affordable_turns(self, turns_left: int) -> int:
        """How many of the remaining turns this budget can actually pay for.

        Dividing by every turn still to come is right only while the shares
        stay usable. Past that it is worse than not rationing: nine turns of a
        480-second budget is 53 seconds each, which buys a criteria call and
        part of a judgment and no answer at all -- measured, and scored below
        the unrationed run it was meant to beat.

        So the count is capped by what the floors can be paid out of. Fewer
        turns answered properly beats every turn answered with nothing, and
        the turns past the cap are not abandoned -- they run on whatever is
        genuinely left, which is the honest version of "there was not enough
        budget for this conversation".
        """
        affordable = turns_left
        if self.hard_at is not None:
            remaining = max(self.hard_at - time.monotonic(), 0.0)
            affordable = min(affordable, max(1, int(remaining // MIN_TURN_SECONDS)))
        if self.max_model_calls is not None:
            remaining_calls = max(self.max_model_calls - self.calls, 0)
            affordable = min(affordable, max(1, remaining_calls // MIN_TURN_CALLS))
        return max(1, affordable)

    def spend(self) -> None:
        """Record one model request. Called from `_call`, once per HTTP
        request rather than once per logical call, so a retry storm is
        visible to the ceiling that is meant to stop it."""
        self.calls += 1

    def recon_once(self) -> str | None:
        """The note to append when the reconnaissance stretch ends, once.

        Paired with `wrap_up_once`: one marks the end of looking, the other
        the end of working. Both are said a single time, because a reminder
        repeated every iteration is one the model stops reading.
        """
        if self.recon_warned or self.in_recon():
            return None
        if self.max_model_calls is None and self.hard_at is None:
            return None  # no budget, so no stretches to be past
        self.recon_warned = True
        return RECON_NOTE

    def phase(self) -> Phase:
        """Where this run is: working, wrapping up, or done.

        The turn ceilings are checked alongside the run's own, and "spent" on
        a turn ceiling is not the end of the task -- the harness starts the
        next turn, which calls `begin_turn` and raises them again.
        """
        now = time.monotonic()
        if self.max_model_calls is not None and self.calls >= self.max_model_calls:
            return "spent"
        if self.hard_at is not None and now >= self.hard_at:
            return "spent"
        if self.turn_max_calls is not None and self.calls >= self.turn_max_calls:
            return "spent"
        if self.turn_hard_at is not None and now >= self.turn_hard_at:
            return "spent"

        if self.turn_max_calls is not None:
            share = self.turn_max_calls - self.turn_start_calls
            if self.calls >= self.turn_start_calls + share * WRAP_UP_FRACTION:
                return "wrap_up"
        if self.max_model_calls is not None:
            if self.calls >= self.max_model_calls * WRAP_UP_FRACTION:
                return "wrap_up"
        if self.turn_wrap_at is not None and now >= self.turn_wrap_at:
            return "wrap_up"
        if self.wrap_up_at is not None and now >= self.wrap_up_at:
            return "wrap_up"
        return "ok"

    def in_recon(self) -> bool:
        """Whether this run is still in its opening, looking-around stretch.

        Its own predicate rather than a value from `phase()`. Reconnaissance
        is about the START of a run and wrapping up is about its END -- the
        same axis, but callers ask different questions of it, and adding a
        fourth value to a Literal that `spent()` and `wrap_up_once()` already
        switch on would have changed what a fresh budget reports to every
        existing reader. It did, and two tests said so.
        """
        if RECON_FRACTION <= 0:
            return False
        if self.max_model_calls is not None:
            return self.calls < self.max_model_calls * RECON_FRACTION
        if self.hard_at is not None and self.wrap_up_at is not None:
            # No call ceiling, so measure against the clock: the recon stretch
            # is the same fraction of the run's total span.
            span = (self.hard_at - self.wrap_up_at) / (1 - WRAP_UP_FRACTION)
            return time.monotonic() < self.hard_at - span * (1 - RECON_FRACTION)
        return False

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
