"""Per-turn record of what each MODEL was asked to do, counted in tokens.

`agent/pipeline/budget.py` already counts model REQUESTS, because that is the
number a ceiling can be enforced against cheaply and the same number whatever
answered. It is not the number anybody reads to understand a turn: a request to
a small diffusion model and a request carrying a 60k-token transcript to a
frontier model cost wildly different amounts, and `model_calls: 14` says
nothing about which happened.

The data was already arriving and being thrown away. Every provider here
attaches langchain's standard `usage_metadata` to the message it streams back
-- agent/router/llm_provider/inception_provider.py goes as far as setting
`stream_options: {include_usage: true}` so the API sends the usage chunk at all
-- and `_call` accumulated the chunks, read the text off them, and dropped the
rest.

SAME CONTEXTVAR SHAPE as budget.py, workspace.py, toolkit.py and execution.py,
for the same reason: `_call` is reached through a graph, several nodes deep,
and cannot see the run it belongs to. Nothing bound is a valid state and makes
every record() a no-op, which is what keeps a direct `_call` in a unit test
free of ceremony.

PRICING lives next door in agent/pipeline/pricing.py, and the split is the
point: tokens are MEASURED and rates are LOOKED UP on a date. A ledger entry is
a fact about what happened; the money on top of it is an estimate against a
table that goes stale. `cost` is None -- never 0.0 -- for a model nothing has a
rate for, so the two never look alike.

A model that reports no usage is recorded with its calls and zero tokens,
and `ModelUsage.reported` is False so a reader can tell "this model is free"
apart from "this model does not say". Silently showing 0 for both is the one
failure this shape exists to avoid.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping

from agent.pipeline.pricing import cost_of


def _int(value: Any) -> int:
    """A token count from a provider payload, or 0. Four vendors' adapters
    feed this and a missing or non-numeric field must never be the reason a
    turn fails."""
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class ModelUsage:
    """One model's share of a turn."""

    model: str
    calls: int = 0
    #: The provider's TOTAL input, which by langchain's convention INCLUDES
    #: the two cache figures below rather than sitting alongside them.
    input_tokens: int = 0
    output_tokens: int = 0
    #: Tokens served from a prompt cache, and tokens written into one. Tracked
    #: separately because every vendor that offers caching prices them
    #: differently from plain input -- reads at a steep discount, writes at a
    #: premium -- and this agent re-sends a growing transcript on every call,
    #: so a long turn is mostly cache reads. Pricing them as plain input would
    #: overstate a real turn substantially.
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    #: Whether this model ever reported token counts. False means the numbers
    #: above are absent rather than zero -- see the module docstring.
    reported: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float | None:
        """USD for this model's share, or None if nothing has a rate for it.

        Computed on read rather than stored: a rate can come from a file that
        is read after some of these calls were already recorded, and a cost
        frozen at record time would be a mix of two answers.
        """
        if not self.reported:
            return None
        return cost_of(
            self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )


@dataclass
class UsageLedger:
    """What one turn spent, per model, in the order the models first answered.

    Mutable and shared, like Budget: `record()` is called from `_call` with no
    way to hand a value back up the graph. Insertion-ordered because the order
    models first appear is the order the turn actually escalated through them,
    which is the reading a person wants and is otherwise lost.
    """

    by_model: dict[str, ModelUsage] = field(default_factory=dict)

    def record(self, model: str, usage: Mapping[str, Any] | None) -> None:
        """Add one model REQUEST, and its tokens if the provider sent any.

        Called once per HTTP request rather than once per logical call, the
        same as Budget.spend() and for the same reason: a retry storm is real
        spend and has to be visible as such.
        """
        name = str(model or "unknown")
        entry = self.by_model.get(name)
        if entry is None:
            entry = self.by_model[name] = ModelUsage(model=name)
        entry.calls += 1
        if not usage:
            return
        # Tolerant reads. `usage_metadata` is langchain's standard shape, but
        # this runs against four vendors' adapters and a missing or non-numeric
        # field must never be the reason a turn fails.
        got = False
        for key, attr in (("input_tokens", "input_tokens"),
                          ("output_tokens", "output_tokens")):
            value = _int(usage.get(key))
            if value:
                setattr(entry, attr, getattr(entry, attr) + value)
                got = True

        # Cache figures live one level down, and only on the vendors that
        # offer caching -- absent everywhere else, which is why this reads
        # rather than requires them.
        details = usage.get("input_token_details")
        if isinstance(details, Mapping):
            entry.cached_input_tokens += _int(details.get("cache_read"))
            entry.cache_write_tokens += _int(details.get("cache_creation"))
        entry.reported = entry.reported or got

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.by_model.values())

    @property
    def input_tokens(self) -> int:
        return sum(m.input_tokens for m in self.by_model.values())

    @property
    def output_tokens(self) -> int:
        return sum(m.output_tokens for m in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cached_input_tokens(self) -> int:
        return sum(m.cached_input_tokens for m in self.by_model.values())

    @property
    def cost(self) -> float:
        """USD across every model that HAS a rate. Read it with
        `fully_priced` -- on its own it is a floor, not a total."""
        return sum(m.cost or 0.0 for m in self.by_model.values())

    @property
    def fully_priced(self) -> bool:
        """Whether every model that reported usage also had a rate. False
        means `cost` is missing somebody's share, and anything showing it has
        to say so rather than presenting a short total as the total."""
        return all(m.cost is not None for m in self.by_model.values() if m.reported)

    def models(self) -> list[ModelUsage]:
        """Every model that answered, in the order it first did."""
        return list(self.by_model.values())

    def snapshot(self) -> dict:
        """A plain dict, for handing across the thread boundary into a UI.

        Plain rather than the dataclasses themselves: the TUI reads this from
        its own worker thread while `_call` may still be writing to the ledger,
        and a snapshot cannot be half-updated under a reader the way a live
        object can.
        """
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "fully_priced": self.fully_priced,
            "models": [
                {
                    "model": m.model,
                    "calls": m.calls,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "cached_input_tokens": m.cached_input_tokens,
                    "total_tokens": m.total_tokens,
                    "reported": m.reported,
                    "cost": m.cost,
                }
                for m in self.models()
            ],
        }


_current: contextvars.ContextVar[UsageLedger | None] = contextvars.ContextVar(
    "otto_current_usage", default=None,
)


@contextmanager
def bind_usage(ledger: UsageLedger | None) -> Iterator[UsageLedger | None]:
    """Make `ledger` the one this block and everything it calls records into.

    None unbinds. Unlike Budget -- where a resumed run must bind a fresh one,
    because the time a person spent answering is not the agent's -- the ledger
    is deliberately the CALLER's to own and reuse: the TUI keeps one for the
    whole session and hands the same one to every turn and every resume, so
    what it shows is cumulative without anything having to add snapshots up.
    """
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


def current_usage() -> UsageLedger | None:
    """The ledger bound by the innermost `bind_usage()`, or None."""
    return _current.get()


def record_usage(model: str, usage: Mapping[str, Any] | None) -> None:
    """Record one request against the bound ledger, if there is one."""
    ledger = _current.get()
    if ledger is not None:
        ledger.record(model, usage)
