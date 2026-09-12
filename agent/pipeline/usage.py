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

WHAT IT DOES NOT DO is price anything. Rates differ per model, change without
notice, and are the sort of thing that is wrong in a way nobody notices for
months. Tokens are what was actually measured; a currency figure would be a
guess wearing a decimal point.

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


@dataclass
class ModelUsage:
    """One model's share of a turn."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Whether this model ever reported token counts. False means the numbers
    #: above are absent rather than zero -- see the module docstring.
    reported: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


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
            try:
                value = int(usage.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if value:
                setattr(entry, attr, getattr(entry, attr) + value)
                got = True
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
            "total_tokens": self.total_tokens,
            "models": [
                {
                    "model": m.model,
                    "calls": m.calls,
                    "input_tokens": m.input_tokens,
                    "output_tokens": m.output_tokens,
                    "total_tokens": m.total_tokens,
                    "reported": m.reported,
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
