"""What a model charges, so a token count can be shown as money.

THE RATES BELOW GO STALE. They are list prices read off vendor pricing pages
on the date in `PRICES_AS_OF`, they are not fetched from anywhere, and nothing
in this repo re-checks them. A vendor can change a price, or a negotiated or
provisioned rate can differ from list, and this file will keep confidently
reporting the old number. Treat what it produces as an ESTIMATE with a date on
it, which is why every place that shows a cost says so.

That is also why `OTTO_MODEL_PRICES` exists. It points at a JSON file of
`{"<model id>": {"input": 1.0, "output": 5.0, "cached_input": 0.1}}`, in USD
per MILLION tokens, and it wins over the table below. Correcting a rate, or
pricing a model this file has never heard of, is a config change rather than a
code change -- the one thing that has to be easy about data with an expiry
date on it.

A model with NO rate is not priced at zero. It reports `None`, the panel shows
"--", and any total containing one is marked partial. The same rule as
`usage.py`'s `reported` flag and for the same reason: "not known" and "free"
are different facts, and the cheapest way to make a cost display worthless is
to let them look the same.

WHAT IS MODELLED. Input and output are charged at different rates everywhere,
so they are separate. Cache READS are charged at a large discount by every
vendor that offers them, and a run of this agent re-sends a growing transcript
on every call, so ignoring the discount would overstate a long turn's cost
substantially -- `cached_input` is therefore its own rate, applied to the
`cache_read` tokens the provider reports. Cache WRITES cost more than plain
input; `cache_write` covers that where a rate is known, and falls back to the
input rate where it is not, which understates slightly rather than guessing.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: When the table below was read off the vendors' pricing pages. Shown in the
#: UI next to any figure derived from it, because a cost with no date on it
#: invites more trust than this file can earn.
PRICES_AS_OF = "2026-09-12"

#: USD per MILLION tokens. Keys are matched against a model id by
#: `rate_for()`: exact first, then longest key that the id starts with, so one
#: entry covers every dated revision of a model ("claude-haiku-4-5" matches
#: "claude-haiku-4-5-20251001") without this table having to track stamps.
#:
#: DELIBERATELY INCOMPLETE. Inception's Mercury models are not here: this file
#: is only worth having if every number in it was actually looked up, and a
#: plausible-looking guess is worse than the "--" an absent entry produces.
#: Add them (or correct anything here) through OTTO_MODEL_PRICES rather than by
#: editing this, so the change survives a pull.


@dataclass(frozen=True)
class Rate:
    """One model's prices, in USD per million tokens."""

    input: float
    output: float
    #: Per million cache-READ tokens. None means "this model has no cache
    #: discount, or none that is known" -- cache reads then cost input rate.
    cached_input: float | None = None
    #: Per million cache-WRITE tokens. None falls back to the input rate,
    #: which understates rather than guesses.
    cache_write: float | None = None

    def cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        """USD for these tokens.

        `input_tokens` is the provider's TOTAL input, which by langchain's
        convention already INCLUDES the cached and cache-written ones -- so
        they are subtracted out here before the full rate is applied, and
        charged at their own. Getting that inclusion backwards double-counts
        the largest number on the page.
        """
        cached = max(int(cached_input_tokens or 0), 0)
        written = max(int(cache_write_tokens or 0), 0)
        plain = max(int(input_tokens or 0) - cached - written, 0)
        cached_rate = self.input if self.cached_input is None else self.cached_input
        write_rate = self.input if self.cache_write is None else self.cache_write
        return (
            plain * self.input
            + cached * cached_rate
            + written * write_rate
            + max(int(output_tokens or 0), 0) * self.output
        ) / 1_000_000


#: List prices as of PRICES_AS_OF. Every entry here corresponds to a `spec` in
#: agent/router/mapping.py -- there is no point carrying rates for models this
#: agent cannot route to.
PRICES: dict[str, Rate] = {
    # Anthropic. Cache reads are a tenth of input; cache writes (5-minute TTL)
    # are input x1.25.
    "claude-haiku-4-5": Rate(input=1.00, output=5.00, cached_input=0.10, cache_write=1.25),
    # OpenAI. Cached input is a tenth of input.
    "gpt-5-mini": Rate(input=0.25, output=2.00, cached_input=0.025),
    # Google. The Flash Lite tier; "latest" tracks whatever the current one is,
    # so this is the rate for that tier rather than for a pinned revision.
    "gemini-flash-lite": Rate(input=0.10, output=0.40, cached_input=0.025),
    "gemini-3-flash": Rate(input=0.30, output=2.50, cached_input=0.075),
    # inception:mercury-* deliberately absent -- see the note above PRICES_AS_OF.
}

#: Path to a JSON file of overrides, read once on first use.
PRICES_ENV = "OTTO_MODEL_PRICES"

_overrides: dict[str, Rate] | None = None


def _load_overrides() -> dict[str, Rate]:
    """Read OTTO_MODEL_PRICES, once. A bad file is logged and ignored: a
    typo in a price file must not be the reason a turn cannot run."""
    global _overrides
    if _overrides is not None:
        return _overrides
    _overrides = {}
    path = os.environ.get(PRICES_ENV, "").strip()
    if not path:
        return _overrides
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        for name, fields in raw.items():
            _overrides[str(name).lower()] = Rate(
                input=float(fields["input"]),
                output=float(fields["output"]),
                cached_input=(None if fields.get("cached_input") is None
                              else float(fields["cached_input"])),
                cache_write=(None if fields.get("cache_write") is None
                             else float(fields["cache_write"])),
            )
    except Exception as exc:  # noqa: BLE001 -- never fail a run over a price file
        logger.warning("%s=%r could not be read (%s); using built-in rates",
                       PRICES_ENV, path, exc)
        _overrides = {}
    return _overrides


def reset_overrides() -> None:
    """Forget the cached override file so the next lookup re-reads it. For
    tests, which change the environment between cases."""
    global _overrides
    _overrides = None


def rate_for(model: str) -> Rate | None:
    """This model's rate, or None if nothing here knows it.

    Matching, in order: the id exactly; the id with a vendor or region prefix
    stripped; then the LONGEST table key the id starts with, which is what
    lets one entry cover every dated revision of a model. Overrides are
    consulted the same way, first.
    """
    name = str(model or "").strip().lower()
    if not name:
        return None
    # "openai/gpt-5-mini" and "us.anthropic.claude-haiku-4-5-20251001" both
    # reduce to the id the tables are keyed on.
    candidates = [name, name.split("/")[-1]]
    candidates.append(candidates[-1].split(":")[-1])
    tail = candidates[-1]
    if "." in tail:
        candidates.append(tail.split(".")[-1])

    for table in (_load_overrides(), PRICES):
        for candidate in candidates:
            if candidate in table:
                return table[candidate]
        # Longest prefix wins, so "claude-haiku-4-5-20251001" prefers a
        # "claude-haiku-4-5" entry over a hypothetical "claude" one.
        for key in sorted(table, key=len, reverse=True):
            if any(candidate.startswith(key) for candidate in candidates):
                return table[key]
    return None


def cost_of(model: str, *, input_tokens: int, output_tokens: int,
            cached_input_tokens: int = 0, cache_write_tokens: int = 0) -> float | None:
    """USD for what this model was asked to do, or None if it has no rate."""
    rate = rate_for(model)
    if rate is None:
        return None
    return rate.cost(input_tokens, output_tokens, cached_input_tokens, cache_write_tokens)


def format_cost(amount: float | None) -> str:
    """A dollar figure at a precision that says something.

    Sub-cent turns are the normal case for the cheap tiers here, so rounding
    everything to cents would show "$0.00" for most of a session and then jump.
    """
    if amount is None:
        return "--"
    if amount and amount < 0.01:
        return f"${amount:.4f}"
    if amount < 10:
        return f"${amount:.3f}"
    return f"${amount:.2f}"
