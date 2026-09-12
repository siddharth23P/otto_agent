"""Which models and providers are temporarily unwell, so the chain stops
re-trying one that just failed.

agent/router/llm_provider/retired.py already handles the PERMANENT case: a
model the vendor will never serve again is dropped from the catalogue for the
life of the process. This is the other half, and it is the common one -- a
rate limit, a five-hundred, a connection that timed out. None of those mean
"gone", they mean "not now", and until this existed Otto's answer to all of
them was to resolve the same candidate again on the very next call.

TWO LAYERS, BECAUSE THE FAILURES ARE DIFFERENT IN KIND.

  A rate limit is about ONE MODEL. Quota is metered per model, so a 429 on the
  cheap model says nothing about the expensive one from the same vendor, and
  locking the vendor over it would take out the seats that were fine.

  A five-hundred or a dead connection is about the PROVIDER. The next model
  from the same vendor goes down the same wire. One of those is a blip and
  gets retried; a run of them is an outage, and re-trying every candidate of
  that vendor on every call for the rest of the run is how a fifteen-minute
  outage costs a whole budget.

NEVER INTO NOTHING. A breaker that leaves a task with no route has turned a
slow provider into a broken agent, which is worse than the problem. Cooldowns
are advisory: `Router.resolve` skips a cooling candidate while another is
available and falls back to the least-cooled one when none is.

HALF-OPEN, THEN LONGER. When a window expires the next call is a probe. If it
works the record clears; if it fails the window doubles, so a vendor that is
genuinely down is asked less and less often rather than on a fixed heartbeat.

PROCESS-LOCAL, NOT PERSISTED -- the same call retired.py makes and for the
same reason: a wrong entry nobody can find is worse than one that disappears
when the process does, and yesterday's outage should not shape today's run.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: First cooldown for one model after a rate limit. Short: quota windows are
#: usually seconds, and a minute of avoidance would cost more than the wait.
MODEL_COOLDOWN_S = 20.0

#: First cooldown for a whole provider once the breaker trips.
PROVIDER_COOLDOWN_S = 45.0

#: Consecutive transport failures before a provider is considered out rather
#: than unlucky. Three, because one is noise and two is a coincidence.
BREAKER_THRESHOLD = 3

#: Ceiling on the doubling. Past this the provider is being asked once every
#: five minutes, which is often enough to notice a recovery and rare enough
#: to cost nothing.
MAX_COOLDOWN_S = 300.0


@dataclass
class _Cooling:
    until: float = 0.0
    #: How many times this has been cooled without a success in between. The
    #: doubling exponent.
    strikes: int = 0
    #: Consecutive failures, for the provider breaker only.
    failures: int = 0
    why: str = ""


def _now() -> float:
    return time.monotonic()


class Health:
    """Everything known about who is currently unwell. One instance per
    process (`HEALTH` below); tests make their own."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], _Cooling] = {}
        self._providers: dict[str, _Cooling] = {}

    # ---- reporting -----------------------------------------------------

    def note_success(self, provider: str, model_id: str) -> None:
        """A call went through. Clears both layers for this pair.

        Clearing the PROVIDER on any success is deliberate: the breaker exists
        to answer "is this vendor reachable", and one answered request settles
        that regardless of which model answered it.
        """
        self._models.pop((provider, model_id), None)
        self._providers.pop(provider, None)

    def note_rate_limit(self, provider: str, model_id: str, *,
                        retry_after: float | None = None) -> None:
        """This model is over its quota. The vendor is fine."""
        if not (provider and model_id):
            return
        record = self._models.setdefault((provider, model_id), _Cooling())
        record.strikes += 1
        wait = retry_after if retry_after and retry_after > 0 else _backoff(
            MODEL_COOLDOWN_S, record.strikes,
        )
        record.until = _now() + min(wait, MAX_COOLDOWN_S)
        record.why = f"rate limited, cooling {min(wait, MAX_COOLDOWN_S):.0f}s"
        logger.info("%s:%s %s", provider, model_id, record.why)

    def note_transport_failure(self, provider: str, detail: str = "") -> None:
        """A five-hundred, a timeout, a refused connection. Counts toward the
        provider's breaker and trips it on the third in a row."""
        if not provider:
            return
        record = self._providers.setdefault(provider, _Cooling())
        record.failures += 1
        if record.failures < BREAKER_THRESHOLD:
            return
        record.strikes += 1
        wait = min(_backoff(PROVIDER_COOLDOWN_S, record.strikes), MAX_COOLDOWN_S)
        record.until = _now() + wait
        record.why = f"{record.failures} failures in a row, cooling {wait:.0f}s"
        logger.warning("provider %s: %s (%s)", provider, record.why, detail[:120])

    # ---- asking --------------------------------------------------------

    def cooling(self, provider: str, model_id: str) -> str:
        """Why this candidate should be passed over, or "" if it should not.

        A window that has expired returns "" -- that IS the half-open probe.
        The record stays, so a probe that fails doubles the window rather than
        starting over.
        """
        now = _now()
        record = self._providers.get(provider)
        if record and record.until > now:
            return f"provider {provider}: {record.why}"
        record = self._models.get((provider, model_id))
        if record and record.until > now:
            return f"{model_id}: {record.why}"
        return ""

    def cooling_until(self, provider: str, model_id: str) -> float:
        """When this candidate is next due, for ordering a chain in which
        everything is cooling. 0 when it is available now."""
        now = _now()
        soonest = 0.0
        for record in (self._providers.get(provider), self._models.get((provider, model_id))):
            if record and record.until > now:
                soonest = max(soonest, record.until)
        return soonest

    def snapshot(self) -> dict:
        """What is cooling right now, for `otto doctor` and for tests."""
        now = _now()
        return {
            "providers": {p: round(r.until - now, 1)
                          for p, r in self._providers.items() if r.until > now},
            "models": {f"{p}:{m}": round(r.until - now, 1)
                       for (p, m), r in self._models.items() if r.until > now},
        }


def _backoff(base: float, strikes: int) -> float:
    return base * (2 ** max(0, strikes - 1))


#: The process-wide instance every caller uses.
HEALTH = Health()


@contextmanager
def bind_health(health: Health | None):
    """Swap the instance for the duration. Tests and measurements only --
    a run that carried a fresh one per call would have no memory and no
    breaker at all."""
    global HEALTH
    previous = HEALTH
    HEALTH = health if health is not None else Health()
    try:
        yield HEALTH
    finally:
        HEALTH = previous


def note_failure(exc: Exception, *, provider: str, model_id: str) -> None:
    """Route one failed call to the right layer.

    Reads the status off the exception the same duck-typed way
    agent/router/llm_provider/base.py's `translate_unknown` does, rather than
    importing four vendor SDKs to ask them politely.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        HEALTH.note_rate_limit(provider, model_id, retry_after=_retry_after(exc))
        return
    if isinstance(status, int) and status >= 500:
        HEALTH.note_transport_failure(provider, f"{type(exc).__name__}: {exc}")
        return
    if status is None and _looks_like_transport(exc):
        HEALTH.note_transport_failure(provider, f"{type(exc).__name__}: {exc}")


_TRANSPORT_MARKERS = ("timeout", "connection", "unavailable", "apierror",
                      "serviceunavailable", "overloaded", "remotedisconnected")


def _looks_like_transport(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    return any(marker in name for marker in _TRANSPORT_MARKERS)


def _retry_after(exc: Exception) -> float | None:
    """The vendor's own answer to "when should I come back", when it gave one.
    Honouring it beats guessing: a 429 with Retry-After: 2 waited out for
    twenty seconds is nineteen seconds of nothing."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset-after"):
        try:
            value = headers.get(key)
        except AttributeError:
            return None
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None
