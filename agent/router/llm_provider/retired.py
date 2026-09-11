"""Models a vendor still advertises but will no longer serve.

A provider's catalogue is not a promise. Gemini's model list returns
`gemini-2.5-flash` and `gemini-2.5-flash-lite` today, and both answer
generateContent with `404 NOT_FOUND ... no longer available`. That combination
is the worst possible shape for this router: `Router._select` can only check
that a pinned id EXISTS in the catalogue, so a retired pin resolves cleanly,
reports a healthy decision, and then fails at call time -- in the middle of a
run, on whichever node happened to need it.

So retired ids are filtered out of every catalogue before routing ever sees
them. A route pinned to one then fails at IMPORT-adjacent time with "no model
matched", which is a bad error but an early and honest one, rather than a
mid-run surprise.

Two halves, deliberately:

  * RETIRED below is the curated list. Each entry carries why and when, so the
    next person can tell a real retirement from someone's bad afternoon.
  * `note_retired()` is the runtime half. When a call fails with a vendor's
    "this model is gone" error, the caller records it here, which drops the id
    for the rest of the process and logs loudly enough that it gets added to
    the curated list. Deliberately NOT persisted: a process-local set cannot
    silently accumulate a wrong entry that nobody can find, and a vendor
    outage misread as a retirement would otherwise be sticky forever.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: provider -> {model id: why it was removed}. Curated, and every entry should
#: say enough that it can be re-tested and removed when a vendor brings
#: something back.
RETIRED: dict[str, dict[str, str]] = {
    "gemini": {
        "gemini-2.5-flash": (
            "2026-09-11: still listed by the models endpoint, but "
            "generateContent returns 404 'no longer available'"
        ),
        "gemini-2.5-flash-lite": (
            "2026-09-11: same as gemini-2.5-flash -- listed, not served"
        ),
    },
}

#: Learned during this process by note_retired(). Never persisted; see the
#: module docstring for why that is on purpose.
_LEARNED: dict[str, set[str]] = {}

#: Substrings that mean "the vendor will not serve this id again", as opposed
#: to "the vendor is having a bad minute". Kept narrow on purpose: treating a
#: transient 404 as a retirement would quietly shrink the catalogue.
RETIREMENT_MARKERS = (
    "no longer available",
    "has been deprecated",
    "is deprecated",
    "model not found",
    "does not exist",
)


def is_retired(provider: str, model_id: str) -> bool:
    return model_id in RETIRED.get(provider, {}) or model_id in _LEARNED.get(provider, set())


def note_retired(provider: str, model_id: str, reason: str) -> None:
    """Record that `provider:model_id` will not serve requests.

    Called from the failure path when a vendor says a model is gone. Takes
    effect immediately for this process and logs at warning, because the real
    fix is a line in RETIRED above -- this only stops the same run from
    retrying a model that has already told it no.
    """
    if is_retired(provider, model_id):
        return
    _LEARNED.setdefault(provider, set()).add(model_id)
    logger.warning(
        "%s:%s looks retired (%s) -- dropped for this process. Add it to "
        "agent/router/llm_provider/retired.py to make that permanent.",
        provider, model_id, reason,
    )


def looks_retired(exc: Exception) -> bool:
    """True if `exc` reads like a vendor refusing a model permanently."""
    text = str(exc).lower()
    return any(marker in text for marker in RETIREMENT_MARKERS)
