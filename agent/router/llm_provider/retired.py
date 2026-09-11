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

#: Model families a vendor serves happily but which cannot do the job this
#: router hands out. Separate from RETIRED because the failure is different in
#: kind: these are alive, they simply are not chat models.
#:
#: This matters because capability detection is generous. Gemini's provider
#: tags every `gemini-*` model VISION-capable, so a capability query for "a
#: model that can see" picked `gemini-2.5-flash-preview-tts` -- a
#: text-to-speech model -- the moment the pinned vision model was withdrawn.
#: A pin hides that; a capability fallback does not, which is exactly what a
#: fallback is for.
#:
#: Patterns rather than ids, because these families grow continuously and a
#: list of exact names would be stale within a release.
UNUSABLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "gemini": (
        "-tts",            # text-to-speech
        "-image",          # image generation, not image understanding
        "imagen",
        "veo",             # video generation
        "embedding",
        "-aqa",            # attributed question answering, a different API
        "transcribe",      # speech-to-text
        "-live-",          # bidirectional streaming, a different API surface
        "native-audio",
        "computer-use",    # tool surface of its own, not a chat model
        "omni",            # 400 "This model only supports Interactions API"
        "learnlm",
        "gemma",           # open weights, served without the chat surface
        "robotics",
    ),
    "openai": (
        "tts", "whisper", "dall-e", "embedding", "moderation",
        "-audio", "-realtime", "-transcribe", "-search-preview", "-image",
        "babbage", "davinci", "codex",
    ),
}


def is_unusable(provider: str, model_id: str) -> bool:
    """True if `model_id` is alive but cannot serve a chat request."""
    lowered = model_id.lower()
    return any(p in lowered for p in UNUSABLE_PATTERNS.get(provider, ()))


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
    # Not a retirement but permanently unusable here all the same: Gemini
    # answers a generateContent call to some ids with "This model only
    # supports Interactions API". Capability tags do not distinguish those, so
    # a capability fallback can pick one -- and picking it twice in one run is
    # pure waste.
    "only supports",
)


def is_retired(provider: str, model_id: str) -> bool:
    return model_id in RETIRED.get(provider, {}) or model_id in _LEARNED.get(provider, set())


def is_serviceable(provider: str, model_id: str) -> bool:
    """True if this model is worth offering to the router at all -- neither
    withdrawn by the vendor nor the wrong kind of model for a chat call."""
    return not is_retired(provider, model_id) and not is_unusable(provider, model_id)


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
