"""Custom OpenAI-compatible endpoints: OpenRouter, a remote vLLM, Ollama, LM
Studio -- anything that answers `GET /v1/models` and `POST /v1/chat/completions`.

One factory, `openai_compatible(name)`, turns a NAME into a provider class
that reads `<NAME>_API_KEY` and `<NAME>_BASE_URL` from the environment and
otherwise behaves exactly like `OpenAIProvider` -- because it IS one, a
subclass built with `type()`. The names live in ~/.otto/routes.json
(agent/router/overrides.py registers each at startup); the key and URL rows
live in the repo `.env` beside the built-in vendors' keys.

Capability classification is a heuristic and says so, the same way
`openai_provider._classify` does: a generic `/v1/models` returns ids and
nothing else. The floor is CHAT+TOOLS; REASONING and VISION are guessed from
well-known family names; a person can pin the truth per model in routes.json
(`endpoints.<name>.capabilities`), which wins over the guess.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from agent.router.llm_provider.base import (
    AuthError,
    BaseProvider,
    Capability,
    HealthReport,
    ProviderStatus,
)

if TYPE_CHECKING:  # pragma: no cover
    from agent.router.llm_provider.openai_provider import OpenAIProvider

__all__ = [
    "NAME_RE", "BUILTIN", "validate_name", "env_prefix", "key_var", "url_var",
    "classify_generic", "openai_compatible",
]

#: Lower-case, starts with a letter, up to 32 characters. The name becomes an
#: environment-variable prefix (upper-cased, hyphens to underscores) and the
#: provider half of a `provider:model` spec, so it has to be safe in both.
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
BUILTIN = ("inception", "anthropic", "openai", "gemini")
_RESERVED = BUILTIN + ("otto", "custom", "any", "none")


def validate_name(name: str) -> None:
    if not isinstance(name, str) or not NAME_RE.match(name):
        raise ValueError(
            f"{name!r} is not a usable endpoint name: lower-case letters, digits, "
            f"'-' or '_', starting with a letter, at most 32 characters"
        )
    if name in _RESERVED:
        raise ValueError(f"{name!r} is reserved")


def env_prefix(name: str) -> str:
    return name.upper().replace("-", "_")


def key_var(name: str) -> str:
    return f"{env_prefix(name)}_API_KEY"


def url_var(name: str) -> str:
    return f"{env_prefix(name)}_BASE_URL"


_NON_CHAT = ("whisper", "tts", "dall-e", "moderation", "sora", "davinci", "babbage")
_REASONING = re.compile(r"^(o\d|gpt-5|deepseek-r|qwq|.*-r1|.*think|.*reason)")
_VISION_MARKERS = ("vision", "-vl", "llava", "gpt-4o", "gpt-4.1", "gpt-5",
                   "claude", "gemini", "pixtral", "gemma-3")


def classify_generic(model_id: str) -> frozenset[Capability]:
    """Best-effort capabilities for an id from an unknown OpenAI-compatible
    server. Heuristic -- see the module docstring."""
    lowered = (model_id or "").lower()
    vendor, _, tail = lowered.rpartition("/")     # OpenRouter's "vendor/model"
    if "embed" in tail:
        return frozenset({Capability.EMBEDDINGS})
    if any(marker in tail for marker in _NON_CHAT):
        return frozenset()
    caps = {Capability.CHAT, Capability.TOOLS}
    if _REASONING.match(tail) or vendor in ("anthropic", "google"):
        caps.add(Capability.REASONING)
    if any(marker in tail for marker in _VISION_MARKERS):
        caps.add(Capability.VISION)
    return frozenset(caps)


def openai_compatible(
    name: str,
    *,
    capability_overrides: Mapping[str, Iterable[str]] | None = None,
) -> "type[OpenAIProvider]":
    """A provider class for the endpoint called `name`.

    The `openai` SDK import is inside the function on purpose: this module is
    imported at startup by agent/router/overrides.py, and a routes.json that
    names an endpoint must not make a machine without the SDK fail to start.
    """
    validate_name(name)
    from agent.router.llm_provider.openai_provider import OpenAIProvider

    overrides: dict[str, frozenset[Capability]] = {
        model_id: frozenset(Capability(c) for c in caps)
        for model_id, caps in (capability_overrides or {}).items()
    }
    key_name, url_name = key_var(name), url_var(name)

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None) -> None:
        url = base_url or os.environ.get(url_name)
        if not url:
            # Without this an endpoint with a key but no URL would quietly
            # send that key to api.openai.com.
            raise AuthError(f"{name}: {url_name} is not set")
        OpenAIProvider.__init__(self, api_key, base_url=url.rstrip("/"))

    @classmethod
    def is_configured(cls) -> bool:
        # Servers that ignore keys (Ollama, a local vLLM) still need SOME
        # value here -- the setup screen says so; "x" is fine.
        return bool(os.environ.get(key_name)) and bool(os.environ.get(url_name))

    @classmethod
    def check(cls) -> HealthReport:
        missing = [v for v in (key_name, url_name) if not os.environ.get(v)]
        if missing:
            return HealthReport(name, ProviderStatus.NO_KEY,
                                detail=f"{' and '.join(missing)} not set")
        return BaseProvider.check.__func__(cls)

    def _classify(self, model_id: str) -> frozenset[Capability]:
        return overrides.get(model_id) or classify_generic(model_id)

    namespace = {
        "__module__": __name__,
        "__doc__": f"OpenAI-compatible endpoint {name!r} ({url_name}).",
        "name": name,
        "env_var": key_name,
        "base_url_var": url_name,
        "default_base_url": None,
        "capabilities": frozenset({Capability.CHAT, Capability.TOOLS,
                                   Capability.VISION, Capability.REASONING}),
        "__init__": __init__,
        "is_configured": is_configured,
        "check": check,
        "_classify": _classify,
    }
    return type(f"CustomProvider_{env_prefix(name)}", (OpenAIProvider,), namespace)
