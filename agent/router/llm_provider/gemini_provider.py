"""Gemini provider, on the `google-genai` SDK.

Note this is `google-genai` (``from google import genai``), not the legacy
`google-generativeai` package -- the two have incompatible APIs and the newer
one is what `langchain-google-genai` pulls in here.

As with Anthropic: SDK for discovery, `ChatGoogleGenerativeAI` for chat.
"""

from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import (
    AuthError,
    BaseProvider,
    Capability,
    ModelInfo,
    ProviderError,
    ProviderUnavailable,
)

__all__ = ["GeminiProvider"]


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        if code in (401, 403):
            return AuthError(f"gemini: HTTP {code}")
        return ProviderError(f"gemini: HTTP {code}: {exc}")
    if isinstance(exc, httpx.TransportError):
        # The SDK only wraps HTTP-level failures; DNS, TLS and timeout errors
        # arrive as raw httpx exceptions and would otherwise reach health_check
        # untranslated.
        return ProviderUnavailable(f"gemini: {exc}")
    return ProviderError(f"gemini: {exc}")


#: `Model.supported_actions` values that matter to us. This is the only
#: capability metadata Gemini publishes.
_ACTION_MAP: dict[str, Capability] = {
    "generateContent": Capability.CHAT,
    "embedContent": Capability.EMBEDDINGS,
}


def _capabilities_of(model: Any, model_id: str) -> set[Capability]:
    caps = {
        _ACTION_MAP[action]
        for action in (model.supported_actions or [])
        if action in _ACTION_MAP
    }

    # Gemini publishes no tool-use or modality flags, so these two are a
    # heuristic -- flagged as such, exactly like the OpenAI classifier. Every
    # current gemini-* model does function calling and image input; the
    # gemma-*, embedding-* and aqa entries in the same list do not.
    if Capability.CHAT in caps and model_id.startswith("gemini-"):
        caps |= {Capability.TOOLS, Capability.VISION, Capability.STRUCTURED_OUTPUT}

    return caps


class GeminiProvider(BaseProvider):
    name = "gemini"
    env_var = "GEMINI_API_KEY"
    default_base_url = None
    capabilities = frozenset(
        {
            Capability.CHAT,
            Capability.TOOLS,
            Capability.VISION,
            Capability.STRUCTURED_OUTPUT,
            Capability.EMBEDDINGS,
        }
    )

    def _build_client(self) -> genai.Client:
        # Explicit key: the SDK would otherwise fall back to GOOGLE_API_KEY or
        # GEMINI_API_KEY on its own, which makes it ambiguous which one won.
        return genai.Client(api_key=self._api_key)

    def _fetch_models(self) -> list[ModelInfo]:
        try:
            # The Pager auto-fetches subsequent pages while iterating; page_size
            # only controls how many round trips that takes.
            models = list(self._client.models.list(config={"page_size": 100}))
        except Exception as exc:
            raise _translate(exc) from exc

        found: list[ModelInfo] = []
        for model in models:
            # `name` arrives as "models/gemini-2.5-pro"; the bare id is what
            # every other API surface expects.
            model_id = (model.name or "").removeprefix("models/")
            if not model_id:
                continue
            found.append(
                ModelInfo(
                    id=model_id,
                    provider=self.name,
                    display_name=model.display_name,
                    capabilities=frozenset(_capabilities_of(model, model_id)),
                    context_window=model.input_token_limit,
                    max_output_tokens=model.output_token_limit,
                    raw=model.model_dump(mode="json"),
                )
            )
        return found

    def chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        return ChatGoogleGenerativeAI(
            model=model_id, google_api_key=self._api_key, **kwargs
        )
