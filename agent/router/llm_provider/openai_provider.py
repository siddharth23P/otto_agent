"""OpenAI provider -- the reference implementation of `BaseProvider`.

Read this before writing the Anthropic, Gemini and Inception subclasses: the
three abstract methods below are the entire per-vendor surface, and the
error-translation block in `_fetch_models` is the part that is easy to skip
and expensive to skip.

Module is named `openai_provider.py`, not `openai.py`, on purpose -- see the
note at the bottom of the file.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    OpenAIError,
)

from agent.router.llm_provider.base import (
    AuthError,
    BaseProvider,
    Capability,
    ModelInfo,
    ProviderError,
    ProviderUnavailable,
)

__all__ = ["OpenAIProvider"]


# OpenAI's /v1/models returns ids and nothing else -- no capability metadata,
# no context window. So classification is a heuristic over the id, and it is
# marked as such rather than dressed up as fact. The untouched payload stays
# on `ModelInfo.raw` for anything this misses.

_NON_CHAT_MARKERS = (
    "whisper", "tts", "dall-e", "moderation", "sora", "davinci", "babbage",
)


def _classify(model_id: str) -> frozenset[Capability]:
    """Best-effort capability guess from an OpenAI model id."""
    mid = model_id.lower()

    if mid.startswith("text-embedding"):
        return frozenset({Capability.EMBEDDINGS})

    if any(marker in mid for marker in _NON_CHAT_MARKERS):
        return frozenset()

    caps = {Capability.CHAT, Capability.TOOLS}

    # Reasoning families.
    if mid.startswith(("o1", "o3", "o4")) or mid.startswith("gpt-5"):
        caps.add(Capability.REASONING)

    # Multimodal families.
    if mid.startswith(("gpt-4o", "gpt-4.1", "gpt-5")) or "vision" in mid:
        caps.add(Capability.VISION)

    return frozenset(caps)


class OpenAIProvider(BaseProvider):
    name = "openai"
    env_var = "OPENAI_API_KEY"
    default_base_url = None  # Inception overrides this; OpenAI uses the SDK default
    capabilities = frozenset(
        {
            Capability.CHAT,
            Capability.TOOLS,
            Capability.VISION,
            Capability.REASONING,
            Capability.EMBEDDINGS,
        }
    )

    # -- 1. client --------------------------------------------------------

    def _build_client(self) -> OpenAI:
        # Pass the key explicitly. Letting the SDK read the environment itself
        # makes the provider untestable and lets a stale shell export win over
        # whatever the caller actually asked for.
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    # -- 2. model discovery -----------------------------------------------

    def _fetch_models(self) -> list[ModelInfo]:
        try:
            page = self._client.models.list()

        # Order matters: AuthenticationError subclasses APIStatusError, so the
        # narrow case has to be caught first or it is swallowed by the broad one.
        except AuthenticationError as exc:
            raise AuthError(f"{self.name}: key rejected ({exc})") from exc
        except APIConnectionError as exc:
            raise ProviderUnavailable(f"{self.name}: {exc}") from exc
        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                raise AuthError(f"{self.name}: HTTP {exc.status_code}") from exc
            raise ProviderError(f"{self.name}: HTTP {exc.status_code}") from exc
        except OpenAIError as exc:
            raise ProviderError(f"{self.name}: {exc}") from exc

        return [
            ModelInfo(
                id=model.id,
                provider=self.name,
                display_name=model.id,
                capabilities=_classify(model.id),
                raw=model.model_dump(),
            )
            for model in page.data
        ]

    # -- 3. chat ----------------------------------------------------------

    def chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        # No hand-rolled chat method. Returning a BaseChatModel is what makes
        # this provider interchangeable inside a LangGraph node and visible to
        # Langfuse without any extra wiring.
        return ChatOpenAI(
            model=model_id,
            api_key=self._api_key,
            base_url=self._base_url,
            **kwargs,
        )


# Why `openai_provider.py` and not `openai.py`:
#
# Python 3 uses absolute imports, so `from openai import OpenAI` inside a module
# named `openai.py` does resolve to site-packages and works fine -- until
# something puts this directory on sys.path directly (running the file as a
# script, some test layouts, some editors). Then the module shadows the SDK and
# you get an import error that reads like the SDK is missing.
#
# `inception.py` is safe because the SDK is `inceptionai`. For consistency,
# name all four `*_provider.py` and rename the existing `inception.py` to match.
