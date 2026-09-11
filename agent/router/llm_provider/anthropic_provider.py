"""Anthropic provider.

Discovery goes through the `anthropic` SDK; chat goes through `ChatAnthropic`.

That split is deliberate and differs from `inception_provider.py`. Inception has
no LangChain integration, so a custom `BaseChatModel` was the only way to reach
it. Anthropic has a maintained one that already handles content blocks, tool
binding, structured output, prompt caching and extended thinking -- reproducing
that by hand over the raw SDK would be a few hundred lines of duplication with
no capability gained.
"""

from __future__ import annotations

from typing import Any

from anthropic import (
    Anthropic,
    APIConnectionError,
    APIStatusError,
    AnthropicError,
    AuthenticationError,
    PermissionDeniedError,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from agent.router.llm_provider.base import (
    AuthError,
    BaseProvider,
    Capability,
    ModelInfo,
    ProviderError,
    ProviderUnavailable,
)

__all__ = ["AnthropicProvider"]


def _translate(exc: Exception) -> ProviderError:
    # AuthenticationError and PermissionDeniedError both subclass APIStatusError,
    # so they must be tested before it.
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return AuthError(f"anthropic: key rejected ({exc})")
    if isinstance(exc, APIConnectionError):  # APITimeoutError subclasses this
        return ProviderUnavailable(f"anthropic: {exc}")
    if isinstance(exc, APIStatusError):
        if exc.status_code in (401, 403):
            return AuthError(f"anthropic: HTTP {exc.status_code}")
        return ProviderError(f"anthropic: HTTP {exc.status_code}")
    return ProviderError(f"anthropic: {exc}")


def _capabilities_of(model: Any) -> set[Capability]:
    """Read capabilities off `ModelInfo.capabilities`.

    Anthropic publishes structured capability flags, so nothing here is guessed
    from the model name -- only TOOLS is assumed, because the Models API exposes
    no tool-use flag and every current Claude model supports tool use.
    """
    caps = {Capability.CHAT, Capability.TOOLS}

    published = getattr(model, "capabilities", None)
    if published is None:
        # Older API versions omit the block entirely. Report the floor rather
        # than inventing flags.
        return caps

    if getattr(published.image_input, "supported", False):
        caps.add(Capability.VISION)
    if getattr(published.structured_outputs, "supported", False):
        caps.add(Capability.STRUCTURED_OUTPUT)
    # Extended thinking and effort levels are two expressions of the same idea.
    if getattr(published.thinking, "supported", False) or getattr(
        published.effort, "supported", False
    ):
        caps.add(Capability.REASONING)

    return caps


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"
    default_base_url = None
    capabilities = frozenset(
        {
            Capability.CHAT,
            Capability.TOOLS,
            Capability.VISION,
            Capability.REASONING,
            Capability.STRUCTURED_OUTPUT,
        }
    )

    def _build_client(self) -> Anthropic:
        kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return Anthropic(**kwargs)

    def _fetch_models(self) -> list[ModelInfo]:
        try:
            # The Models API paginates and defaults to 20 per page. In this SDK
            # SyncPage.__iter__ walks iter_pages() itself, so plain iteration
            # already spans every page -- but the `limit` still matters: it sets
            # the page size, and without it you pay a round trip per 20 models.
            models = list(self._client.models.list(limit=1000))
        except AnthropicError as exc:
            raise _translate(exc) from exc

        return [
            ModelInfo(
                id=model.id,
                provider=self.name,
                display_name=model.display_name,
                capabilities=frozenset(_capabilities_of(model)),
                context_window=model.max_input_tokens,
                max_output_tokens=model.max_tokens,
                raw=model.model_dump(mode="json"),
            )
            for model in models
        ]

    def chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        if self._base_url:
            kwargs.setdefault("base_url", self._base_url)
        return ChatAnthropic(model=model_id, api_key=self._api_key, **kwargs)
