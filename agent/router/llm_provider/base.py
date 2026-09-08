"""Provider abstraction for Otto.

One rule governs this module: the base class holds only what *every* vendor
genuinely does. Anything a subset of vendors offer is a `Capability` plus an
optional `Protocol` -- never an inherited method that raises.

Subclasses implement exactly three things:

    _build_client()   construct the vendor SDK client
    _fetch_models()   one network call, normalised into `ModelInfo`
    chat_model()      return a configured LangChain `BaseChatModel`

Everything else is inherited. `chat_model` returning a LangChain model rather
than a bespoke `chat()` method is deliberate: streaming, tool binding, message
history, Langfuse callbacks and LangGraph node compatibility then come for
free and behave identically across vendors.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from langchain_core.language_models import BaseChatModel

__all__ = [
    "Capability",
    "ModelInfo",
    "ProviderStatus",
    "HealthReport",
    "ProviderError",
    "AuthError",
    "ModelNotFound",
    "CapabilityNotSupported",
    "ProviderUnavailable",
    "Completion",
    "BaseProvider",
    "SupportsFIM",
    "SupportsEdit",
    "SupportsEmbeddings",
]


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


class Capability(StrEnum):
    """A thing a provider or a specific model can do.

    Used in two distinct places, and they mean different things:

      * `BaseProvider.capabilities` -- what the *vendor* offers.
      * `ModelInfo.capabilities`    -- what *that model* offers.

    Inception supports FIM; a given Inception model may not. Route on the
    per-model set, gate the CLI on the per-provider one.
    """

    CHAT = "chat"
    TOOLS = "tools"
    VISION = "vision"
    REASONING = "reasoning"
    STRUCTURED_OUTPUT = "structured_output"
    FIM = "fim"
    EDIT = "edit"
    EMBEDDINGS = "embeddings"


# --------------------------------------------------------------------------
# Value types
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model, normalised across vendors.

    `raw` keeps the untouched vendor payload so provider-specific code can
    reach fields the common shape drops, without another network call. It is
    excluded from `repr` because vendor payloads are large.
    """

    id: str
    provider: str
    display_name: str | None = None
    capabilities: frozenset[Capability] = frozenset()
    context_window: int | None = None
    max_output_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def spec(self) -> str:
        """Fully-qualified reference, e.g. ``inception:mercury-2``."""
        return f"{self.provider}:{self.id}"

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def __str__(self) -> str:
        return self.spec


@dataclass(frozen=True, slots=True)
class Completion:
    """A raw-SDK completion: the text, plus the usage the endpoint reported.

    FIM and edit bypass LangChain, so nothing collects their token counts for
    us. Returning a bare `str` threw them away -- and a call with no usage has
    no cost, which silently defeats the whole point of a cost-driven router.

    `usage` uses Langfuse's key names (input / output / total, plus optional
    cached_input and reasoning) so it can be handed to `usage_details` without
    translation.
    """

    text: str
    usage: Mapping[str, int] | None = None

    def __str__(self) -> str:
        return self.text


class ProviderStatus(StrEnum):
    """Outcome of `BaseProvider.health_check`."""

    OK = "ok"
    NO_KEY = "no key"
    AUTH_FAILED = "auth failed"
    UNREACHABLE = "unreachable"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Per-provider result for the `doctor` / `models` commands.

    Deliberately a value, not an exception: checking four providers should
    report four outcomes, never abort on the first failure.
    """

    provider: str
    status: ProviderStatus
    model_count: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is ProviderStatus.OK


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ProviderError(Exception):
    """Base for every provider failure.

    Subclasses MUST translate vendor SDK exceptions into these before they
    escape `_fetch_models`. That translation is the whole reason `health_check`
    can stay vendor-agnostic -- it is the one piece of per-provider work that
    is easy to forget.
    """


class AuthError(ProviderError):
    """Key missing, malformed, or rejected by the vendor."""


class ProviderUnavailable(ProviderError):
    """Network failure, timeout, or vendor-side outage."""


class ModelNotFound(ProviderError):
    """Requested model id is not offered by this provider."""


class CapabilityNotSupported(ProviderError):
    """Provider or model cannot do the requested thing."""


# --------------------------------------------------------------------------
# Base provider
# --------------------------------------------------------------------------


class BaseProvider(ABC):
    """A single LLM vendor.

    Subclasses set the class attributes below and implement the three
    abstract methods. Nothing here performs I/O at construction time except
    building the SDK client -- model discovery is lazy, so constructing a
    provider is cheap and safe.
    """

    #: Short lowercase identifier used in model specs, e.g. ``"inception"``.
    name: str = ""
    #: Environment variable holding this vendor's API key.
    env_var: str = ""
    #: Optional default endpoint, for OpenAI-compatible vendors.
    default_base_url: str | None = None
    #: What the vendor offers. Narrow this per subclass.
    capabilities: frozenset[Capability] = frozenset({Capability.CHAT})

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
    ) -> None:
        if not self.name or not self.env_var:
            raise TypeError(
                f"{type(self).__name__} must set both 'name' and 'env_var'"
            )

        key = api_key or os.environ.get(self.env_var)
        if not key:
            raise AuthError(f"{self.name}: {self.env_var} is not set")

        self._api_key: str = key
        self._base_url: str | None = base_url or self.default_base_url
        self._models: list[ModelInfo] | None = None
        self._client: Any = self._build_client()

    # -- subclass responsibilities ----------------------------------------

    @abstractmethod
    def _build_client(self) -> Any:
        """Construct and return the vendor SDK client.

        Use `self._api_key` and `self._base_url` explicitly rather than
        letting the SDK read the environment on its own -- an explicitly
        passed key is testable and cannot be shadowed by a stale export.
        """

    @abstractmethod
    def _fetch_models(self) -> list[ModelInfo]:
        """Fetch every model this provider offers, as `ModelInfo`.

        Called at most once per instance unless `refresh=True`. Set each
        model's `capabilities` here -- it is the only place with enough
        vendor context to know them.

        Must raise `AuthError` on a rejected key, `ProviderUnavailable` on a
        network or vendor fault, and `ProviderError` for anything else.
        Letting a raw SDK exception escape breaks `health_check`.
        """

    @abstractmethod
    def chat_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        """Return a LangChain chat model bound to `model_id`.

        Do not implement chat by hand. Returning a `BaseChatModel` is what
        keeps every provider interchangeable inside a LangGraph node and
        visible to Langfuse.
        """

    # -- configuration ----------------------------------------------------

    @classmethod
    def is_configured(cls) -> bool:
        """True if the key is present, without constructing the provider.

        `__init__` raises on a missing key, so a `doctor` command that wants
        to report rather than crash needs this cheap check first.
        """
        return bool(os.environ.get(cls.env_var))

    @property
    def masked_key(self) -> str:
        """Key rendered for display. Never log or print `_api_key` itself."""
        tail = self._api_key[-4:] if len(self._api_key) >= 4 else ""
        return f"{'*' * 8}{tail}"

    # -- model discovery --------------------------------------------------

    def list_models(
        self,
        capability: Capability | None = None,
        *,
        refresh: bool = False,
    ) -> list[ModelInfo]:
        """Models offered, optionally narrowed to one capability.

        Cached after the first call; pass `refresh=True` to re-fetch.
        """
        if self._models is None or refresh:
            self._models = self._fetch_models()
        if capability is None:
            return list(self._models)
        return [m for m in self._models if capability in m.capabilities]

    def model_ids(self, capability: Capability | None = None) -> list[str]:
        return [m.id for m in self.list_models(capability)]

    def resolve_model(self, model_id: str) -> ModelInfo:
        """Look up a model by id, raising `ModelNotFound` on a typo.

        Call this before `chat_model` so a bad id fails with a useful list
        instead of a vendor 404 halfway through a graph run.
        """
        for model in self.list_models():
            if model.id == model_id:
                return model
        raise ModelNotFound(
            f"{model_id!r} is not offered by {self.name}. "
            f"Available: {', '.join(self.model_ids()) or '<none>'}"
        )

    # -- capability gating ------------------------------------------------

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def require(self, capability: Capability) -> None:
        """Guard clause for optional capabilities.

        Prefer this over `isinstance(p, SupportsFIM)` at call sites -- it
        yields an error naming the provider and the capability.
        """
        if not self.supports(capability):
            raise CapabilityNotSupported(
                f"{self.name} does not support {capability}"
            )

    # -- health -----------------------------------------------------------

    def health_check(self) -> HealthReport:
        """Verify the key actually works by listing models.

        Env-var presence proves nothing -- only a real call distinguishes a
        valid key from a revoked one. Never raises.
        """
        try:
            models = self.list_models(refresh=True)
        except AuthError as exc:
            return HealthReport(self.name, ProviderStatus.AUTH_FAILED, detail=str(exc))
        except ProviderUnavailable as exc:
            return HealthReport(self.name, ProviderStatus.UNREACHABLE, detail=str(exc))
        except ProviderError as exc:
            return HealthReport(self.name, ProviderStatus.ERROR, detail=str(exc))
        except Exception as exc:  # a subclass forgot to translate
            return HealthReport(
                self.name,
                ProviderStatus.ERROR,
                detail=f"untranslated {type(exc).__name__}: {exc}",
            )
        return HealthReport(self.name, ProviderStatus.OK, model_count=len(models))

    @classmethod
    def check(cls) -> HealthReport:
        """Health check that tolerates a missing key and never raises."""
        if not cls.is_configured():
            return HealthReport(
                cls.name or cls.__name__,
                ProviderStatus.NO_KEY,
                detail=f"{cls.env_var} is not set",
            )
        try:
            return cls().health_check()
        except ProviderError as exc:
            return HealthReport(cls.name, ProviderStatus.ERROR, detail=str(exc))

    # -- misc -------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "?" if self._models is None else str(len(self._models))
        return (
            f"<{type(self).__name__} name={self.name!r} "
            f"models={loaded} caps={sorted(c.value for c in self.capabilities)}>"
        )


# --------------------------------------------------------------------------
# Optional capabilities
# --------------------------------------------------------------------------
#
# These are Protocols, not base-class methods, on purpose. A base method that
# raises NotImplementedError lies in its own signature: the caller cannot tell
# whether it will work without calling it. A Protocol lets both a static type
# checker and `isinstance` answer the question up front.


@runtime_checkable
class SupportsFIM(Protocol):
    """Fill-in-the-middle completion. Not an OpenAI-standard endpoint."""

    def fim(
        self,
        model_id: str,
        prefix: str,
        suffix: str = "",
        **kwargs: Any,
    ) -> Completion: ...


@runtime_checkable
class SupportsEdit(Protocol):
    """Next-edit prediction over a code region.

    Note the absence of an `instruction` argument. Edit endpoints of this kind
    infer the change from surrounding context -- the file, recently viewed
    snippets, recent diffs, and cursor position -- rather than from a natural
    language instruction. Providers that want an instruction should pass it
    through `**kwargs` or expose a separate method.
    """

    def code_edit(
        self,
        model_id: str,
        code_to_edit: str,
        *,
        current_file: str = "",
        **kwargs: Any,
    ) -> Completion: ...


@runtime_checkable
class SupportsEmbeddings(Protocol):
    def embed(
        self,
        model_id: str,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]: ...
