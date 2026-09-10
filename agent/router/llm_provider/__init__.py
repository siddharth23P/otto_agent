"""Provider registry -- Inception is the only vendor Otto knows.

Still a registry, not a hardcoded import: a second vendor showing up later is
a dict entry here, not a rewrite of every call site that reaches through
get_provider()/provider_names(). Importing this package stays cheap for the
same reason it always did -- lazy import, so a missing optional dependency
breaks one provider, not the whole app.
"""

from __future__ import annotations

import importlib
import os
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from .base import (
    AuthError,
    BaseProvider,
    Capability,
    CapabilityNotSupported,
    HealthReport,
    ModelInfo,
    ModelNotFound,
    ProviderError,
    ProviderStatus,
    ProviderUnavailable,
    SupportsEdit,
    SupportsEmbeddings,
    SupportsFIM,
)

__all__ = [
    "AuthError",
    "BaseProvider",
    "Capability",
    "CapabilityNotSupported",
    "HealthReport",
    "ModelInfo",
    "ModelNotFound",
    "ProviderError",
    "ProviderStatus",
    "ProviderUnavailable",
    "SupportsEdit",
    "SupportsEmbeddings",
    "SupportsFIM",
    "UnknownProvider",
    "FALLBACK_MODEL_SPEC",
    "default_model_spec",
    "provider_names",
    "provider_class",
    "get_provider",
    "parse_spec",
    "get_chat_model",
    "health_report",
    "all_models",
    "reset",
]

#: Used when OTTO_DEFAULT_MODEL is unset. Mercury 2.5 is Inception's current
#: chat model (2026-09-09; mercury-2 is still documented but no longer what
#: a fresh install should reach for).
FALLBACK_MODEL_SPEC = "inception:mercury-2.5"

_PROVIDER_MODULES: dict[str, tuple[str, str]] = {
    "inception": (".inception_provider", "InceptionProvider"),
}


class UnknownProvider(ProviderError):
    """Spec named a provider that is not registered."""


def default_model_spec() -> str:
    """Resolve the default model, honouring OTTO_DEFAULT_MODEL.

    A function, not a module constant: this package may well be imported before
    the Typer callback calls `load_dotenv`, and a constant evaluated at import
    time would freeze whatever the environment happened to hold then.
    """
    return os.environ.get("OTTO_DEFAULT_MODEL") or FALLBACK_MODEL_SPEC


def provider_names() -> tuple[str, ...]:
    return tuple(_PROVIDER_MODULES)


@lru_cache(maxsize=None)
def provider_class(name: str) -> type[BaseProvider]:
    """Import and return a provider class by name."""
    try:
        module_path, attr = _PROVIDER_MODULES[name]
    except KeyError:
        raise UnknownProvider(
            f"unknown provider {name!r}. Known: {', '.join(provider_names())}"
        ) from None
    module = importlib.import_module(module_path, __package__)
    return getattr(module, attr)


@lru_cache(maxsize=None)
def get_provider(name: str) -> BaseProvider:
    """Return a constructed, cached provider. Raises `AuthError` without a key."""
    return provider_class(name)()


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"provider:model"``. A bare model name uses the default provider."""
    provider, sep, model_id = spec.partition(":")
    if not sep:
        default_provider, _ = parse_spec(default_model_spec())
        return default_provider, spec
    if not model_id:
        raise UnknownProvider(f"spec {spec!r} names a provider but no model")
    return provider, model_id


def get_chat_model(spec: str | None = None, **kwargs) -> BaseChatModel:
    """Resolve ``"provider:model"`` to a LangChain chat model.

    Validates the id against the provider's catalogue first, so a typo fails
    here with the list of alternatives rather than as a vendor 404 midway
    through a graph run.
    """
    provider_name, model_id = parse_spec(spec or default_model_spec())
    provider = get_provider(provider_name)
    provider.resolve_model(model_id)
    return provider.chat_model(model_id, **kwargs)


def health_report() -> list[HealthReport]:
    """Check every registered provider. Never raises -- this is the doctor command."""
    reports: list[HealthReport] = []
    for name in provider_names():
        try:
            reports.append(provider_class(name).check())
        except Exception as exc:  # import failure, e.g. SDK not installed
            reports.append(
                HealthReport(name, ProviderStatus.ERROR, detail=f"import failed: {exc}")
            )
    return reports


def all_models(capability: Capability | None = None) -> list[ModelInfo]:
    """Every model across every configured provider, skipping unusable ones."""
    models: list[ModelInfo] = []
    for name in provider_names():
        cls = provider_class(name)
        if not cls.is_configured():
            continue
        try:
            models.extend(get_provider(name).list_models(capability))
        except ProviderError:
            # One dead key must not blank out the rest.
            continue
    return models


def reset() -> None:
    """Drop cached provider instances -- call after changing keys in-process."""
    get_provider.cache_clear()
    provider_class.cache_clear()
