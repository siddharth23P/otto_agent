"""Custom OpenAI-compatible endpoints (agent/router/llm_provider/custom.py)
and the registry/mapping/temperature hooks that let one behave as a vendor.
Offline: the SDK client is faked where it would be constructed."""
from __future__ import annotations

import pytest

from agent.router import mapping
from agent.router.llm_provider import (
    all_models, custom, is_custom, provider_class, provider_names,
    register_custom, reset, unregister_custom,
)
from agent.router.llm_provider import openai_provider, temperature
from agent.router.llm_provider.base import AuthError, Capability, ProviderStatus
from agent.router.mapping import Candidate, MappingError, TASK_ROUTES, Task, validate

CHAT = Capability.CHAT


@pytest.fixture
def registered(monkeypatch):
    """A custom endpoint called `local`, configured, with a fake SDK client
    whose /v1/models lists two ids. Torn down completely afterwards."""
    captured: dict = {}

    class FakeModel:
        def __init__(self, id): self.id = id
        def model_dump(self): return {"id": self.id}

    class FakeModels:
        def list(self):
            return type("Page", (), {"data": [FakeModel("llama3.2:latest"), FakeModel("qwen-vl")]})()

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"], captured["base_url"] = api_key, base_url
            self.models = FakeModels()

    monkeypatch.setattr(openai_provider, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("LOCAL_API_KEY", "x")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:11434/v1/")
    cls = custom.openai_compatible("local")
    register_custom("local", cls)
    mapping.register_provider_name("local")
    temperature.register_provider("local")
    yield cls, captured
    unregister_custom("local")
    mapping.unregister_provider_name("local")
    temperature._BY_PROVIDER.pop("local", None)
    temperature._FIXED_TEMPERATURE_PATTERNS.pop("local", None)
    reset()


# ---- names and variables ---------------------------------------------------

@pytest.mark.parametrize("bad", ["OpenRouter", "has space", "1st", "a", "openai", "otto", "x" * 40, ""])
def test_bad_or_reserved_names_are_rejected(bad):
    with pytest.raises(ValueError):
        custom.validate_name(bad)


def test_names_become_environment_variables():
    assert custom.key_var("open-router") == "OPEN_ROUTER_API_KEY"
    assert custom.url_var("open-router") == "OPEN_ROUTER_BASE_URL"
    custom.validate_name("open-router")
    custom.validate_name("vllm_2")


# ---- configuration ------------------------------------------------------

def test_a_key_without_a_url_is_not_configured(monkeypatch):
    cls = custom.openai_compatible("remote")
    monkeypatch.setenv("REMOTE_API_KEY", "k")
    monkeypatch.delenv("REMOTE_BASE_URL", raising=False)
    assert cls.is_configured() is False
    report = cls.check()
    assert report.status is ProviderStatus.NO_KEY
    assert "REMOTE_BASE_URL" in report.detail
    with pytest.raises(AuthError):
        cls()          # never silently talks to api.openai.com


def test_the_client_is_built_against_the_base_url(registered):
    cls, captured = registered
    provider = cls()
    assert captured == {"api_key": "x", "base_url": "http://localhost:11434/v1"}, "trailing slash stripped"
    assert provider.name == "local"


# ---- classification -----------------------------------------------------

@pytest.mark.parametrize("model_id, expected", [
    ("text-embedding-3-small", {Capability.EMBEDDINGS}),
    ("whisper-1", set()),
    ("llama3.2:latest", {CHAT, Capability.TOOLS}),
    ("openai/o4-mini", {CHAT, Capability.TOOLS, Capability.REASONING}),
    ("deepseek-r1:32b", {CHAT, Capability.TOOLS, Capability.REASONING}),
    ("anthropic/claude-sonnet-4", {CHAT, Capability.TOOLS, Capability.REASONING, Capability.VISION}),
    ("qwen2.5-vl", {CHAT, Capability.TOOLS, Capability.VISION}),
])
def test_classify_generic(model_id, expected):
    assert set(custom.classify_generic(model_id)) == expected


def test_a_capability_override_beats_the_guess(monkeypatch):
    monkeypatch.setenv("PINNED_API_KEY", "k")
    monkeypatch.setenv("PINNED_BASE_URL", "http://h/v1")
    cls = custom.openai_compatible("pinned", capability_overrides={"llama3.2:latest": ["chat", "vision"]})
    provider = cls.__new__(cls)
    assert provider._classify("llama3.2:latest") == frozenset({CHAT, Capability.VISION})
    assert provider._classify("other") == custom.classify_generic("other")


# ---- registry -------------------------------------------------------------

def test_a_registered_endpoint_is_a_provider_everywhere(registered):
    cls, _ = registered
    assert "local" in provider_names()
    assert is_custom("local")
    assert provider_class("local") is cls
    found = {m.id: m for m in all_models() if m.provider == "local"}
    assert set(found) == {"llama3.2:latest", "qwen-vl"}
    assert Capability.VISION in found["qwen-vl"].capabilities
    assert found["llama3.2:latest"].context_window is None


def test_unregistering_removes_every_trace(registered):
    unregister_custom("local")
    mapping.unregister_provider_name("local")
    assert "local" not in provider_names()
    assert "local" not in mapping.known_providers()
    assert "local" not in mapping.PARAMS_BY_PROVIDER


def test_a_builtin_cannot_be_replaced():
    from agent.router.llm_provider import UnknownProvider
    with pytest.raises(UnknownProvider):
        register_custom("openai", object)


# ---- mapping and temperature ------------------------------------------------

def test_validate_accepts_a_pin_on_a_registered_endpoint(registered):
    validate({**TASK_ROUTES, Task.CHAT_FAST: (
        Candidate(spec="local:llama3.2:latest", requires=frozenset({CHAT}),
                  params={"temperature": 0.2, "max_tokens": 512}),)})
    with pytest.raises(MappingError, match="diffusing"):
        validate({**TASK_ROUTES, Task.CHAT_FAST: (
            Candidate(spec="local:llama3.2:latest", requires=frozenset({CHAT}),
                      params={"diffusing": True}),)})


def test_an_unregistered_endpoint_is_still_unknown():
    with pytest.raises(MappingError, match="unknown provider"):
        validate({**TASK_ROUTES, Task.CHAT_FAST: (
            Candidate(spec="nowhere:model", requires=frozenset({CHAT})),)})


def test_temperature_policy_for_a_custom_endpoint(registered):
    assert temperature.apply_to_params("local", "openai/o3", {"temperature": 0.0}) == {}
    assert temperature.apply_to_params("local", "llama3.2:latest", {"temperature": 3.0}) == {"temperature": 2.0}
