"""An image goes down the VISION chain when a model cannot answer (agent/pipeline/vision.py
describe_with_fallback, agent/router/router.py Router.chain)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.pipeline.vision import describe_with_fallback
from agent.router import health
from agent.router.llm_provider.base import ProviderError


class _LLM:
    def __init__(self, answer=None, error=None):
        self.answer, self.error, self.calls = answer, error, 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.answer)


class _Router:
    def __init__(self, models):
        self.models = models  # [(provider, model_id, _LLM)]
        self.overrides = []

    def chain(self, task):
        return [SimpleNamespace(provider=p, model=SimpleNamespace(id=m)) for p, m, _ in self.models]

    def model_for(self, decision, **overrides):
        self.overrides.append(overrides)
        return next(llm for p, m, llm in self.models if (p, m) == (decision.provider, decision.model.id))


@pytest.fixture(autouse=True)
def fresh_health():
    with health.bind_health(health.Health()):
        yield


class _Exhausted(Exception):
    code = 429


DEPLETED = _Exhausted("429 RESOURCE_EXHAUSTED. Your prepayment credits are depleted.")


def test_the_first_model_that_answers_is_used():
    router = _Router([("gemini", "gemini-3.8-flash", _LLM("a cat")), ("gemini", "gemini-3.7-flash", _LLM("unused"))])
    assert describe_with_fallback(router, "aGk=", "image/png", "what?", max_retries=1) == ("a cat", "gemini:gemini-3.8-flash")
    assert router.overrides == [{"max_retries": 1}]


def test_an_account_out_of_credit_skips_the_whole_vendor_and_the_next_one_answers():
    g38, g37 = _LLM(error=DEPLETED), _LLM("never asked")
    claude = _LLM("a dog")
    router = _Router([("gemini", "gemini-3.8-flash", g38), ("gemini", "gemini-3.7-flash", g37),
                      ("anthropic", "claude-haiku-4-5-20251001", claude)])
    assert describe_with_fallback(router, "aGk=", "image/png", "what?")[1] == "anthropic:claude-haiku-4-5-20251001"
    # One refusal on credit cools every Gemini model; the next image does not ask Gemini again.
    assert g37.calls == 0
    describe_with_fallback(router, "aGk=", "image/png", "again?")
    assert g38.calls == 1 and claude.calls == 2


def test_a_rate_limited_model_gives_way_to_the_next_version():
    class _Busy(Exception):
        code = 429

    router = _Router([("gemini", "gemini-3.8-flash", _LLM(error=_Busy("429 too many requests"))),
                      ("gemini", "gemini-3.7-flash", _LLM("a bird"))])
    assert describe_with_fallback(router, "aGk=", "image/png", "what?") == ("a bird", "gemini:gemini-3.7-flash")
    assert health.HEALTH.cooling("gemini", "gemini-3.8-flash")
    assert not health.HEALTH.cooling("gemini", "gemini-3.7-flash")


def test_when_none_answers_every_reason_is_in_the_error():
    router = _Router([("gemini", "gemini-3.8-flash", _LLM(error=DEPLETED)),
                      ("openai", "gpt-5-mini", _LLM(error=RuntimeError("boom")))])
    with pytest.raises(ProviderError) as info:
        describe_with_fallback(router, "aGk=", "image/png", "what?")
    message = str(info.value)
    assert message.startswith("no vision model could answer -- gemini:gemini-3.8-flash: ")
    assert "openai:gpt-5-mini: " in message and "boom" in message


def test_no_configured_vision_model_says_so():
    with pytest.raises(ProviderError, match="no vision model is configured"):
        describe_with_fallback(_Router([]), "aGk=", "image/png", "what?")


def test_the_router_chain_lists_usable_candidates_ready_ones_first():
    from agent.router.mapping import Task
    from agent.router.router import FakeCatalogue, Router

    def info(provider, model_id):
        from agent.router.llm_provider.base import Capability, ModelInfo
        return ModelInfo(id=model_id, provider=provider, capabilities=frozenset({Capability.CHAT, Capability.VISION}))

    catalogue = FakeCatalogue({
        "gemini": [info("gemini", "gemini-3.8-flash"), info("gemini", "gemini-3.6-flash")],
        "anthropic": [info("anthropic", "claude-haiku-4-5-20251001")],
        "inception": [info("inception", "mercury-2.5")],
    })
    router = Router(catalogue=catalogue)
    health.HEALTH.note_rate_limit("gemini", "gemini-3.8-flash", retry_after=60)
    names = [f"{d.provider}:{d.model.id}" for d in router.chain(Task.VISION)]
    # 3.7 is not in this catalogue, openai has no key here; the cooling 3.8 goes last.
    assert names == ["gemini:gemini-3.6-flash", "anthropic:claude-haiku-4-5-20251001", "gemini:gemini-3.8-flash"]
