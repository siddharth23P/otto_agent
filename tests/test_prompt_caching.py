"""Input-token caching is on for all four providers (agent/router/llm_provider/base.py
PROMPT_CACHE_ENV), and what each vendor reports as cached is priced as cached."""
from __future__ import annotations

import pytest

from agent.pipeline.usage import UsageLedger
from agent.router.llm_provider import base
from agent.router.llm_provider.anthropic_provider import PROMPT_CACHE, AnthropicProvider
from agent.router.llm_provider.inception_provider import _usage_metadata
from agent.router.llm_provider.openai_provider import OpenAIProvider


def _anthropic(**kw):
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._api_key, provider._base_url = "sk-ant-test", kw.get("base_url")
    return provider


def _openai(base_url=None):
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider._api_key, provider._base_url = "sk-test", base_url
    return provider


def test_anthropic_asks_for_automatic_caching_at_the_top_of_the_request(monkeypatch):
    monkeypatch.delenv(base.PROMPT_CACHE_ENV, raising=False)
    llm = _anthropic().chat_model("claude-haiku-4-5-20251001", max_tokens=64)
    assert llm.model_kwargs["cache_control"] == PROMPT_CACHE
    payload = llm._get_request_payload([("human", "hi")])
    assert payload["cache_control"] == {"type": "ephemeral"}


def test_a_route_keeps_its_own_model_kwargs_and_may_opt_out(monkeypatch):
    monkeypatch.delenv(base.PROMPT_CACHE_ENV, raising=False)
    llm = _anthropic().chat_model("claude-haiku-4-5", model_kwargs={"service_tier": "auto"})
    assert llm.model_kwargs == {"service_tier": "auto", "cache_control": PROMPT_CACHE}
    llm = _anthropic().chat_model("claude-haiku-4-5", model_kwargs={"cache_control": {"type": "ephemeral", "ttl": "1h"}})
    assert llm.model_kwargs["cache_control"]["ttl"] == "1h"
    llm = _anthropic().chat_model("claude-haiku-4-5", model_kwargs={"cache_control": None})
    assert "cache_control" not in llm.model_kwargs


def test_openai_routes_a_models_calls_to_one_cache(monkeypatch):
    monkeypatch.delenv(base.PROMPT_CACHE_ENV, raising=False)
    llm = _openai().chat_model("gpt-5-mini")
    assert llm.model_kwargs["prompt_cache_key"] == "otto:gpt-5-mini"
    assert "prompt_cache_key" in llm._get_request_payload([("human", "hi")])
    # An OpenAI-compatible server is not sent a field it may not know.
    assert "prompt_cache_key" not in _openai("http://localhost:11434/v1").chat_model("llama3").model_kwargs


@pytest.mark.parametrize("value", ["0", "false", "OFF"])
def test_the_switch_turns_it_off(monkeypatch, value):
    monkeypatch.setenv(base.PROMPT_CACHE_ENV, value)
    assert "cache_control" not in _anthropic().chat_model("claude-haiku-4-5").model_kwargs
    assert "prompt_cache_key" not in _openai().chat_model("gpt-5-mini").model_kwargs


def test_inception_reports_its_cached_tokens():
    assert _usage_metadata(100, 5, 105, 80) == {
        "input_tokens": 100, "output_tokens": 5, "total_tokens": 105,
        "input_token_details": {"cache_read": 80}}
    assert "input_token_details" not in _usage_metadata(100, 5, 105, 0)
    assert "input_token_details" not in _usage_metadata(100, 5, 105, None)


def test_inception_streaming_carries_the_cached_count():
    from tests.test_inception_stream import accumulate, content_chunk, model, usage_chunk

    llm = model([content_chunk("hi"), usage_chunk(
        {"prompt_tokens": 2000, "completion_tokens": 3, "total_tokens": 2003, "cached_input_tokens": 1800})])
    assert accumulate(llm).usage_metadata["input_token_details"] == {"cache_read": 1800}


def test_cached_reads_and_writes_reach_the_snapshot():
    ledger = UsageLedger()
    ledger.record("claude-haiku-4-5", {
        "input_tokens": 5000, "output_tokens": 10, "total_tokens": 5010,
        "input_token_details": {"cache_read": 4000, "cache_creation": 900}})
    snap = ledger.snapshot()
    assert snap["cached_input_tokens"] == 4000 and snap["cache_write_tokens"] == 900
    assert snap["models"][0]["cache_write_tokens"] == 900


def test_inception_reads_the_cached_count_where_the_api_puts_it():
    from agent.router.llm_provider.inception_provider import _cached_tokens

    assert _cached_tokens({"prompt_tokens": 9, "prompt_tokens_details": {"cached_tokens": 7}}) == 7
    assert _cached_tokens({"cached_input_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}}) == 5
    assert not _cached_tokens({"cached_input_tokens": None, "prompt_tokens_details": {"cached_tokens": 0}})
    assert not _cached_tokens({"prompt_tokens": 9})


def test_an_anthropic_write_reported_by_its_ttl_is_priced_as_a_write():
    # The shape langchain-anthropic 1.7 returns for a first cached call (seen live, 2026-09-17).
    ledger = UsageLedger()
    ledger.record("claude-haiku-4-5", {
        "input_tokens": 12203, "output_tokens": 4, "total_tokens": 12207,
        "input_token_details": {"cache_read": 0, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 12200, "ephemeral_1h_input_tokens": 0}})
    assert ledger.snapshot()["cache_write_tokens"] == 12200
    ledger.record("claude-haiku-4-5", {
        "input_tokens": 12203, "output_tokens": 4, "total_tokens": 12207,
        "input_token_details": {"cache_read": 12200, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}})
    snap = ledger.snapshot()
    assert snap["cached_input_tokens"] == 12200 and snap["cache_write_tokens"] == 12200
