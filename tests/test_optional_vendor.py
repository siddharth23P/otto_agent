"""A vendor SDK that is not installed disables that vendor, not Otto.

An embedded Otto (agent/embed.py) can ship without one of the compiled
packages a vendor adapter imports -- tiktoken under langchain_openai, jiter
under langchain_anthropic. `Router()` is built at import of the pipeline, so
an ImportError from one adapter used to be an ImportError for the package."""
from __future__ import annotations

import sys

from agent.router import llm_provider
from agent.router.router import Router


def _without_openai_sdk(monkeypatch):
    # The adapter module may already be imported by an earlier test; evict it
    # so provider_class() imports it again and meets the missing SDK.
    monkeypatch.delitem(sys.modules, "agent.router.llm_provider.openai_provider", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_openai", None)  # import raises
    llm_provider.reset()


def test_router_constructs_when_one_vendor_sdk_is_missing(monkeypatch):
    _without_openai_sdk(monkeypatch)
    try:
        router = Router()
        assert "openai" not in router.usable()
        assert router.ready()  # Inception is conftest's placeholder key, still configured
    finally:
        llm_provider.reset()


def test_health_report_names_the_import_failure(monkeypatch):
    _without_openai_sdk(monkeypatch)
    try:
        reports = {r.provider: r for r in llm_provider.health_report()}
        assert reports["openai"].status is llm_provider.ProviderStatus.ERROR
        assert "import failed" in reports["openai"].detail
    finally:
        llm_provider.reset()
