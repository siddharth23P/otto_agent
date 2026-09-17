"""agent/router/automap.py: a proposal per task from a detected pool.
Pure functions over ModelInfo lists; no router, no network."""
from __future__ import annotations

import random

import pytest

from agent.router import automap
from agent.router.llm_provider.base import Capability, ModelInfo
from agent.router.mapping import Task

CHAT, R, V, T = Capability.CHAT, Capability.REASONING, Capability.VISION, Capability.TOOLS


def m(id, provider, caps, ctx):
    return ModelInfo(id=id, provider=provider, capabilities=frozenset(caps), context_window=ctx)


MERCURY = m("mercury-2.5", "inception", {CHAT}, 260_000)
EDIT2 = m("mercury-edit-2", "inception", {Capability.FIM, Capability.EDIT}, 128_000)
GPT = m("gpt-5-mini", "openai", {CHAT, R, V, T}, 400_000)
HAIKU = m("claude-haiku-4-5-20251001", "anthropic", {CHAT, R, T}, 200_000)
FLASH_LITE = m("gemini-flash-lite-latest", "gemini", {CHAT, V, T}, 1_000_000)
FLASH3 = m("gemini-3.8-flash", "gemini", {CHAT, V, T, R}, 1_000_000)
LOCAL_SMALL = m("llama3.2:latest", "local", {CHAT, T}, 8_000)
LOCAL_R1 = m("deepseek-r1:32b", "local", {CHAT, T, R}, 64_000)
LOCAL_BIG = m("llama3.3:70b", "local", {CHAT, T}, 128_000)


def test_the_shipped_head_is_kept_when_it_resolves():
    p = automap.propose([MERCURY, EDIT2, GPT, HAIKU, FLASH_LITE, FLASH3])
    assert p[Task.REASON].spec == "openai:gpt-5-mini" and p[Task.REASON].source == "shipped"
    assert p[Task.CODE_COMPLETE].spec == "inception:mercury-edit-2"
    assert p[Task.VISION].spec == "gemini:gemini-3.8-flash"
    assert not p[Task.REASON].needs_pin


def test_a_later_shipped_candidate_is_a_fallback():
    p = automap.propose([MERCURY, EDIT2])
    assert p[Task.REASON].spec == "inception:mercury-2.5" and p[Task.REASON].source == "fallback"
    assert p[Task.REASON].needs_pin


def test_the_pool_is_used_only_when_no_shipped_candidate_resolves():
    p = automap.propose([LOCAL_SMALL, LOCAL_R1, LOCAL_BIG])
    assert p[Task.CHAT_FAST].spec == "local:llama3.2:latest", "smallest context for the cheap seat"
    assert p[Task.REASON].spec == "local:deepseek-r1:32b", "REASONING first, before the bigger CHAT-only model"
    assert p[Task.SUMMARIZE].spec == "local:llama3.2:latest"
    assert p[Task.PLAN].spec == "local:deepseek-r1:32b"
    assert all(x.source == "pool" for x in (p[Task.CHAT_FAST], p[Task.REASON]))


def test_reason_relaxes_to_chat_when_nothing_reasons():
    p = automap.propose([LOCAL_SMALL, LOCAL_BIG])
    assert p[Task.REASON].spec == "local:llama3.3:70b", "largest context once the tier is relaxed"


def test_vision_never_proposes_a_model_that_cannot_see():
    p = automap.propose([MERCURY, LOCAL_BIG, HAIKU])
    assert p[Task.VISION].model is None and "vision" in p[Task.VISION].reason


def test_provider_bound_tasks_stay_bound():
    p = automap.propose([GPT, LOCAL_R1])
    assert p[Task.CODE_COMPLETE].model is None and "inception" in p[Task.CODE_COMPLETE].reason
    assert p[Task.WEB].model is None and "anthropic" in p[Task.WEB].reason
    p = automap.propose([HAIKU])
    assert p[Task.WEB].spec == "anthropic:claude-haiku-4-5-20251001"


def test_an_empty_pool_proposes_nothing():
    p = automap.propose([])
    assert all(x.model is None and x.source == "none" for x in p.values())
    assert set(p) == set(Task)


def test_proposals_do_not_depend_on_pool_order():
    pool = [MERCURY, EDIT2, GPT, HAIKU, FLASH_LITE, FLASH3, LOCAL_SMALL, LOCAL_R1, LOCAL_BIG]
    baseline = {t: x.spec for t, x in automap.propose(pool).items()}
    for seed in range(5):
        shuffled = list(pool)
        random.Random(seed).shuffle(shuffled)
        assert {t: x.spec for t, x in automap.propose(shuffled).items()} == baseline
