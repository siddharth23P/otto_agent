"""Tests for resolution — Phase 2.

Nothing here needs an API key, a network connection, or a vendor SDK. That is
the point of the `Catalogue` protocol: `FakeCatalogue` is a dict, and "provider
is configured" means "provider is a key in that dict".

Fixtures use the real `TASK_ROUTES`. Routing behaviour is only interesting
against the table you actually ship, and the table already contains both
single-candidate chains (CHAT_FAST, CODE_COMPLETE) and three- and
four-candidate ones (REASON, PLAN, SUMMARIZE).
"""

from __future__ import annotations

import pytest

from agent.router import router as router_mod
from agent.router.llm_provider.base import BaseProvider, Capability, ModelInfo
from agent.router.mapping import TASK_ROUTES, Candidate, Endpoint, Preference, Task
from agent.router.router import (
    Catalogue,
    FakeCatalogue,
    NoViableRoute,
    RegistryCatalogue,
    Router,
    Skip,
    render,
)

CHAT, TOOLS = Capability.CHAT, Capability.TOOLS


def model(id: str, provider: str, caps, ctx: int) -> ModelInfo:
    """Build a ModelInfo without the fields no test cares about.

    ModelInfo has seven fields; a routing test cares about three. Spelling out
    display_name and raw in every fixture buries the differences that matter.
    """
    return ModelInfo(
        id=id, provider=provider,
        capabilities=frozenset(caps), context_window=ctx,
    )


MERCURY = model("mercury-2", "inception", {CHAT}, 128_000)
EDIT2 = model("mercury-edit-2", "inception", {Capability.FIM, Capability.EDIT}, 128_000)
FLASH = model("gemini-2.5-flash", "gemini", {CHAT, TOOLS}, 1_048_576)
PRO = model("gemini-2.5-pro", "gemini", {CHAT, TOOLS}, 1_048_576)
SMALL = model("gemini-1.0-flash", "gemini", {CHAT, TOOLS}, 32_000)
OPUS = model("claude-opus-4-6", "anthropic", {CHAT, TOOLS, Capability.REASONING}, 200_000)
HAIKU = model("claude-haiku-4-5", "anthropic", {CHAT, TOOLS, Capability.REASONING}, 200_000)

INCEPTION_ONLY = {"inception": [MERCURY, EDIT2]}


def router(**catalogue) -> Router:
    """A Router over a fake catalogue. Absent vendor == unconfigured."""
    return Router(catalogue=FakeCatalogue(catalogue))


# ---------------------------------------------------------------------------
# The happy path and the skip path
# ---------------------------------------------------------------------------


def test_first_candidate_wins_with_no_skips():
    d = router(**INCEPTION_ONLY).resolve(Task.CHAT_FAST)
    assert d.model.id == "mercury-2"
    assert d.index == 0
    assert d.fell_back is False
    assert d.skipped == ()


def test_falls_through_to_the_next_candidate():
    """PLAN leads with a Gemini query; with no Gemini key it must fall back."""
    d = router(**INCEPTION_ONLY).resolve(Task.PLAN)
    assert d.model.id == "mercury-2"
    assert d.fell_back is True
    assert len(d.skipped) == 1
    assert "GEMINI_API_KEY" in d.skipped[0].reason


def test_a_configured_secondary_wins_when_it_leads_the_chain():
    d = router(inception=[MERCURY], gemini=[FLASH]).resolve(Task.PLAN)
    assert d.provider == "gemini"
    assert d.model.id == "gemini-2.5-flash"
    assert d.fell_back is False


def test_mercury_leads_the_chat_chains_even_when_others_are_available():
    """The cost policy, asserted. Secondaries are failover, not preference."""
    d = router(inception=[MERCURY], anthropic=[OPUS, HAIKU]).resolve(Task.REASON)
    assert d.provider == "inception"
    assert d.skipped == ()


def test_no_viable_route_names_every_skip():
    with pytest.raises(NoViableRoute) as exc:
        router().resolve(Task.CHAT_FAST)
    message = str(exc.value)
    for env in ("INCEPTION_API_KEY", "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        assert env in message, message
    assert len(exc.value.skipped) == len(TASK_ROUTES[Task.CHAT_FAST])


def test_skips_are_ordered_and_indexed():
    with pytest.raises(NoViableRoute) as exc:
        router().resolve(Task.CHAT_FAST)
    assert [s.index for s in exc.value.skipped] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_name_contains_picks_the_tier_not_the_flagship():
    """The whole reason name_contains exists: both models share a window,
    so no context rule could tell them apart."""
    assert OPUS.context_window == HAIKU.context_window
    d = router(gemini=[], anthropic=[OPUS, HAIKU]).resolve(Task.REASON)
    assert d.model.id == "claude-haiku-4-5"


def test_min_context_excludes_a_model_that_otherwise_qualifies():
    """SMALL has the right capabilities and the right name, but a 32k window,
    and PLAN demands 900k."""
    d = router(inception=[MERCURY], gemini=[SMALL]).resolve(Task.PLAN)
    assert d.provider == "inception"
    assert "gemini" in d.skipped[0].target


def test_missing_capability_is_skipped():
    no_tools = model("gemini-2.5-flash", "gemini", {CHAT}, 1_048_576)
    d = router(inception=[MERCURY], gemini=[no_tools]).resolve(Task.PLAN)
    assert d.provider == "inception"          # PLAN's gemini candidate needs TOOLS


def test_open_query_is_refused_rather_than_silently_wrong():
    r = router(**INCEPTION_ONLY)
    outcome = r._match(Candidate(requires=frozenset({CHAT})))
    assert isinstance(outcome, str)
    assert "open quer" in outcome


# ---------------------------------------------------------------------------
# _select: the tiebreak
# ---------------------------------------------------------------------------
#
# Exercised directly because no shipped route uses LARGEST_CONTEXT -- the cost
# policy makes SMALLEST the default everywhere.


@pytest.mark.parametrize("prefer, expected", [
    (Preference.SMALLEST_CONTEXT, "gemini-1.0-flash"),
    (Preference.LARGEST_CONTEXT, "gemini-2.5-flash"),
])
def test_prefer_chooses_the_end_of_the_range(prefer, expected):
    c = Candidate(provider="gemini", requires=frozenset({CHAT}), prefer=prefer)
    chosen = router()._select([FLASH, SMALL], c)
    assert chosen.id == expected


def test_equal_context_windows_break_on_id_deterministically():
    c = Candidate(provider="gemini", requires=frozenset({CHAT}))
    r = router()
    first = r._select([PRO, FLASH], c)
    second = r._select([FLASH, PRO], c)          # pool order reversed
    assert first.id == second.id, "catalogue order must not decide"


def test_no_match_returns_none():
    c = Candidate(provider="gemini", requires=frozenset({Capability.EMBEDDINGS}))
    assert router()._select([FLASH, PRO], c) is None


# ---------------------------------------------------------------------------
# The decision object
# ---------------------------------------------------------------------------


def test_params_are_copied_not_shared():
    """A caller mutating decision.params must not rewrite the routing table."""
    d = router(**INCEPTION_ONLY).resolve(Task.CHAT_FAST)
    original = dict(TASK_ROUTES[Task.CHAT_FAST][0].params)
    d.params["temperature"] = 99
    assert dict(TASK_ROUTES[Task.CHAT_FAST][0].params) == original


def test_endpoint_is_carried_through():
    d = router(**INCEPTION_ONLY).resolve(Task.CODE_COMPLETE)
    assert d.endpoint is Endpoint.FIM
    assert d.model.id == "mercury-edit-2"


def test_decision_carries_the_whole_model_not_just_an_id():
    d = router(**INCEPTION_ONLY).resolve(Task.CHAT_FAST)
    assert d.model.context_window == 128_000      # Phase 6 wants this in the trace


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate, expected", [
    (Candidate(spec="inception:mercury-2", requires=frozenset({CHAT})),
     "inception:mercury-2"),
    (Candidate(provider="anthropic", name_contains="haiku", requires=frozenset({CHAT})),
     "anthropic:*haiku*"),
    (Candidate(provider="openai", requires=frozenset({CHAT})),
     "openai:*"),
    (Candidate(requires=frozenset({CHAT})),
     "any:*"),
])
def test_render(candidate, expected):
    assert render(candidate) == expected


def test_skip_renders_on_one_line():
    assert str(Skip(2, "openai:*mini*", "no key")) == "[2] openai:*mini*: no key"


# ---------------------------------------------------------------------------
# The Catalogue contract
# ---------------------------------------------------------------------------


def test_fake_catalogue_satisfies_the_protocol():
    assert isinstance(FakeCatalogue({}), Catalogue)
    assert isinstance(RegistryCatalogue(), Catalogue)


def test_registry_catalogue_calls_is_configured_rather_than_returning_it(monkeypatch):
    """Regression guard for a bug that inverts the whole design.

    `provider_class(p).is_configured` without parens returns a bound method,
    which is always truthy -- so every provider reports configured, and the
    cheap check that exists to avoid constructing a keyless provider instead
    guarantees you always do.
    """
    calls: list[str] = []

    class FakeProvider:
        @classmethod
        def is_configured(cls) -> bool:
            calls.append("called")
            return False

    monkeypatch.setattr(router_mod, "provider_class", lambda name: FakeProvider)
    result = RegistryCatalogue().is_configured("anything")

    assert calls == ["called"], "is_configured was returned, not called"
    assert result is False
    assert isinstance(result, bool)


def test_base_is_configured_reads_the_env_without_constructing(monkeypatch):
    class Dummy(BaseProvider):
        name, env_var = "dummy", "DUMMY_KEY"
        def _build_client(self): raise AssertionError("must not construct")
        def _fetch_models(self): raise AssertionError("must not construct")
        def chat_model(self, model_id, **kw): raise AssertionError("must not construct")

    monkeypatch.delenv("DUMMY_KEY", raising=False)
    assert Dummy.is_configured() is False
    monkeypatch.setenv("DUMMY_KEY", "sk-x")
    assert Dummy.is_configured() is True
