"""Tests for resolution — Phase 2, trimmed for Inception-only routing (2026-09-09).

Nothing here needs an API key, a network connection, or a vendor SDK. That is
the point of the `Catalogue` protocol: `FakeCatalogue` is a dict, and "provider
is configured" means "provider is a key in that dict".

Fixtures use the real `TASK_ROUTES`. Every chain in the shipped table is now a
single pinned Inception candidate, so the interesting coverage moved: less
about which of several vendors wins, more about a pin resolving to the EXACT
model it names once more than one same-vendor model can satisfy it -- the gap
that mattered the day mercury-2 and mercury-2.5 were both live at once.
"""

from __future__ import annotations

import pytest
from collections import Counter
from dataclasses import dataclass, field

from agent.router import router as router_mod
from agent.router.llm_provider.base import (
    AuthError,
    BaseProvider,
    Capability,
    CapabilityNotSupported,
    Completion,
    ModelInfo,
    ProviderError,
)
from agent.router.mapping import TASK_ROUTES, Candidate, Endpoint, Preference, Task
from agent.router.router import (
    Catalogue,
    RoutingDegraded,
    FakeCatalogue,
    NoViableRoute,
    RegistryCatalogue,
    Router,
    Skip,
    render,
)

CHAT = Capability.CHAT


def model(id: str, provider: str, caps, ctx: int) -> ModelInfo:
    """Build a ModelInfo without the fields no test cares about.

    ModelInfo has seven fields; a routing test cares about three. Spelling out
    display_name and raw in every fixture buries the differences that matter.
    """
    return ModelInfo(
        id=id, provider=provider,
        capabilities=frozenset(caps), context_window=ctx,
    )


MERCURY_25 = model("mercury-2.5", "inception", {CHAT}, 260_000)
MERCURY_2 = model("mercury-2", "inception", {CHAT}, 128_000)
EDIT2 = model("mercury-edit-2", "inception", {Capability.FIM, Capability.EDIT}, 128_000)

DEFAULT_CATALOGUE = {"inception": [MERCURY_25, EDIT2]}


def catalogue(**providers) -> FakeCatalogue:
    """A fake catalogue. Absent vendor == unconfigured.

    Inception is always present because `__init__` refuses to construct
    without it. Pass `inception=[]` for "configured but offering nothing" --
    that is how you reach NoViableRoute now.
    """
    providers.setdefault("inception", [MERCURY_25, EDIT2])
    return FakeCatalogue(providers)


def router(**providers) -> Router:
    return Router(catalogue=catalogue(**providers))


# ---------------------------------------------------------------------------
# The happy path and the skip path
# ---------------------------------------------------------------------------


def test_first_candidate_wins_with_no_skips():
    d = router().resolve(Task.CHAT_FAST)
    assert d.model.id == "mercury-2.5"
    assert d.index == 0
    assert d.fell_back is False
    assert d.skipped == ()


def test_no_viable_route_names_the_skip():
    """Inception configured but offering nothing -- no secondary left to
    fall back to, so this is the only way to reach NoViableRoute now."""
    with pytest.raises(NoViableRoute) as exc:
        router(inception=[]).resolve(Task.CHAT_FAST)
    message = str(exc.value)
    assert "no model matched" in message
    assert len(exc.value.skipped) == len(TASK_ROUTES[Task.CHAT_FAST]) == 1


def test_skips_are_ordered_and_indexed():
    with pytest.raises(NoViableRoute) as exc:
        router(inception=[]).resolve(Task.CHAT_FAST)
    assert [s.index for s in exc.value.skipped] == [0]


# ---------------------------------------------------------------------------
# Pinning: a pin names one exact model, never a tier to search within
# ---------------------------------------------------------------------------
#
# This is the gap that mattered in practice: mercury-2 and mercury-2.5 are
# both CHAT-capable Inception models, both live in the catalogue at once once
# 2.5 ships. A pinned Candidate must resolve to the id it names, not to
# "whichever Inception model matches CHAT", or updating a route's spec string
# would not reliably change what actually gets called.


def test_pin_resolves_to_its_exact_id_not_whichever_matches():
    """Two CHAT-capable inception models in the pool; the pin must not pick
    the other one even though it also satisfies `requires`."""
    d = router(inception=[MERCURY_2, MERCURY_25, EDIT2]).resolve(Task.CHAT_FAST)
    assert d.model.id == "mercury-2.5"


def test_pin_ignores_context_window_ordering():
    """Before the fix, `_select` sorted matches by context window under
    `Preference.SMALLEST_CONTEXT` (the default) -- with mercury-2 (128k)
    smaller than mercury-2.5 (260k), that sort would have silently preferred
    the OLDER model. A pin must not be swayed by window size at all."""
    d = router(inception=[MERCURY_25, MERCURY_2]).resolve(Task.CHAT_FAST)
    assert d.model.id == "mercury-2.5"

    d2 = router(inception=[MERCURY_2, MERCURY_25]).resolve(Task.CHAT_FAST)
    assert d2.model.id == "mercury-2.5", "pool order must not decide either"


def test_pin_to_a_missing_id_is_a_clean_skip_not_a_wrong_match():
    """mercury-2.5 pinned, but the pool only has mercury-2 (e.g. an account
    not yet upgraded) -- must skip, never silently substitute."""
    d = router(inception=[MERCURY_2, EDIT2])
    with pytest.raises(NoViableRoute) as exc:
        d.resolve(Task.CHAT_FAST)
    assert "no model matched" in str(exc.value)


def test_pin_still_checks_requires():
    """A pinned id present in the pool but missing the required capability
    (e.g. a provider bug, or a model whose metadata is wrong) must still be
    treated as no match -- a pin does not bypass capability gating."""
    broken = model("mercury-2.5", "inception", set(), 260_000)  # no CHAT
    d = router(inception=[broken])
    with pytest.raises(NoViableRoute):
        d.resolve(Task.CHAT_FAST)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_min_context_excludes_a_model_that_otherwise_qualifies():
    c = Candidate(provider="inception", requires=frozenset({CHAT}), min_context=900_000)
    assert router()._select([MERCURY_25], c) is None       # 260k < 900k


def test_missing_capability_is_skipped():
    no_chat = model("mercury-2.5", "inception", set(), 260_000)
    c = Candidate(provider="inception", requires=frozenset({CHAT}))
    assert router()._select([no_chat], c) is None


def test_open_query_is_refused_rather_than_silently_wrong():
    r = router()
    outcome = r._match(Candidate(requires=frozenset({CHAT})))
    assert isinstance(outcome, str)
    assert "open quer" in outcome


# ---------------------------------------------------------------------------
# _select: the tiebreak for QUERY candidates
# ---------------------------------------------------------------------------
#
# No shipped route is a query anymore -- every candidate in TASK_ROUTES is a
# pin -- but the mechanism is still real code, reachable the day a route goes
# back to "whichever Mercury variant fits" instead of naming one. Exercised
# directly rather than through a route.


@pytest.mark.parametrize("prefer, expected", [
    (Preference.SMALLEST_CONTEXT, "mercury-2"),
    (Preference.LARGEST_CONTEXT, "mercury-2.5"),
])
def test_prefer_chooses_the_end_of_the_range(prefer, expected):
    c = Candidate(provider="inception", requires=frozenset({CHAT}), prefer=prefer)
    chosen = router()._select([MERCURY_2, MERCURY_25], c)
    assert chosen.id == expected


def test_equal_context_windows_break_on_id_deterministically():
    tied_a = model("mercury-2.5", "inception", {CHAT}, 260_000)
    tied_b = model("mercury-2.5-preview", "inception", {CHAT}, 260_000)
    c = Candidate(provider="inception", requires=frozenset({CHAT}))
    r = router()
    first = r._select([tied_a, tied_b], c)
    second = r._select([tied_b, tied_a], c)          # pool order reversed
    assert first.id == second.id, "catalogue order must not decide"


def test_no_match_returns_none():
    c = Candidate(provider="inception", requires=frozenset({Capability.EMBEDDINGS}))
    assert router()._select([MERCURY_25, MERCURY_2], c) is None


# ---------------------------------------------------------------------------
# The decision object
# ---------------------------------------------------------------------------


def test_params_are_copied_not_shared():
    """A caller mutating decision.params must not rewrite the routing table."""
    d = router().resolve(Task.CHAT_FAST)
    original = dict(TASK_ROUTES[Task.CHAT_FAST][0].params)
    d.params["temperature"] = 99
    assert dict(TASK_ROUTES[Task.CHAT_FAST][0].params) == original


def test_endpoint_is_carried_through():
    d = router().resolve(Task.CODE_COMPLETE)
    assert d.endpoint is Endpoint.FIM
    assert d.model.id == "mercury-edit-2"


def test_decision_carries_the_whole_model_not_just_an_id():
    d = router().resolve(Task.CHAT_FAST)
    assert d.model.context_window == 260_000      # Phase 6 wants this in the trace


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("candidate, expected", [
    (Candidate(spec="inception:mercury-2.5", requires=frozenset({CHAT})),
     "inception:mercury-2.5"),
    (Candidate(provider="inception", name_contains="edit", requires=frozenset({CHAT})),
     "inception:*edit*"),
    (Candidate(provider="inception", requires=frozenset({CHAT})),
     "inception:*"),
    (Candidate(requires=frozenset({CHAT})),
     "any:*"),
])
def test_render(candidate, expected):
    assert render(candidate) == expected


def test_skip_renders_on_one_line():
    assert str(Skip(0, "inception:mercury-2.5", "no key")) == "[0] inception:mercury-2.5: no key"


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


# ---------------------------------------------------------------------------
# Construction and policy
# ---------------------------------------------------------------------------


def test_requires_inception():
    with pytest.raises(AuthError) as exc:
        Router(catalogue=FakeCatalogue({}))
    assert "INCEPTION_API_KEY" in str(exc.value)


def test_strict_is_keyword_only():
    with pytest.raises(TypeError):
        Router(FakeCatalogue({"inception": [MERCURY_25]}), True)   # type: ignore[misc]


def test_usable_with_inception_only():
    assert router().usable() == ("inception",)


def test_every_configured_provider_is_usable():
    """Superseded policy (2026-09-11): this used to admit `REQUIRED` plus one
    `secondary`, so a third configured vendor was silently unreachable. The
    routing table now names four at once -- Anthropic judges and plans, OpenAI
    solves, Gemini reads images, Inception keeps chat-fast and the FIM/edit
    endpoints -- and under the old rule three of the four would never resolve.
    """
    r = Router(catalogue=FakeCatalogue({
        "inception": [MERCURY_25], "anthropic": [MERCURY_25], "gemini": [MERCURY_25],
    }))

    assert set(r.usable()) == {"inception", "anthropic", "gemini"}


def test_a_provider_without_a_key_is_simply_not_usable():
    """Optional in the real sense: configure a vendor and its routes resolve,
    leave it out and they are skipped with a legible reason."""
    r = Router(catalogue=FakeCatalogue({"inception": [MERCURY_25]}))

    assert r.usable() == ("inception",)


def test_inception_is_still_required():
    """It alone serves Endpoint.FIM/EDIT (mapping.py's INCEPTION_ONLY_ENDPOINTS),
    so a missing key there is a broken install, not a degraded one."""
    with pytest.raises(AuthError):
        Router(catalogue=FakeCatalogue({"anthropic": [MERCURY_25]}))


def test_a_vendor_the_registry_does_not_know_can_never_become_usable():
    """`_snapshot` walks `provider_names()` -- the real registry -- and only
    then asks the catalogue whether each is configured. So a catalogue that
    claims some other vendor cannot promote it: being keyed is not the same as
    being registered, and only the registry can add a provider."""
    r = Router(catalogue=FakeCatalogue({"inception": [MERCURY_25], "mistral": [MERCURY_25]}))

    assert "mistral" not in r.usable()


# ---------------------------------------------------------------------------
# Models and hard routes (Phase 4)
# ---------------------------------------------------------------------------
#
# `Catalogue` covers reading the catalogue, but these methods call
# get_provider() to *construct* a provider, which needs a real key. So the
# seam here is a monkeypatched get_provider.


class FakeProvider:
    """Records what the router handed it, returns sentinels."""

    def __init__(self) -> None:
        self.chat: dict = {}
        self.fim_call: dict = {}
        self.edit_call: dict = {}
        self.required: list[Capability] = []

    def require(self, capability: Capability) -> None:
        self.required.append(capability)

    def chat_model(self, model_id, **kw):
        self.chat = {"model": model_id, **kw}
        return "a-chat-model"

    def fim(self, model_id, prefix, suffix="", **kw):
        self.fim_call = {"model": model_id, "prefix": prefix, "suffix": suffix, **kw}
        return Completion(text="  return a + b", usage={"input": 9, "output": 4, "total": 13})

    def code_edit(self, model_id, code_to_edit, **kw):
        self.edit_call = {"model": model_id, "code": code_to_edit, **kw}
        return Completion(text="edited", usage={"input": 20, "output": 3, "total": 23})


@pytest.fixture
def provider(monkeypatch) -> FakeProvider:
    fake = FakeProvider()
    monkeypatch.setattr(router_mod, "get_provider", lambda name: fake)
    return fake


# --- chat -------------------------------------------------------------------


def test_chat_model_passes_the_resolved_id_and_route_params(provider):
    assert router().chat_model(Task.CHAT_FAST) == "a-chat-model"
    assert provider.chat["model"] == "mercury-2.5"
    assert provider.chat["temperature"] == 0.2
    assert provider.chat["diffusing"] is True


def test_call_site_overrides_beat_route_params(provider):
    router().chat_model(Task.CHAT_FAST, temperature=0.9)
    assert provider.chat["temperature"] == 0.9      # override won
    assert provider.chat["diffusing"] is True       # unrelated route param survived


def test_chat_model_refuses_a_non_chat_route(provider):
    with pytest.raises(CapabilityNotSupported) as exc:
        router().chat_model(Task.CODE_COMPLETE)
    assert "not chat" in str(exc.value)
    assert provider.chat == {}, "must fail before constructing anything"


def test_model_for_accepts_a_decision_made_earlier(provider):
    """Phase 6 needs this split: resolve once, trace it, then build."""
    r = router()
    d = r.resolve(Task.CHAT_FAST)
    r.model_for(d)
    assert provider.chat["model"] == d.model.id


# --- fim / edit -------------------------------------------------------------


def test_fim_forwards_prefix_and_suffix_positionally(provider):
    out = router().fim("def add(a, b):\n", "\nreturn c")
    assert out == "  return a + b"
    assert provider.fim_call["model"] == "mercury-edit-2"
    assert provider.fim_call["prefix"] == "def add(a, b):\n"
    assert provider.fim_call["suffix"] == "\nreturn c"
    assert provider.fim_call["max_tokens"] == 256          # from the route
    assert provider.required == [Capability.FIM]


def test_fim_refuses_a_non_fim_route(provider):
    with pytest.raises(CapabilityNotSupported):
        router().fim("x", task=Task.CHAT_FAST)


def test_code_edit_forwards_every_context_block(provider):
    router().code_edit(
        "print('hi')",
        current_file="def greet():\n    print('hi')",
        recently_viewed=["a.py"],
        edit_history=["- old\n+ new"],
    )
    call = provider.edit_call
    assert call["model"] == "mercury-edit-2"
    assert call["code"] == "print('hi')"
    assert call["recently_viewed"] == ["a.py"]
    assert call["edit_history"] == ["- old\n+ new"]
    assert call["temperature"] == 0.4                      # from the route
    assert provider.required == [Capability.EDIT]


def test_code_edit_has_no_instruction_parameter():
    """Next-edit prediction infers the change from context. An instruction
    argument would be an API that silently does nothing."""
    import inspect
    params = inspect.signature(Router.code_edit).parameters
    assert "instruction" not in params


# --- strict -----------------------------------------------------------------
#
# No shipped chain has a second candidate anymore, so `strict` can no longer
# be demonstrated against a real route falling back -- it is exercised
# directly against a hand-built two-candidate chain instead.


def test_strict_raises_when_the_chain_falls_back(monkeypatch):
    chain = (
        Candidate(spec="inception:mercury-2", requires=frozenset({CHAT})),
        Candidate(spec="inception:mercury-2.5", requires=frozenset({CHAT})),
    )
    monkeypatch.setitem(TASK_ROUTES, Task.CHAT_FAST, chain)
    r = Router(catalogue=catalogue(inception=[MERCURY_25]), strict=True)  # mercury-2 absent
    with pytest.raises(RoutingDegraded) as exc:
        r.resolve(Task.CHAT_FAST)
    assert exc.value.task is Task.CHAT_FAST


def test_strict_is_quiet_when_the_first_candidate_wins():
    Router(catalogue=catalogue(), strict=True).resolve(Task.CHAT_FAST)


def test_strict_reaches_every_entry_point(provider, monkeypatch):
    """The flag lives in resolve(), so chat_model inherits it rather than
    needing its own check."""
    chain = (
        Candidate(spec="inception:mercury-2", requires=frozenset({CHAT})),
        Candidate(spec="inception:mercury-2.5", requires=frozenset({CHAT})),
    )
    monkeypatch.setitem(TASK_ROUTES, Task.CHAT_FAST, chain)
    r = Router(catalogue=catalogue(inception=[MERCURY_25]), strict=True)
    with pytest.raises(RoutingDegraded):
        r.chat_model(Task.CHAT_FAST)
    assert provider.chat == {}, "must fail before constructing anything"


def test_non_strict_falls_back_silently_but_records_it(monkeypatch):
    chain = (
        Candidate(spec="inception:mercury-2", requires=frozenset({CHAT})),
        Candidate(spec="inception:mercury-2.5", requires=frozenset({CHAT})),
    )
    monkeypatch.setitem(TASK_ROUTES, Task.CHAT_FAST, chain)
    d = Router(catalogue=catalogue(inception=[MERCURY_25])).resolve(Task.CHAT_FAST)
    assert d.fell_back is True
    assert d.skipped


def test_fim_returns_text_but_carries_usage_for_the_span(provider):
    """The router's contract is `str`; the usage rides on the provider's
    Completion so the Langfuse span can be costed. A node never sees it."""
    out_text = router().fim("def add(a, b):\n")
    assert out_text == "  return a + b"
    assert isinstance(out_text, str)


def test_code_edit_returns_text_not_a_completion(provider):
    assert router().code_edit("x = 1") == "edited"


# ---------------------------------------------------------------------------
# Warm-up and reset (Phase 7)
# ---------------------------------------------------------------------------
#
# `prewarm()` and `reset()` both walk the catalogue rather than the routing
# table, so the fakes here are catalogues with extra behaviour bolted on --
# a call counter, an exception, or a `reset()` that actually mutates state.


@dataclass
class CountingCatalogue(FakeCatalogue):
    """Records every provider `models()` was actually called for."""

    calls: Counter = field(default_factory=Counter)

    def models(self, provider):
        self.calls[provider] += 1
        return super().models(provider)


@dataclass
class ExplodingCatalogue(FakeCatalogue):
    """A catalogue whose Inception entry is configured but unreachable.

    Distinct from "not configured": `is_configured` says yes, but the network
    call behind `models()` fails, the way an expired key or a vendor outage
    would. `prewarm()` exists to turn exactly this into a report instead of a
    crash.
    """

    def models(self, provider):
        raise ProviderError("inception is down")


def test_prewarm_touches_the_only_provider():
    cat = CountingCatalogue({"inception": [MERCURY_25]})
    r = Router(catalogue=cat)

    failures = r.prewarm()

    assert failures == {}
    assert cat.calls == Counter({"inception": 1})


def test_prewarm_reports_a_dead_provider_instead_of_raising():
    """A ProviderError must not stop the caller."""
    cat = ExplodingCatalogue({"inception": [MERCURY_25]})
    r = Router(catalogue=cat)

    failures = r.prewarm()

    assert failures == {"inception": "inception is down"}


def test_usable_is_the_configured_snapshot():
    r = router()
    assert r.usable() == ("inception",)


@dataclass
class VanishingCatalogue(FakeCatalogue):
    """A catalogue whose `reset()` drops the required provider.

    Models the case reset exists to guard against: the required key was
    unset (or revoked) between runs, and `reset()` must fail loudly the same
    way `__init__` does, not silently leave the router in its old, stale
    state.
    """

    def reset(self) -> None:
        self.data = {k: v for k, v in self.data.items() if k != "inception"}


def test_reset_raises_when_inception_disappears():
    cat = VanishingCatalogue({"inception": [MERCURY_25]})
    r = Router(catalogue=cat)

    with pytest.raises(AuthError):
        r.reset()
