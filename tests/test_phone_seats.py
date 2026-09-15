"""A seat a host binds for one run (agent/router/overrides.py bind_seats):
the phone's fast judge, and nothing else moving. Offline -- the router reads a
FakeCatalogue, the pipeline is a fake generator."""
from __future__ import annotations

import logging

import pytest

from agent import embed
from agent.phone import PHONE_SEATS
from agent.pipeline import run as pipeline
from agent.pipeline.toolkit import ExtraTool
from agent.pipeline.tools import ToolResult
from agent.router import overrides as ov
from agent.router import router as router_mod
from agent.router.llm_provider.base import Capability, ModelInfo
from agent.router.mapping import TASK_ROUTES, Task
from agent.router.router import FakeCatalogue, Router
from tests.test_embed import _own_environment, configured  # noqa: F401 -- fixtures
from tests.test_serve import _hello, _recv_until, server  # noqa: F401 -- fixture

CHAT = Capability.CHAT


def model(id, provider, caps):
    return ModelInfo(id=id, provider=provider, capabilities=frozenset(caps), context_window=200_000)


MERCURY = model("mercury-2.5", "inception", {CHAT})
FLASH = model("gemini-3.8-flash", "gemini", {CHAT, Capability.TOOLS, Capability.VISION})
ABLE = {CHAT, Capability.REASONING, Capability.TOOLS, Capability.VISION}


def everyone(*, without: str = "") -> FakeCatalogue:
    """Every model the shipped table (and an opus pin) names, per vendor."""
    specs = {c.spec for chain in ov._SHIPPED.values() for c in chain if c.spec}
    specs |= {"anthropic:claude-opus-4-6", "gemini:gemini-3.8-flash"}
    data: dict = {}
    for spec in sorted(specs):
        provider, _, model_id = spec.partition(":")
        data.setdefault(provider, []).append(model(model_id, provider, ABLE))
    data.setdefault("inception", []).append(MERCURY)
    data.pop(without, None)
    return FakeCatalogue(data)


@pytest.fixture(autouse=True)
def _shipped_table(monkeypatch):
    """The shipped table, with this machine's routes.json pins and recorded
    seat outcomes out of the way, and everything put back after."""
    before, active = dict(TASK_ROUTES), dict(ov._ACTIVE)
    TASK_ROUTES.update(ov._SHIPPED)
    ov._ACTIVE.clear()
    monkeypatch.setattr(router_mod.seat_outcomes, "reorder", lambda task, chain, *a, **k: list(chain))
    yield
    TASK_ROUTES.update(before)
    ov._ACTIVE.clear()
    ov._ACTIVE.update(active)


def _seen(d):
    return d.provider, d.model.id, d.index


def test_a_bound_evaluate_seat_leads_and_reason_does_not_move():
    r = Router(catalogue=everyone())
    reason_before = _seen(r.resolve(Task.REASON))

    with ov.bind_seats(PHONE_SEATS) as bound:
        d = r.resolve(Task.EVALUATE)
        reason_during = _seen(r.resolve(Task.REASON))

    assert bound == {Task.EVALUATE: "gemini:gemini-3.8-flash"}
    assert (d.provider, d.model.id, d.index, d.skipped) == ("gemini", "gemini-3.8-flash", 0, ())
    assert reason_during == reason_before


def test_unbound_evaluate_resolves_the_live_chain():
    r = Router(catalogue=everyone())
    d = r.resolve(Task.EVALUATE)
    assert d.model.id == TASK_ROUTES[Task.EVALUATE][0].spec.partition(":")[2]
    assert ov.bound_chain(Task.EVALUATE) is None


def test_the_live_chain_is_kept_behind_the_seat_as_the_fallback():
    with ov.bind_seats(PHONE_SEATS):
        chain = ov.bound_chain(Task.EVALUATE)
    assert chain[0].spec == "gemini:gemini-3.8-flash"
    assert chain[1:] == tuple(c for c in TASK_ROUTES[Task.EVALUATE] if c.spec != chain[0].spec)


def test_a_seat_leads_even_a_pinned_chain_and_the_pin_is_its_fallback():
    ov.apply(o=ov.Overrides(pins={Task.EVALUATE: "anthropic:claude-opus-4-6"}))
    r = Router(catalogue=everyone())
    assert r.resolve(Task.EVALUATE).model.id == "claude-opus-4-6"
    with ov.bind_seats(PHONE_SEATS):
        assert r.resolve(Task.EVALUATE).model.id == "gemini-3.8-flash"
        assert ov.bound_chain(Task.EVALUATE)[1].spec == "anthropic:claude-opus-4-6"
    r = Router(catalogue=everyone(without="gemini"))
    with ov.bind_seats(PHONE_SEATS):
        assert r.resolve(Task.EVALUATE).model.id == "claude-opus-4-6"


def test_a_seat_whose_provider_has_no_key_falls_back_legibly():
    r = Router(catalogue=everyone(without="gemini"))
    with ov.bind_seats(PHONE_SEATS):
        d = r.resolve(Task.EVALUATE)
    assert d.provider != "gemini" and d.fell_back
    assert d.skipped[0].target == "gemini:gemini-3.8-flash"
    assert "GEMINI_API_KEY" in d.skipped[0].reason


def test_a_bare_model_id_is_warned_about_and_not_bound(caplog):
    with caplog.at_level(logging.WARNING, logger=ov.__name__):
        with ov.bind_seats({"evaluate": "gemini-3.8-flash"}) as bound:
            assert bound == {}
            assert ov.bound_chain(Task.EVALUATE) is None
    assert any("gemini-3.8-flash" in rec.getMessage() for rec in caplog.records)


def test_an_unknown_task_is_skipped_not_raised(caplog):
    with caplog.at_level(logging.WARNING, logger=ov.__name__):
        with ov.bind_seats({"judging": "gemini:gemini-3.8-flash"}) as bound:
            assert bound == {}


def test_the_binding_does_not_outlive_its_block():
    with ov.bind_seats(PHONE_SEATS):
        seat = ov.bound_chain(Task.EVALUATE)[0]
        assert ov.is_bound(Task.EVALUATE, seat)
    assert ov.bound_chain(Task.EVALUATE) is None
    assert not ov.is_bound(Task.EVALUATE, seat)


def test_a_bound_seat_is_never_handed_to_evidence_reordering(monkeypatch):
    calls: list = []
    monkeypatch.setattr(router_mod.seat_outcomes, "reorder",
                        lambda task, chain, *a, **k: calls.append(task) or list(chain))
    r = Router(catalogue=everyone())
    with ov.bind_seats(PHONE_SEATS):
        r.resolve(Task.EVALUATE)
        assert calls == []
        r.resolve(Task.REASON)
    assert calls == ["reason"]


# ---- who binds it ----------------------------------------------------------

def _tool(name):
    return ExtraTool(name=name, description="x", call=lambda body: ToolResult("", "", 0), mutates=False)


def _observing(monkeypatch):
    seen: dict = {}

    def fake_run(text, **kwargs):
        chain = ov.bound_chain(Task.EVALUATE)
        seen["evaluate"] = chain[0].spec if chain else None
        yield {"__final__": {"final_output": "done"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    return seen


def test_a_turn_with_the_phone_tools_binds_the_phone_seats(configured, monkeypatch):  # noqa: F811
    seen = _observing(monkeypatch)
    handle = embed.Runtime().open_session()
    handle.run("add milk", events=lambda e: None, tools=[_tool("phone_screen"), _tool("phone_act")])
    assert seen["evaluate"] == "gemini:gemini-3.8-flash"
    assert ov.bound_chain(Task.EVALUATE) is None
    handle.close()


def test_a_turn_without_them_binds_nothing(configured, monkeypatch):  # noqa: F811
    seen = _observing(monkeypatch)
    handle = embed.Runtime().open_session()
    handle.run("fix the test", events=lambda e: None, tools=[_tool("export_pdf")])
    assert seen["evaluate"] is None
    handle.close()


def test_an_explicit_empty_mapping_wins_over_the_phone_default(configured, monkeypatch):  # noqa: F811
    seen = _observing(monkeypatch)
    handle = embed.Runtime().open_session()
    handle.run("add milk", events=lambda e: None, tools=[_tool("phone_screen")], seats={})
    assert seen["evaluate"] is None
    handle.close()


def test_the_server_passes_the_phone_seats(server, monkeypatch):  # noqa: F811
    from agent.server import protocol

    got: dict = {}
    real = embed.SessionHandle.run

    def spy(self, text, **kwargs):
        got.update(kwargs)
        return real(self, text, **kwargs)

    monkeypatch.setattr(embed.SessionHandle, "run", spy)
    seen = _observing(monkeypatch)

    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="add milk"))
    final, _ = _recv_until(ws, "event")
    ws.close()

    assert final["event"]["type"] == "final"
    assert got["seats"] == PHONE_SEATS
    assert seen["evaluate"] == "gemini:gemini-3.8-flash"
