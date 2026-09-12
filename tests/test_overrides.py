"""~/.otto/routes.json and how it lands in the live table
(agent/router/overrides.py). Offline; the file is a tmp_path."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.router import overrides as ov
from agent.router import router as router_mod
from agent.router.llm_provider import unregister_custom
from agent.router.llm_provider.base import Capability, ModelInfo
from agent.router.mapping import PARAMS_BY_PROVIDER, TASK_ROUTES, Candidate, Endpoint, Task
from agent.router.router import FakeCatalogue, Router

CHAT = Capability.CHAT


def model(id, provider, caps, ctx=100_000):
    return ModelInfo(id=id, provider=provider, capabilities=frozenset(caps), context_window=ctx)


MERCURY = model("mercury-2.5", "inception", {CHAT})
EDIT2 = model("mercury-edit-2", "inception", {Capability.FIM, Capability.EDIT})
GPT = model("gpt-5-mini", "openai", {CHAT, Capability.REASONING, Capability.VISION})
HAIKU = model("claude-haiku-4-5-20251001", "anthropic", {CHAT, Capability.REASONING, Capability.TOOLS})


@pytest.fixture
def routes_file(tmp_path):
    path = tmp_path / "routes.json"
    with ov.bind_routes(path):
        yield path


@pytest.fixture
def live_table():
    """Let a test apply pins to the REAL table, and put everything back."""
    before = dict(TASK_ROUTES)
    active = dict(ov._ACTIVE)
    yield TASK_ROUTES
    ov._register_endpoints({})
    for task, chain in before.items():
        TASK_ROUTES[task] = chain
    ov._ACTIVE.clear()
    ov._ACTIVE.update(active)
    ov._PROBLEMS.clear()


# ---- load / save ---------------------------------------------------------

def test_a_missing_file_is_empty(routes_file):
    assert ov.load() == ov.Overrides()


@pytest.mark.parametrize("content", ["not json", "[]", '{"version": 99}', '{"pins": 3}'])
def test_a_bad_file_is_ignored_not_fatal(routes_file, content):
    routes_file.write_text(content)
    assert ov.load() == ov.Overrides()


def test_save_then_load_round_trips(routes_file):
    o = ov.Overrides(
        pins={Task.REASON: "openai:gpt-5-mini", Task.CHAT_FAST: "ollama:llama3.2:latest"},
        endpoints={"ollama": ov.EndpointSpec("Ollama", {"llama3.2:latest": ("chat", "tools")})},
    )
    ov.save(o)
    assert ov.load() == o
    raw = json.loads(routes_file.read_text())
    assert raw["version"] == 1 and raw["pins"]["reason"] == "openai:gpt-5-mini"


def test_unknown_tasks_and_bad_endpoint_names_are_skipped(routes_file):
    routes_file.write_text(json.dumps({
        "version": 1, "pins": {"reason": "openai:x", "turbo": "a:b"},
        "endpoints": {"Bad Name": {}, "fine": {"label": "ok"}},
    }))
    o = ov.load()
    assert o.pins == {Task.REASON: "openai:x"}
    assert set(o.endpoints) == {"fine"}


# ---- apply ---------------------------------------------------------------

def test_a_pin_leads_the_chain_and_the_shipped_chain_follows(live_table, routes_file):
    problems = ov.apply(o=ov.Overrides(pins={Task.CHAT_FAST: "openai:gpt-5-mini"}))
    assert problems == []
    chain = TASK_ROUTES[Task.CHAT_FAST]
    assert chain[0].spec == "openai:gpt-5-mini"
    assert chain[1:] == ov.shipped(Task.CHAT_FAST)
    assert ov.is_pin(Task.CHAT_FAST, chain[0])
    assert ov.pinned_spec(Task.CHAT_FAST) == "openai:gpt-5-mini"
    assert chain[0].requires == frozenset({CHAT}), "a heuristic REASONING tag does not gate a pin"


def test_same_provider_params_are_carried_over(live_table, routes_file):
    ov.apply(o=ov.Overrides(pins={Task.REASON: "inception:mercury-2.5"}))
    pin = TASK_ROUTES[Task.REASON][0]
    inception_shipped = next(c for c in ov.shipped(Task.REASON) if c.provider_name == "inception")
    assert pin.params == dict(inception_shipped.params)
    assert "reasoning_effort" in pin.params
    # The shipped inception candidate is not repeated behind its own pin.
    assert [c.spec for c in TASK_ROUTES[Task.REASON]].count("inception:mercury-2.5") == 1


def test_foreign_provider_params_are_filtered(live_table, routes_file):
    ov.apply(o=ov.Overrides(pins={Task.CHAT_FAST: "anthropic:claude-haiku-4-5-20251001"}))
    pin = TASK_ROUTES[Task.CHAT_FAST][0]
    assert "diffusing" not in pin.params and "reasoning_effort" not in pin.params
    assert set(pin.params) <= PARAMS_BY_PROVIDER["anthropic"][Endpoint.CHAT]


def test_a_bad_pin_is_reported_and_the_task_keeps_its_shipped_chain(live_table, routes_file):
    problems = ov.apply(o=ov.Overrides(pins={
        Task.CHAT_FAST: "nowhere:model",
        Task.REASON: "openai:gpt-5-mini",
    }))
    assert len(problems) == 1 and "chat_fast" in problems[0] and "unknown provider" in problems[0]
    assert TASK_ROUTES[Task.CHAT_FAST] == ov.shipped(Task.CHAT_FAST)
    assert TASK_ROUTES[Task.REASON][0].spec == "openai:gpt-5-mini"
    assert ov.last_problems() == problems


@pytest.mark.parametrize("task, spec, why", [
    (Task.CODE_COMPLETE, "openai:gpt-5-mini", "inception"),
    (Task.CODE_EDIT, "anthropic:x", "inception"),
    (Task.WEB, "openai:gpt-5-mini", "anthropic"),
    (Task.REASON, "gpt-5-mini", "provider:model"),
    (Task.REASON, "openai:", "provider:model"),
])
def test_validate_pin_refuses_what_cannot_work(task, spec, why):
    with pytest.raises(ov.PinError, match=why):
        ov.validate_pin(task, spec)


def test_validate_pin_accepts_an_ollama_style_id(live_table, routes_file):
    ov.apply(o=ov.Overrides(endpoints={"ollama": ov.EndpointSpec("Ollama")}))
    ov.validate_pin(Task.CHAT_FAST, "ollama:llama3.2:latest")
    problems = ov.apply(o=ov.Overrides(
        pins={Task.CHAT_FAST: "ollama:llama3.2:latest"},
        endpoints={"ollama": ov.EndpointSpec("Ollama")},
    ))
    assert problems == []
    assert TASK_ROUTES[Task.CHAT_FAST][0].spec == "ollama:llama3.2:latest"


def test_a_fim_pin_keeps_the_fim_requirement(live_table, routes_file):
    ov.apply(o=ov.Overrides(pins={Task.CODE_COMPLETE: "inception:mercury-edit-2"}))
    pin = TASK_ROUTES[Task.CODE_COMPLETE][0]
    assert pin.endpoint is Endpoint.FIM and Capability.FIM in pin.requires


def test_a_vision_pin_keeps_the_vision_requirement(live_table, routes_file):
    ov.apply(o=ov.Overrides(pins={Task.VISION: "openai:gpt-5-mini"}))
    assert Capability.VISION in TASK_ROUTES[Task.VISION][0].requires


def test_apply_is_idempotent_and_clearing_restores_the_exact_tuple(live_table, routes_file):
    o = ov.Overrides(pins={Task.REASON: "openai:gpt-5-mini"})
    ov.apply(o=o)
    ov.apply(o=o)
    assert [c.spec for c in TASK_ROUTES[Task.REASON]].count("openai:gpt-5-mini") == 1
    ov.apply(o=ov.Overrides())
    assert TASK_ROUTES[Task.REASON] is ov.shipped(Task.REASON)
    assert ov.pinned_spec(Task.REASON) is None


def test_apply_on_a_copy_leaves_the_live_table_alone(routes_file):
    copy = dict(TASK_ROUTES)
    ov.apply(copy, ov.Overrides(pins={Task.CHAT_FAST: "openai:gpt-5-mini"}), shipped_routes=TASK_ROUTES)
    assert copy[Task.CHAT_FAST][0].spec == "openai:gpt-5-mini"
    assert TASK_ROUTES[Task.CHAT_FAST][0].spec != "openai:gpt-5-mini"
    ov._ACTIVE.clear()


# ---- the router honours it -------------------------------------------------

def test_the_router_resolves_a_pin_first_and_never_reorders_it(live_table, routes_file, monkeypatch):
    ov.apply(o=ov.Overrides(pins={Task.CHAT_FAST: "openai:gpt-5-mini"}))
    calls: list = []
    real = router_mod.seat_outcomes.reorder
    monkeypatch.setattr(router_mod.seat_outcomes, "reorder",
                        lambda task, chain, *a, **k: calls.append(task) or real(task, chain, *a, **k))
    r = Router(catalogue=FakeCatalogue({"inception": [MERCURY], "openai": [GPT]}))

    d = r.resolve(Task.CHAT_FAST)

    assert (d.provider, d.model.id, d.index, d.skipped) == ("openai", "gpt-5-mini", 0, ())
    assert calls == [], "a pinned chain is not handed to evidence reordering"
    r.resolve(Task.REASON)
    assert calls == ["reason"], "an unpinned one still is"


def test_a_pin_whose_provider_has_no_key_falls_back_legibly(live_table, routes_file):
    ov.apply(o=ov.Overrides(pins={Task.CHAT_FAST: "openai:gpt-5-mini"}))
    r = Router(catalogue=FakeCatalogue({"inception": [MERCURY]}))
    d = r.resolve(Task.CHAT_FAST)
    assert d.model.id == "mercury-2.5" and d.fell_back
    assert "OPENAI_API_KEY" in d.skipped[0].reason


# ---- edits ---------------------------------------------------------------

def test_set_and_clear_pin_persist_and_apply(live_table, routes_file):
    ov.set_pin(Task.REASON, "openai:gpt-5-mini")
    assert ov.pins() == {Task.REASON: "openai:gpt-5-mini"}
    assert TASK_ROUTES[Task.REASON][0].spec == "openai:gpt-5-mini"
    ov.clear_pin(Task.REASON)
    assert ov.pins() == {}
    assert TASK_ROUTES[Task.REASON] is ov.shipped(Task.REASON)


def test_set_pin_refuses_and_leaves_the_file_untouched(live_table, routes_file):
    with pytest.raises(ov.PinError):
        ov.set_pin(Task.WEB, "openai:gpt-5-mini")
    assert not routes_file.exists()


def test_endpoints_are_registered_and_forgotten(live_table, routes_file):
    from agent.router.llm_provider import provider_names
    ov.add_endpoint("openrouter", "OpenRouter")
    assert "openrouter" in provider_names()
    ov.set_pin(Task.EVALUATE, "openrouter:anthropic/claude-sonnet-4")
    ov.remove_endpoint("openrouter")
    assert "openrouter" not in provider_names()
    assert ov.pins() == {}, "pins on a removed endpoint go with it"


def test_set_capabilities_is_stored_per_model(live_table, routes_file):
    ov.add_endpoint("ollama")
    ov.set_capabilities("ollama", "llava", [Capability.CHAT, "vision"])
    assert ov.load().endpoints["ollama"].capabilities == {"llava": ("chat", "vision")}


def test_ignore_env_makes_startup_a_no_op(live_table, routes_file, monkeypatch):
    ov.save(ov.Overrides(pins={Task.REASON: "openai:gpt-5-mini"}))
    monkeypatch.setenv(ov.IGNORE_ENV, "1")
    ov.apply_at_startup()
    assert TASK_ROUTES[Task.REASON] is ov.shipped(Task.REASON)
    monkeypatch.delenv(ov.IGNORE_ENV)
    ov.apply_at_startup()
    assert TASK_ROUTES[Task.REASON][0].spec == "openai:gpt-5-mini"


def test_import_needs_no_credentials_or_file():
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env[ov.PATH_ENV] = "/nonexistent/otto/routes.json"
    result = subprocess.run(
        [sys.executable, "-c",
         "import agent.router.overrides, agent.router.automap; agent.router.overrides.apply_at_startup()"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
