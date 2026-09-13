"""The setup wizard (agent/cli/setup_screen.py) and the quick pin
(agent/cli/modals.py ModelPinDialog), driven headlessly against fakes
monkeypatched into agent.cli.tui's namespace. No network, no .env, no
~/.otto: every effect is recorded on a fake and asserted."""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static

from agent.cli import tui as tui_mod
from agent.cli.setup_screen import MappingRow, SetupScreen, pin_options
from agent.router.llm_provider.base import Capability, HealthReport, ModelInfo, ProviderStatus
from agent.router.mapping import TASK_ROUTES, Task
from agent.router.router import NoViableRoute
from agent.router.setup import ProbeResult, VendorRow
from tests.test_tui import _async_test, _make_app, _texts, _until

CHAT, R, V, T = Capability.CHAT, Capability.REASONING, Capability.VISION, Capability.TOOLS


def _m(id, provider, caps, ctx=100_000):
    return ModelInfo(id=id, provider=provider, capabilities=frozenset(caps), context_window=ctx)


MERCURY = _m("mercury-2.5", "inception", {CHAT})
GPT = _m("gpt-5-mini", "openai", {CHAT, R, V, T})
MODELS = [MERCURY, GPT]


def _rows(env):
    def row(name, label, custom=False):
        key = f"{name.upper()}_API_KEY"
        value = env.get(key)
        return VendorRow(name=name, label=label, key_var=key,
                         url_var=f"{name.upper()}_BASE_URL" if custom else None,
                         key_present=bool(value), url_present=bool(env.get(f"{name.upper()}_BASE_URL")),
                         custom=custom, masked_key=("********" + value[-4:]) if value else "not set")
    rows = [row("inception", "Inception (required)"), row("openai", "OpenAI")]
    rows += [row(n, n, custom=True) for n in env.get("_customs", [])]
    return rows


class FakeOverrides:
    def __init__(self):
        self.set_pin_calls: list = []
        self.clear_pin_calls: list = []
        self._pins: dict = {}

    def pins(self):
        return dict(self._pins)

    def set_pin(self, task, spec):
        self.set_pin_calls.append((task, spec))
        self._pins[task] = spec

    def clear_pin(self, task):
        self.clear_pin_calls.append(task)
        self._pins.pop(task, None)

    def is_pin(self, task, c):
        return False

    def pinned_spec(self, task):
        return self._pins.get(task)

    def active_pins(self):
        return dict(self._pins)


class Proposal(SimpleNamespace):
    @property
    def spec(self):
        return self.model.spec if self.model else None

    @property
    def needs_pin(self):
        return self.model is not None and self.source != "shipped"


@pytest.fixture
def setup_env(monkeypatch):
    """Everything the screen can reach, as fakes that record."""
    env = {"INCEPTION_API_KEY": "inc-1234", "_customs": []}
    calls = SimpleNamespace(set_key=[], set_base_url=[], add_endpoint=[], reload=0, probes=[])
    gate = SimpleNamespace(entered=threading.Event(), release=threading.Event(), block=False)

    def probe(name):
        calls.probes.append(name)
        if gate.block:
            gate.entered.set()
            gate.release.wait(5)
        if env.get(f"{name.upper()}_API_KEY"):
            return ProbeResult(HealthReport(name, ProviderStatus.OK, model_count=1), [m for m in MODELS if m.provider == name])
        return ProbeResult(HealthReport(name, ProviderStatus.NO_KEY, detail="not set"), [])

    def set_key(name, value):
        calls.set_key.append((name, value))
        env[f"{name.upper()}_API_KEY"] = value
        return "********" + value[-4:]

    def set_base_url(name, url):
        calls.set_base_url.append((name, url))
        env[f"{name.upper()}_BASE_URL"] = url

    def add_endpoint(name, label):
        calls.add_endpoint.append((name, label))
        env["_customs"].append(name)

    fake_setup = SimpleNamespace(
        vendor_rows=lambda: _rows(env), probe=probe,
        detected_pool=lambda: [m for m in MODELS if env.get(f"{m.provider.upper()}_API_KEY")],
        set_key=set_key, set_base_url=set_base_url, add_endpoint=add_endpoint,
    )
    overrides = FakeOverrides()

    def propose(pool):
        return {
            Task.REASON: Proposal(task=Task.REASON, model=GPT if GPT in pool else None,
                                  source="pool" if GPT in pool else "none",
                                  reason="only reasoning model" if GPT in pool else "no reasoning model"),
            Task.CHAT_FAST: Proposal(task=Task.CHAT_FAST, model=MERCURY, source="shipped", reason="shipped"),
        }

    def reload():
        calls.reload += 1
        return []

    monkeypatch.setattr(tui_mod, "provider_setup", fake_setup)
    monkeypatch.setattr(tui_mod, "route_overrides", overrides)
    monkeypatch.setattr(tui_mod, "propose", propose)
    monkeypatch.setattr(tui_mod, "reload_everything", reload)
    monkeypatch.setattr(tui_mod, "configured_providers", lambda: ("inception",))
    monkeypatch.setattr(tui_mod, "all_models", lambda cap=None: list(MODELS))
    return SimpleNamespace(env=env, calls=calls, gate=gate, overrides=overrides)


class _Decision(SimpleNamespace):
    pass


def _app(monkeypatch, tmp_path):
    app = _make_app(monkeypatch, tmp_path, lambda *a, **k: iter(()))

    def resolve(task):
        if task is Task.VISION:
            raise NoViableRoute(task, ())
        return _Decision(provider="inception", model=MERCURY, fell_back=False, index=0)

    app.ctx.router.resolve = resolve
    return app


async def _open(app, pilot):
    app.action_setup()
    await _until(pilot, lambda: isinstance(app.screen, SetupScreen), "the setup screen")
    await pilot.pause()
    return app.screen


def _cell(screen, row, col):
    return screen.query_one("#provider-table", DataTable).get_cell(row, col).plain


def _status(screen) -> str:
    return screen.query_one("#setup-status", Static).content.plain


# ---- opening ---------------------------------------------------------------

@_async_test
async def test_setup_opens_from_its_action_and_is_refused_mid_turn(setup_env, monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    def fake_stream(text, **kwargs):
        started.set()
        release.wait(5)
        yield {"__final__": {"final_output": "x"}, "__trace_id__": "t"}

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test(size=(120, 40)) as pilot:
        app.message_box.value = "go"
        await pilot.press("enter")
        await asyncio.to_thread(started.wait, 5)
        await pilot.pause()
        app.action_setup()
        await pilot.pause()
        assert not isinstance(app.screen, SetupScreen)
        assert any("a turn is still running" in t for t in _texts(app))
        release.set()
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        await _open(app, pilot)


@_async_test
async def test_setup_auto_opens_when_nothing_is_configured(setup_env, monkeypatch, tmp_path):
    monkeypatch.setattr(tui_mod, "configured_providers", lambda: ())
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _until(pilot, lambda: isinstance(app.screen, SetupScreen), "setup on a keyless launch")

    monkeypatch.setattr(tui_mod, "configured_providers", lambda: ("inception",))
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert not isinstance(app.screen, SetupScreen)


# ---- keys ------------------------------------------------------------------

@_async_test
async def test_a_pasted_key_is_written_and_never_shown(setup_env, monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        assert _cell(screen, "inception", "key") == "********1234"
        assert _cell(screen, "openai", "key") == "not set"
        screen.query_one("#key-provider", Select).value = "openai"
        box = screen.query_one("#key-input", Input)
        assert box.password is True
        box.focus()
        box.value = "sk-verysecret9876"
        await pilot.press("enter")
        await _until(pilot, lambda: setup_env.calls.set_key == [("openai", "sk-verysecret9876")], "the save")
        assert box.value == ""
        await _until(pilot, lambda: _cell(screen, "openai", "key") == "********9876", "the mask")
        await _until(pilot, lambda: _cell(screen, "openai", "status") == "ok", "the probe after a save")
        everything = " ".join(_texts(app)) + _status(screen) + " ".join(
            screen.query_one("#provider-table", DataTable).get_row("openai")[i].plain for i in range(5))
        assert "verysecret" not in everything
        assert "OPENAI_API_KEY" in _status(screen)
        await pilot.press("escape")
        await pilot.pause()
    assert "verysecret" not in " ".join(_texts(app))


@_async_test
async def test_probing_runs_off_the_ui_thread_and_updates_rows_live(setup_env, monkeypatch, tmp_path):
    setup_env.gate.block = True
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        await pilot.click("#probe")
        await asyncio.to_thread(setup_env.gate.entered.wait, 5)
        assert _cell(screen, "inception", "status") in ("probing…", "⠋")
        app._post("ui still alive")
        await pilot.pause()
        assert "ui still alive" in _texts(app)
        setup_env.gate.release.set()
        await _until(pilot, lambda: _cell(screen, "inception", "status") == "ok", "the row to update")
        await _until(pilot, lambda: _cell(screen, "openai", "status") == "no key", "the other row")
        assert _cell(screen, "inception", "models") == "1"
        await _until(pilot, lambda: "probe finished" in _status(screen), "the summary")


@_async_test
async def test_a_custom_endpoint_writes_both_variables_and_gets_a_row(setup_env, monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        screen.query_one("#key-provider", Select).value = "__custom__"
        await pilot.pause()
        assert screen.query_one("#custom").has_class("shown")
        screen.query_one("#custom-name", Input).value = "Local"
        screen.query_one("#custom-url", Input).value = "http://localhost:1234/v1"
        key = screen.query_one("#custom-key", Input)
        key.focus()
        key.value = "x"
        await pilot.press("enter")
        await _until(pilot, lambda: setup_env.calls.set_key == [("local", "x")], "the key save")
        assert setup_env.calls.add_endpoint == [("local", "local")]
        assert setup_env.calls.set_base_url == [("local", "http://localhost:1234/v1")]
        assert key.value == ""
        await _until(pilot, lambda: "local" in screen.query_one("#provider-table", DataTable).rows, "a row")
        assert "LOCAL_API_KEY" in _status(screen)


# ---- models and mapping ---------------------------------------------------

@_async_test
async def test_models_tab_and_pin_selects_follow_the_probe(setup_env, monkeypatch, tmp_path):
    setup_env.env["OPENAI_API_KEY"] = "sk-1"
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        await pilot.click("#probe")
        await _until(pilot, lambda: screen.query_one("#model-table", DataTable).row_count == 2, "the models")
        row = screen.query_one("#model-table", DataTable).get_row("openai:gpt-5-mini")
        assert [str(c) for c in row][:2] == ["openai", "gpt-5-mini"]
        assert "reasoning" in str(row[4])

        await _until(pilot, lambda: "not probed" not in screen.query_one("#row-reason", MappingRow)
                     .query_one(".current", Static).content.plain, "mapping filled")
        reason = screen.query_one("#row-reason", MappingRow)
        values = [v for _, v in reason.select._options]
        assert values == ["", "openai:gpt-5-mini"], "the chat-only model is not offered for REASON"
        summarize = screen.query_one("#row-summarize", MappingRow)
        assert "inception:mercury-2.5" in [v for _, v in summarize.select._options]
        vision = screen.query_one("#row-vision", MappingRow)
        assert vision.query_one(".current", Static).content.plain == "none"
        assert "only reasoning model" in reason.query_one(".proposal", Static).content.plain


@_async_test
async def test_apply_automap_then_save_pins_and_reloads(setup_env, monkeypatch, tmp_path):
    setup_env.env["OPENAI_API_KEY"] = "sk-1"
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        await pilot.click("#probe")
        await _until(pilot, lambda: screen.query_one("#model-table", DataTable).row_count == 2, "the probe")
        await _until(pilot, lambda: "probe finished" in _status(screen), "the mapping")
        screen.query_one("#tabs").active = "mapping"
        await pilot.pause()
        screen.query_one("#automap", Button).press()   # below the fold headless; press, not click
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one("#row-reason", MappingRow).select.value == "openai:gpt-5-mini"
        assert screen.query_one("#row-chat_fast", MappingRow).select.value == "", "a shipped head needs no pin"
        screen.query_one("#save", Button).press()
        await _until(pilot, lambda: not isinstance(app.screen, SetupScreen), "the screen to close")
        assert setup_env.overrides.set_pin_calls == [(Task.REASON, "openai:gpt-5-mini")]
        assert setup_env.calls.reload >= 1
        await _until(pilot, lambda: any("pinned reason -> openai:gpt-5-mini" in t for t in _texts(app)),
                     "the summary line")
        assert any("providers ok: inception, openai" in t for t in _texts(app))


@_async_test
async def test_escape_discards_unsaved_pins_on_the_second_press(setup_env, monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _open(app, pilot)
        await pilot.click("#probe")
        await _until(pilot, lambda: "probe finished" in _status(screen), "the probe")
        row = screen.query_one("#row-summarize", MappingRow)
        row.select.value = "inception:mercury-2.5"
        await pilot.pause()
        assert screen._pending == {Task.SUMMARIZE: "inception:mercury-2.5"}
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen) and "unsaved" in _status(screen)
        await pilot.press("escape")
        await _until(pilot, lambda: not isinstance(app.screen, SetupScreen), "closed")
        assert setup_env.overrides.set_pin_calls == []
        assert not any("setup:" in t for t in _texts(app))


@_async_test
async def test_setup_inputs_do_not_start_a_turn(setup_env, tmp_path):
    escaped: list[str] = []

    class Host(App):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            escaped.append(event.value)

    from agent.cli.setup_screen import SetupBackend
    backend = SetupBackend(
        vendor_rows=lambda: _rows(setup_env.env), probe=lambda n: None, detected_pool=lambda: [],
        resolve=lambda t: None, routes=lambda: TASK_ROUTES, propose=lambda pool: {},
        pins=lambda: {}, set_pin=lambda t, s: None, clear_pin=lambda t: None,
        set_key=lambda n, v: None, set_base_url=lambda n, u: None, add_endpoint=lambda n, l: None,
        reload=lambda: [],
    )
    app = Host()
    async with app.run_test(size=(120, 40)) as pilot:
        app.push_screen(SetupScreen(backend))
        await pilot.pause()
        await pilot.pause()
        for iid in ("#key-input", "#custom-key", "#custom-url"):
            box = app.screen.query_one(iid, Input)
            box.focus()
            box.value = "sk-secret"
            await pilot.press("enter")
            await pilot.pause()
    assert escaped == []


@_async_test
async def test_a_probe_finishing_after_the_screen_closed_does_not_kill_the_app(setup_env, monkeypatch, tmp_path):
    setup_env.gate.block = True
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        await _open(app, pilot)
        await pilot.click("#probe")
        await asyncio.to_thread(setup_env.gate.entered.wait, 5)
        await pilot.press("escape")
        await _until(pilot, lambda: not isinstance(app.screen, SetupScreen), "closed")
        setup_env.gate.release.set()
        await _until(pilot, lambda: not any(w.is_running for w in app.workers), "the worker to finish")
        app._post("alive")
        await pilot.pause()
        assert "alive" in _texts(app)


# ---- the quick pin ---------------------------------------------------------

@_async_test
async def test_pinning_a_model_is_two_picks(setup_env, monkeypatch, tmp_path):
    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        app.action_pin_model()
        await _until(pilot, lambda: isinstance(app.screen, tui_mod.TaskPicker), "the task picker")
        picker = app.screen.query_one(OptionList)
        picker.highlighted = list(Task).index(Task.REASON)
        picker.action_select()
        await _until(pilot, lambda: isinstance(app.screen, tui_mod.ModelPinDialog), "the model dialog")
        select = app.screen.query_one(Select)
        assert [v for _, v in select._options] == ["", "openai:gpt-5-mini"]
        select.value = "openai:gpt-5-mini"
        await _until(pilot, lambda: setup_env.overrides.set_pin_calls == [(Task.REASON, "openai:gpt-5-mini")],
                     "the pin")
        await _until(pilot, lambda: any("pinned reason -> openai:gpt-5-mini" in t for t in _texts(app)),
                     "the line")
        assert setup_env.calls.reload == 1


@_async_test
async def test_pinning_is_refused_mid_turn(setup_env, monkeypatch, tmp_path):
    started = threading.Event()
    release = threading.Event()

    def fake_stream(text, **kwargs):
        started.set()
        release.wait(5)
        yield {"__final__": {"final_output": "x"}, "__trace_id__": "t"}

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test(size=(120, 40)) as pilot:
        app.message_box.value = "go"
        await pilot.press("enter")
        await asyncio.to_thread(started.wait, 5)
        app.action_pin_model()
        await pilot.pause()
        assert not isinstance(app.screen, tui_mod.TaskPicker)
        release.set()
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")


def test_pin_options_narrow_by_requirement_and_provider():
    assert pin_options(Task.REASON, MODELS, TASK_ROUTES) == [("(no pin — default route)", ""), ("openai:gpt-5-mini", "openai:gpt-5-mini")]
    assert [v for _, v in pin_options(Task.SUMMARIZE, MODELS, TASK_ROUTES)] == ["", "inception:mercury-2.5", "openai:gpt-5-mini"]
    assert [v for _, v in pin_options(Task.WEB, MODELS, TASK_ROUTES)] == [""], "WEB is anthropic-only"
    assert [v for _, v in pin_options(Task.CODE_COMPLETE, MODELS, TASK_ROUTES)] == [""], "FIM needs an inception FIM model"


# ---- mount order ------------------------------------------------------------

@_async_test
async def test_the_screen_fills_even_when_its_pane_widgets_mount_late(setup_env, monkeypatch, tmp_path):
    """CI on Windows once raised NoMatches for `#key-provider` from on_mount:
    the widgets inside the TabbedContent panes were not in the DOM yet when
    the screen's own mount fired. The fill now waits a refresh for them.
    Reproduced by making the first look-up of that widget miss."""
    from textual.css.query import NoMatches as _NoMatches

    app = _app(monkeypatch, tmp_path)
    async with app.run_test(size=(120, 40)) as pilot:
        screen = SetupScreen(app._setup_backend())
        original = screen.query_one
        misses = {"n": 0}

        def late(selector, *args, **kwargs):
            if selector == "#key-provider" and misses["n"] == 0:
                misses["n"] += 1
                raise _NoMatches("not mounted yet")
            return original(selector, *args, **kwargs)

        monkeypatch.setattr(screen, "query_one", late)
        app.push_screen(screen)
        await _until(pilot, lambda: isinstance(app.screen, SetupScreen), "the setup screen")
        await _until(
            pilot,
            lambda: "inception" in screen.query_one("#provider-table", DataTable).rows,
            "the provider rows after a retried mount",
        )
        assert misses["n"] == 1, "the first look-up missed, the retry filled the screen"
        assert _cell(screen, "inception", "key") == "********1234"
