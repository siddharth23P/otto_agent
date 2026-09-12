"""What the TUI's front-end layer has to get right, separately from the graph.

There were no tests here at all before 2026-09-12, and the bug report that
prompted these ("in otto tui it's acting weird and slow") was three separate
front-end faults, none of which the pipeline suite could have caught: a second
Enter starting a concurrent pipeline run, a modal's Input.Submitted escaping
into a brand-new turn, and every thinking line being buffered unrendered
behind a collapsed container. Each one is pinned below, along with the two
Textual behaviours the fixes rest on -- those are asserted directly against
the installed Textual rather than taken on trust, because all three faults
came from assuming the opposite.

Nothing here touches a model: `run_pipeline_stream`/`resume_pipeline_stream`
are patched in agent.cli.tui's namespace with generators that yield the same
event shapes agent/pipeline/run.py documents.
"""
from __future__ import annotations

import asyncio
import functools
import threading
import time

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Input, RichLog, Static

from agent.cli import tui as tui_mod
from agent.cli.tui import OttoApp


def _async_test(fn):
    """Run an async test body without pulling pytest-asyncio into the dev
    group for one module. Textual's own `run_test()` needs a running loop and
    nothing else here does, so a plain `asyncio.run` per test is the whole
    requirement -- `functools.wraps` keeps pytest's fixture introspection
    (monkeypatch, tmp_path) working on the wrapper's signature."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return asyncio.run(fn(*args, **kwargs))
    return wrapper


async def _until(pilot, predicate, what: str, timeout: float = 15.0):
    """Pump the UI until `predicate()` holds, or fail saying what was awaited.

    A real deadline rather than `for _ in range(40): await pilot.pause()`. The
    iteration-count form is a bet on how fast the runner is, and this file
    already lost that bet once: a fixed 0.4s window passed on Linux and
    Windows and failed on a loaded macOS runner at 3 steps of 5. Everything
    here waits on a pipeline stub or a worker thread, so what it is really
    waiting for is a scheduler, and only wall time bounds that honestly.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause()
        if predicate():
            return
    raise AssertionError(f"timed out after {timeout}s waiting for {what}")


# --------------------------------------------------------------------------
# The Textual behaviours the fixes depend on. Asserted, not assumed.
# --------------------------------------------------------------------------

@_async_test
async def test_modal_input_submitted_bubbles_to_the_app():
    """Why every modal here calls event.stop(): dismissing does NOT consume
    the event. Without the stop, text typed into AskUserModal reaches
    OttoApp.on_input_submitted too and is started as a whole new turn."""
    seen: list[str] = []

    class Modal(ModalScreen[str]):
        def compose(self) -> ComposeResult:
            with Vertical():
                yield Input()

        def on_mount(self) -> None:
            self.query_one(Input).focus()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.dismiss(event.value)  # deliberately NOT stopped

    class Host(App):
        def compose(self) -> ComposeResult:
            yield Input(id="message-input")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            seen.append(event.value)

    app = Host()
    async with app.run_test() as pilot:
        app.push_screen(Modal())
        await pilot.pause()
        app.screen.query_one(Input).value = "answer"
        await pilot.press("enter")
        await pilot.pause()

    assert seen == ["answer"]


@_async_test
async def test_exclusive_does_not_stop_a_running_thread_worker():
    """Why run_turn no longer carries exclusive=True and `_turn_running`
    exists instead: Worker.cancel() cancels the asyncio task wrapping
    run_in_executor, and the executor thread runs to completion regardless.

    Gated on events rather than sleeps. An earlier version raced two workers
    and asserted how far each had got inside a fixed 0.4s window, which is a
    claim about CI scheduling, not about Textual -- it passed on Linux and
    Windows and failed on a loaded macOS runner at 3 steps of 5. The claim
    that actually matters needs no clock: a worker that has been cancelled
    still reaches its own last line.
    """
    from textual import work

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class Host(App):
        @work(thread=True, exclusive=True, group="turn")
        def first(self) -> None:
            entered.set()
            release.wait(10)
            finished.set()

        @work(thread=True, exclusive=True, group="turn")
        def second(self) -> None:
            pass

    app = Host()
    async with app.run_test() as pilot:
        worker = app.first()
        await asyncio.to_thread(entered.wait, 10)

        app.second()  # same group + exclusive: Textual cancels `worker`
        await _until(pilot, lambda: worker.is_cancelled, "Textual to mark the worker cancelled")

        # ...and the thread underneath it is still sitting there, very much
        # alive, waiting to be let go.
        assert not finished.is_set()
        release.set()
        assert await asyncio.to_thread(finished.wait, 10), (
            "a cancelled thread worker still runs to completion -- which is "
            "why a second turn used to start a second pipeline run"
        )


@_async_test
async def test_a_collapsed_richlog_renders_nothing_until_expanded():
    """Why this turn's thinking block is mounted OPEN: RichLog.write() defers
    every write until the widget's size is known, and a collapsed Collapsible
    never lays its contents out. Collapsed-from-birth meant a blank screen
    for the whole turn and one synchronous render burst on expand."""
    class Host(App):
        CSS = "#t { height: 1fr; } .thinking-log { height: auto; max-height: 16; }"

        def compose(self) -> ComposeResult:
            yield VerticalScroll(id="t")

    app = Host()
    async with app.run_test(size=(100, 30)) as pilot:
        log = RichLog(wrap=True, markup=True, highlight=False)
        log.add_class("thinking-log")
        block = Collapsible(log, title="thinking…", collapsed=True)
        app.query_one("#t").mount(block)
        await pilot.pause()
        await pilot.pause()
        for i in range(40):
            log.write(f"board line {i}")
        await pilot.pause()
        assert len(log.lines) == 0  # nothing on screen at all

        block.collapsed = False
        await pilot.pause()
        await pilot.pause()
        assert len(log.lines) == 40  # all of it, at once


@_async_test
async def test_each_modal_stops_its_own_submission():
    """The modals' half of "modal input must not escape into a new turn",
    pinned on its own. OttoApp.on_input_submitted also filters by
    `event.input.id`, so a regression here would be invisible through the
    app -- this hosts the real modals under a bare App instead, where the
    only thing that can stop the event is the modal itself."""
    escaped: list[str] = []

    class Host(App):
        def compose(self) -> ComposeResult:
            yield Static("host")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            escaped.append(event.value)

    app = Host()
    async with app.run_test() as pilot:
        app.push_screen(tui_mod.AskUserModal("which one?", []))
        await pilot.pause()
        app.screen.query_one(Input).value = "an answer"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        app.push_screen(tui_mod.ScoreDialog())
        await pilot.pause()
        app.screen.query_one(Input).value = "0.8 nice"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert escaped == []


# --------------------------------------------------------------------------
# OttoApp itself
# --------------------------------------------------------------------------

class FakeRouter:
    def prewarm(self):
        return []


class FakeCtx:
    """Enough AppContext for the TUI: it only ever reaches for `router` and
    `client`, and the patched stream functions mean neither is used here."""

    def __init__(self):
        self.router = FakeRouter()
        self.scores: list[tuple] = []

    @property
    def client(self):
        return self

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        pass


def _final(text: str) -> dict:
    return {"__final__": {"final_output": text}, "__trace_id__": "trace-1"}


def _make_app(monkeypatch, tmp_path, stream_fn, resume_fn=None):
    monkeypatch.setattr(tui_mod, "run_pipeline_stream", stream_fn)
    if resume_fn is not None:
        monkeypatch.setattr(tui_mod, "resume_pipeline_stream", resume_fn)
    monkeypatch.setattr(tui_mod, "save_final", lambda *a, **k: tmp_path / "out.md")
    app = OttoApp(FakeCtx())
    return app


def _texts(app) -> list[str]:
    """Every result-side block, in mounted order, as whatever `_post` was
    handed -- a markup string for the status lines, a Rich renderable for the
    final panel. `Static.content` is the original object, before Textual
    visualises it."""
    return [c if isinstance(c, str) else str(c)
            for c in (w.content for w in app.query("#transcript > Static"))]


def _panel_text(app) -> str:
    """The token panel's Rich table, rendered to plain text so a test can read
    what a person would see rather than the dict behind it."""
    import io

    from rich.console import Console

    buf = io.StringIO()
    Console(file=buf, width=32, force_terminal=False).print(app.usage_panel.content)
    return buf.getvalue()


@_async_test
async def test_a_second_enter_mid_turn_starts_no_second_run(monkeypatch, tmp_path):
    """The reported symptom: the same message posted twice, and an older
    run's answer landing under a newer run's thinking block."""
    started = threading.Event()
    release = threading.Event()
    runs: list[str] = []

    def fake_stream(text, **kwargs):
        runs.append(text)
        started.set()
        release.wait(5)
        yield _final(f"answer to {text}")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        box = app.message_box
        box.value = "first message"
        await pilot.press("enter")
        await asyncio.to_thread(started.wait, 5)
        await pilot.pause()

        assert app._turn_running is True
        assert box.disabled is True
        assert box.placeholder == OttoApp.BUSY_PLACEHOLDER

        # The impatient second Enter. A disabled Input swallows keys, so
        # submit the message directly -- the guard, not the disabled flag,
        # is what has to hold.
        app.post_message(Input.Submitted(box, "second message", None))
        await pilot.pause()
        await pilot.pause()

        assert runs == ["first message"]
        assert any("still working on the previous message" in t for t in _texts(app))

        release.set()
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")

        assert box.disabled is False
        assert box.placeholder == OttoApp.IDLE_PLACEHOLDER
        assert runs == ["first message"]


@_async_test
async def test_answering_a_mid_run_question_does_not_start_a_new_turn(monkeypatch, tmp_path):
    """AskUserModal stops its own Input.Submitted, so the answer resumes the
    paused run and nothing else -- it used to also be posted a second time
    and launched as a concurrent turn on the answer text."""
    starts: list[str] = []
    resumes: list[str] = []

    def fake_stream(text, **kwargs):
        starts.append(text)
        yield {"__ask__": {"question": "which one?", "choices": [], "thread_id": "t-1"}}

    def fake_resume(answer, **kwargs):
        resumes.append(answer)
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream, fake_resume)
    async with app.run_test() as pilot:
        app.message_box.value = "do the thing"
        await pilot.press("enter")
        await _until(pilot, lambda: isinstance(app.screen, tui_mod.AskUserModal),
                     "the question modal")

        app.screen.query_one(Input).value = "the second one"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the resumed turn to finish")

        assert starts == ["do the thing"]
        assert resumes == ["the second one"]
        # Posted once, as the answer -- not twice, and not as a fresh turn.
        assert [t for t in _texts(app) if "the second one" in t] == [
            "[bold]you[/] the second one"
        ]


@_async_test
async def test_thinking_block_is_open_while_running_and_shut_after(monkeypatch, tmp_path):
    """Board lines have to be visible as they arrive (that is the whole
    progress signal during a turn), and folded away once the answer is up."""
    release = threading.Event()
    started = threading.Event()

    def fake_stream(text, **kwargs):
        yield {"agent": {"board": ["agent: execute_bash -> ok"]}}
        started.set()
        release.wait(5)
        yield {"agent": {"board": ["otto has an answer"]}}
        yield _final("the answer")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test(size=(100, 30)) as pilot:
        app.message_box.value = "hi"
        await pilot.press("enter")
        await asyncio.to_thread(started.wait, 5)
        await pilot.pause()
        await pilot.pause()

        block = app.query_one("#transcript Collapsible", Collapsible)
        log = block.query_one(RichLog)
        assert block.collapsed is False
        assert len(log.lines) > 0, "board lines must render while the turn runs"

        release.set()
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")

        assert block.collapsed is True
        assert "steps" in block.title


@_async_test
async def test_a_failing_turn_still_hands_the_session_back(monkeypatch, tmp_path):
    """A provider error mid-turn must not leave the message box disabled
    forever -- which is indistinguishable from the app having hung."""
    def fake_stream(text, **kwargs):
        raise RuntimeError("provider exploded")
        yield  # pragma: no cover -- makes this a generator

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        app.message_box.value = "hi"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the failed turn to release the session")

        assert app.message_box.disabled is False
        assert any("provider exploded" in t for t in _texts(app))


@_async_test
async def test_the_workspace_reaches_the_pipeline_and_the_screen(monkeypatch, tmp_path):
    """A session's workspace has to arrive as run_pipeline_stream's own
    argument, not as a contextvar the CLI thread set -- the worker thread that
    consumes the stream cannot see that one."""
    passed: list = []

    def fake_stream(text, **kwargs):
        passed.append(kwargs.get("workspace"))
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    app.session.workspace = tmp_path
    async with app.run_test() as pilot:
        assert any(str(tmp_path) in t for t in _texts(app)), "say it up front"
        app.message_box.value = "fix the tests"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        assert passed == [str(tmp_path)]


@_async_test
async def test_changing_the_workspace_is_refused_mid_turn(monkeypatch, tmp_path):
    """A running turn already handed its workspace to the pipeline, so a
    change now would take effect next turn while appearing to take effect on
    this one."""
    started = threading.Event()
    release = threading.Event()

    def fake_stream(text, **kwargs):
        started.set()
        release.wait(5)
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    app.session.workspace = tmp_path
    async with app.run_test() as pilot:
        app.message_box.value = "go"
        await pilot.press("enter")
        await asyncio.to_thread(started.wait, 5)
        await pilot.pause()

        app.action_workspace()
        await pilot.pause()
        assert not isinstance(app.screen, tui_mod.WorkspacePrompt)
        assert any("a turn is still running" in t for t in _texts(app))
        assert app.session.workspace == tmp_path

        release.set()
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")


@_async_test
async def test_the_workspace_prompt_opens_and_closes_file_access(monkeypatch, tmp_path):
    other = tmp_path / "other"
    other.mkdir()

    app = _make_app(monkeypatch, tmp_path, lambda *a, **k: iter(()))
    app.session.workspace = tmp_path
    async with app.run_test() as pilot:
        app.action_workspace()
        await pilot.pause()
        assert isinstance(app.screen, tui_mod.WorkspacePrompt)
        # Pre-filled with the current root, so "one level over" is an edit.
        assert app.screen.query_one(Input).value == str(tmp_path)
        app.screen.query_one(Input).value = str(other)
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert app.session.workspace == other.resolve()

        app.action_workspace()
        await pilot.pause()
        app.screen.query_one(Input).value = "off"
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert app.session.workspace is None
        assert any("file tools are off" in t for t in _texts(app))


@_async_test
async def test_a_bad_workspace_path_is_reported_and_changes_nothing(monkeypatch, tmp_path):
    app = _make_app(monkeypatch, tmp_path, lambda *a, **k: iter(()))
    app.session.workspace = tmp_path
    async with app.run_test() as pilot:
        app.action_workspace()
        await pilot.pause()
        app.screen.query_one(Input).value = str(tmp_path / "projcts")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert app.session.workspace == tmp_path
        assert any("does not exist" in t for t in _texts(app))
        assert not (tmp_path / "projcts").exists()


@_async_test
async def test_scoring_does_not_block_the_ui_thread(monkeypatch, tmp_path):
    """create_score/flush are Langfuse network calls; on the event loop they
    freeze the app. The score must be posted from a worker, against the
    trace id that was on screen when the command was invoked."""
    in_flight = threading.Event()
    release = threading.Event()

    app = _make_app(monkeypatch, tmp_path, lambda *a, **k: iter(()))

    def slow_flush():
        in_flight.set()
        release.wait(5)

    app.ctx.flush = slow_flush
    app.session.trace_id = "trace-abc"

    async with app.run_test() as pilot:
        app.action_score(1.0, "good")
        await asyncio.to_thread(in_flight.wait, 5)
        # The UI is still live while the "network call" is outstanding.
        await pilot.pause()
        app._post("ui still responsive")
        await pilot.pause()
        assert "ui still responsive" in _texts(app)

        release.set()
        await _until(pilot, lambda: any("scored 1" in t for t in _texts(app)),
                     "the score to be posted")

    assert app.ctx.scores == [
        {"name": "user_feedback", "value": 1.0, "data_type": "NUMERIC",
         "trace_id": "trace-abc", "comment": "good"}
    ]


# --------------------------------------------------------------------------
# The token panel (agent/pipeline/usage.py, UsagePanel). The ledger is the
# APP's, handed to every turn and every resume as an argument -- the worker
# thread that consumes the stream cannot see a contextvar the UI thread set,
# which is the same trap the workspace test above exists for.
# --------------------------------------------------------------------------

@_async_test
async def test_the_apps_ledger_is_the_one_the_pipeline_records_into(monkeypatch, tmp_path):
    passed: list = []

    def fake_stream(text, **kwargs):
        passed.append(kwargs.get("usage"))
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        app.message_box.value = "do a thing"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        assert passed == [app.usage]


@_async_test
async def test_the_panel_shows_what_a_turn_spent(monkeypatch, tmp_path):
    def fake_stream(text, **kwargs):
        # Stand in for _call, which is what records in a real run.
        kwargs["usage"].record("us.anthropic.claude-sonnet-4-20250514-v1:0",
                               {"input_tokens": 41000, "output_tokens": 1200})
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        assert "nothing yet" in _panel_text(app)
        app.message_box.value = "do a thing"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        await pilot.pause()

        shown = _panel_text(app)
        assert "claude-sonnet-4" in shown, "the model that answered is named"
        assert "42.2k" in shown, "its tokens are totalled"


@_async_test
async def test_the_panel_is_cumulative_across_turns(monkeypatch, tmp_path):
    def fake_stream(text, **kwargs):
        kwargs["usage"].record("m", {"input_tokens": 1000, "output_tokens": 0})
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        for _ in range(3):
            app.message_box.value = "again"
            await pilot.press("enter")
            await _until(pilot, lambda: not app._turn_running, "the turn to finish")

        assert app.usage.calls == 3
        assert app.usage.total_tokens == 3000


@_async_test
async def test_a_model_that_reports_no_tokens_is_not_shown_as_zero(monkeypatch, tmp_path):
    # "said nothing" and "cost nothing" have to look different, or the panel
    # quietly lies about any provider that does not report usage.
    def fake_stream(text, **kwargs):
        kwargs["usage"].record("quiet-model", None)
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        app.message_box.value = "do a thing"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        await pilot.pause()

        shown = _panel_text(app)
        assert "quiet-model" in shown
        assert "--" in shown


@_async_test
async def test_the_panel_can_be_hidden_to_give_the_transcript_its_columns(monkeypatch, tmp_path):
    app = _make_app(monkeypatch, tmp_path, lambda text, **kw: iter([_final("x")]))
    async with app.run_test() as pilot:
        assert app.usage_panel.display
        await pilot.press("ctrl+t")
        assert not app.usage_panel.display
        await pilot.press("ctrl+t")
        assert app.usage_panel.display


@_async_test
async def test_a_new_session_starts_the_panel_at_nothing(monkeypatch, tmp_path):
    def fake_stream(text, **kwargs):
        kwargs["usage"].record("m", {"input_tokens": 500, "output_tokens": 5})
        yield _final("done")

    app = _make_app(monkeypatch, tmp_path, fake_stream)
    async with app.run_test() as pilot:
        app.message_box.value = "do a thing"
        await pilot.press("enter")
        await _until(pilot, lambda: not app._turn_running, "the turn to finish")
        assert app.usage.total_tokens == 505

        app.action_new_session()
        await _until(pilot, lambda: not app._turn_running, "the reset to finish")
        await pilot.pause()

        assert app.usage.total_tokens == 0
        assert "nothing yet" in _panel_text(app)
