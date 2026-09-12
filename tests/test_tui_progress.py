"""The TUI half of agent/pipeline/progress.py, driven headlessly.

Textual's `run_test()` runs the real app against a real (offscreen) screen,
so these assert on what a person would actually see rather than on the
methods that put it there. `_onscreen` drives its own event loop for each
one: `run_test()` is an async context manager and this repo has no asyncio
pytest plugin, and six functions is not enough to grow a dependency for.

The progress seam is stubbed at the source -- these are about the front
end, and a test that needed a provider would not run offline.
"""
import asyncio
import threading

from rich.panel import Panel

from agent.cli import tui as t


class _Ctx:
    """Enough AppContext for OttoApp.__init__ -- it stores ctx and builds a
    Session, and nothing here reaches a router or a client."""
    router = None
    client = None


def _app() -> t.OttoApp:
    return t.OttoApp(_Ctx())


def _shown(widget) -> str:
    """What a Static is currently displaying, as text.

    `.content` is the raw renderable it was handed -- a str for a status
    line, a Panel for an answer -- so this flattens both rather than making
    each test know which it is looking at.
    """
    content = widget.content
    inner = getattr(content, "renderable", content)
    return str(getattr(inner, "markup", inner))


def _onscreen(body) -> None:
    """Run `body(app)` inside a live offscreen app."""
    async def driver() -> None:
        app = _app()
        async with app.run_test():
            body(app)
    asyncio.run(driver())


def test_the_clock_reads_as_minutes_and_seconds():
    assert t._clock(0) == "0:00"
    assert t._clock(9.9) == "0:09"
    assert t._clock(131) == "2:11"


def test_the_status_line_is_empty_until_a_turn_starts():
    def body(app):
        assert _shown(app.query_one("#status")) == ""
    _onscreen(body)


def test_the_status_line_says_what_the_run_is_doing():
    """The whole point. Before this the middle of a turn was blank for as
    long as the turn took -- 131 seconds on the longest golden item."""
    def body(app):
        app._turn_running = True
        app._on_progress(t.Progress(kind="phase", text="working it out", calls=2))
        app._on_progress(t.Progress(kind="call_start", text="gpt-5-mini", calls=3))
        app._on_progress(t.Progress(
            kind="tool", text="execute_bash", calls=3, detail={"target": "pytest"}))
        app._draw_status()
        shown = _shown(app.query_one("#status"))

        assert "execute_bash pytest" in shown
        assert "gpt-5-mini" in shown
        assert "3 calls" in shown
        assert "esc to stop" in shown
    _onscreen(body)


def test_the_clock_keeps_moving_when_nothing_is_reported():
    """Between two model calls a run reports nothing for ten seconds at a
    stretch. A status line that only moves when the run moves reads as a
    hung app, which is the complaint this is answering."""
    def body(app):
        app._turn_running = True
        app._started = 0.0              # a long time ago, on the monotonic clock
        app._tick()
        first = _shown(app.query_one("#status"))
        app._tick()
        second = _shown(app.query_one("#status"))

        assert first != second, "the spinner did not advance on its own"
    _onscreen(body)


def test_an_answer_appears_as_it_is_written():
    def body(app):
        app._show_partial_answer("the answer so f")
        app._show_partial_answer("the answer so far")
        assert app._answer is not None
        assert "the answer so far" in _shown(app._answer)
    _onscreen(body)


def test_the_finished_answer_replaces_the_streamed_one():
    """Not a second block beside it. Posting both would leave the same
    answer on screen twice, once raw and once rendered."""
    def body(app):
        before = len(app.transcript.children)
        app._show_partial_answer("partial")
        app._settle_answer(Panel("done"))

        assert len(app.transcript.children) == before + 1, "the stream was left behind"
        assert app._answer is None
    _onscreen(body)


def test_an_answer_that_never_streamed_still_gets_mounted():
    """A run that died, or one whose whole reply arrived in a single chunk,
    never reaches _show_partial_answer."""
    def body(app):
        before = len(app.transcript.children)
        app._settle_answer(Panel("done"))
        assert len(app.transcript.children) == before + 1
    _onscreen(body)


def test_only_a_reply_that_has_reached_final_is_shown():
    """Before FINAL: the reply is a tool call. Showing it as the answer
    would put "ACTION: execute_bash" on screen dressed as a result."""
    app = _app()
    app._stream_answer("ACTION: execute_bash\nCODE:\npytest -q")
    assert app._answer is None


def test_a_streamed_frame_is_the_whole_reply_not_the_next_slice():
    """The seam hands over settled text each time; this checks the TUI
    treats it that way and shows what follows the marker, not the marker."""
    app = _app()
    shown: list[str] = []
    app.call_from_thread = lambda fn, *a: shown.append(a[0])
    app._drawn_at = 0.0

    app._stream_answer("thinking about it\nFINAL:\nforty two")

    assert shown == ["forty two"]


def test_a_second_message_during_a_turn_is_refused_not_queued():
    """`exclusive=True` cancels the previous worker, and a worker blocked in
    a network call cannot be cancelled out from under it -- so a second
    submit used to leave two runs writing into the same transcript."""
    app = _app()
    posted: list[object] = []
    app._post = posted.append
    app._turn_running = True

    class _Box:
        """Enough of Textual's Input for the handler: it reads `.id` to tell
        the message box from a modal's own, and clears `.value`."""
        id = "message-input"
        value = ""

    class _Event:
        value = "another one"
        input = _Box()

    app.on_input_submitted(_Event())

    assert posted and "still working" in str(posted[0])
    assert "esc to stop" in str(posted[0])


def test_escape_does_nothing_when_no_turn_is_running():
    app = _app()
    app._turn_running = False
    app._cancel = None
    app.action_stop_turn()              # must not raise


def test_escape_asks_the_run_to_stop():
    """Cooperative: _call checks this before it spends again, so the stop
    lands within one model call and never pays for another."""
    app = _app()
    app._turn_running = True
    app._cancel = threading.Event()
    app._draw_status = lambda: None

    app.action_stop_turn()

    assert app._cancel.is_set()
    assert "stopping" in app._phase
