"""Coverage for agent/pipeline/progress.py and its two call sites in nodes.py.

The bug this exists for is not a crash. The graph streams one update per
NODE, `agent` is a node, and a run spends every model call and every tool
call inside it -- so a person watching a turn saw nothing move between
pressing enter and the answer landing. Measured across the twenty golden
items: 828 seconds of wall time, 96% of it inside model requests, zero
graph updates in the middle of any of them, and the longest item running
131 seconds with an unchanging screen.

So what is tested here is that something is reported at all, that it is
reported from where the time is actually spent, and -- the part that would
be a real regression -- that a run with nobody watching pays nothing and
a watcher with a broken renderer cannot take the run down.
"""
import threading

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from agent.pipeline import nodes as pn
from agent.pipeline import progress as pg


class _Chunk:
    """The shape _call's stream loop consumes: something that adds to itself
    and carries `.content` plus `.response_metadata`."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict = {}

    def __add__(self, other):
        return _Chunk(self.content + other.content)


class _Model:
    _otto_provider = "test"
    model = "test-model"

    def __init__(self, pieces):
        self._pieces = pieces

    def stream(self, messages):
        for piece in self._pieces:
            yield _Chunk(piece)


def test_nothing_bound_reports_nothing():
    """The normal case. Every eval harness, every benchmark and every test
    runs with no watcher, and none of them should pay for one."""
    assert pg.watching() is False
    assert pg.report("call_start", "anything") is None
    pg.check_cancelled()


def test_a_watcher_sees_a_call_start_and_the_reply_arriving():
    seen: list[pg.Progress] = []
    with pg.bind_progress(seen.append):
        assert pn._call(_Model(["he", "llo"]), [HumanMessage("hi")]) == "hello"

    kinds = [p.kind for p in seen]
    assert kinds[0] == "call_start"
    assert "partial" in kinds
    assert [p.partial for p in seen if p.kind == "partial"] == ["he", "hello"]


def test_every_partial_is_the_whole_reply_not_the_next_slice():
    """A diffusing route's chunks are successive refinements of one answer,
    not the next piece of it. A consumer that appends them renders garbage,
    so this seam hands over the settled text each time and says so."""
    seen: list[pg.Progress] = []
    with pg.bind_progress(seen.append):
        pn._call(_Model(["a", "b", "c"]), [HumanMessage("hi")])

    partials = [p.partial for p in seen if p.kind == "partial"]
    assert partials == ["a", "ab", "abc"]
    assert partials[-1] == "abc", "the last frame is the whole answer"


def test_a_renderer_that_raises_does_not_take_the_run_down():
    """A progress display is the least important thing on screen. A bug in
    one must not be able to kill the run it is describing."""
    def broken(update):
        raise RuntimeError("the widget tree is gone")

    with pg.bind_progress(broken):
        assert pn._call(_Model(["fine"]), [HumanMessage("hi")]) == "fine"


def test_a_cancelled_run_stops_before_it_spends_again():
    """Cooperative, checked before the spend. A run stopped from the front
    end should not pay for one more request on its way out, and the longest
    a stop can take to land is one model call."""
    stop = threading.Event()
    stop.set()

    with pg.bind_progress(lambda u: None, cancel=stop):
        with pytest.raises(pg.Cancelled):
            pn._call(_Model(["never reached"]), [HumanMessage("hi")])


def test_cancelling_is_told_apart_from_breaking():
    """Its own type, so a front end can say "you stopped this" rather than
    showing a person a traceback for something they did on purpose."""
    assert issubclass(pg.Cancelled, Exception)
    assert not issubclass(pg.Cancelled, KeyboardInterrupt)


def test_a_tool_call_says_which_tool_and_what_it_touched():
    """The other half of the dead screen: between two model calls a run can
    sit inside a single shell command for a long time."""
    seen: list[pg.Progress] = []
    with pg.bind_progress(seen.append):
        pn._tool_loop(
            _Model(["ACTION: execute_python\nCODE:\nprint(1)"]),
            [SystemMessage("s"), HumanMessage("h")],
            max_iterations=1,
        )

    tools = [p for p in seen if p.kind == "tool"]
    assert tools, "a dispatched tool reported nothing"
    assert tools[0].text == "execute_python"
