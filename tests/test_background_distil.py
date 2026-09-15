"""A phone run answers before it learns (agent/pipeline/nodes.py
learned_from -> _learn_in_background). Offline: a scripted judge and
distiller, a fake phone, a tmp_path bank."""
from __future__ import annotations

import logging
import threading

from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph import END

from agent.memory import lessons as L
from agent.phone import JsonBackend, phone_tools
from agent.pipeline import nodes as pn
from agent.pipeline.progress import bind_progress
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import FakePhone
from tests.test_lessons import bank, fake_embeddings  # noqa: F401 -- fixtures

LESSON = '[{"cue": "a results list opens sorted by relevance", "action": "open sort, pick price", "outcome": "worked"}]'


class _JudgeAndDistiller:
    """Approves as the judge; answers the distil prompt with one lesson."""

    def stream(self, messages):
        distilling = pn.DISTIL_PROMPT in str(messages[0].content)
        yield AIMessageChunk(content=LESSON if distilling else "FINAL:\nAPPROVE: yes\nWHY: the screen shows it")


def _state() -> dict:
    return {
        "messages": [HumanMessage("find the cheapest phone")], "node": "agent",
        "output": "the cheapest is model 3 at Rs 9003", "feedback": "", "context": "", "board": [],
        "checklist": [{"text": "the cheapest phone is named", "status": "pending"}],
        "actions": ["solve: phone_act tap ok"], "transcript": [{"kind": "ai", "content": "FINAL:\ndone"}],
        "rejections": 0, "mode_log": [],
    }


def _phone():
    return bind_extra_tools(phone_tools(JsonBackend(FakePhone([]))))


def _install(monkeypatch):
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _JudgeAndDistiller())


def _blocked(monkeypatch, release: threading.Event, started: threading.Event | None = None):
    real = pn._distil

    def distil(state, *, succeeded):
        if started is not None:
            started.set()
        assert release.wait(10)
        return real(state, succeeded=succeeded)

    monkeypatch.setattr(pn, "_distil", distil)


def test_a_phone_run_answers_before_its_lesson_is_distilled(bank, fake_embeddings, monkeypatch):  # noqa: F811
    _install(monkeypatch)
    release, started = threading.Event(), threading.Event()
    _blocked(monkeypatch, release, started)

    with _phone():
        result = pn.evaluator(_state())

    assert result.goto == END and result.update["final_output"].startswith("the cheapest is")
    assert "learning from this run in the background" in result.update["board"]
    assert not any(line.startswith("learned:") for line in result.update["board"])
    assert started.wait(5) and not release.is_set()
    with L.bind_kind(L.PHONE_KIND):
        assert L.all_lessons() == []

    release.set()
    assert pn.wait_for_learning(10)
    assert L.all_lessons() == [], "a phone lesson never lands in the workspace bank"
    with L.bind_kind(L.PHONE_KIND):
        assert [lesson.cue for lesson in L.all_lessons()] == ["a results list opens sorted by relevance"]


def test_nothing_reaches_the_callers_progress_sink_from_the_thread(bank, fake_embeddings, monkeypatch):  # noqa: F811
    _install(monkeypatch)
    release = threading.Event()
    _blocked(monkeypatch, release)
    heard: list = []

    with bind_progress(heard.append), _phone():
        pn.evaluator(_state())
    before = len(heard)
    release.set()
    assert pn.wait_for_learning(10)
    with L.bind_kind(L.PHONE_KIND):
        assert L.all_lessons(), "the thread did make its call"
    assert len(heard) == before


def test_a_failing_distil_is_logged_not_raised(bank, fake_embeddings, monkeypatch, caplog):  # noqa: F811
    _install(monkeypatch)

    def boom(state, *, succeeded):
        raise RuntimeError("distiller fell over")

    monkeypatch.setattr(pn, "_distil", boom)
    with caplog.at_level(logging.ERROR, logger=pn.logger.name), _phone():
        result = pn.evaluator(_state())
        assert pn.wait_for_learning(10)
    assert result.goto == END
    assert any("learning from a finished phone run failed" in r.getMessage() for r in caplog.records)


def test_a_coding_run_still_learns_before_it_ends(bank, fake_embeddings, monkeypatch):  # noqa: F811
    _install(monkeypatch)
    result = pn.evaluator(_state())
    assert result.goto == END
    assert "learned: When a results list opens sorted by relevance: open sort, pick price [worked]" in (
        result.update["board"])
    assert not [t for t in threading.enumerate() if t.name == "otto-distil"]
    assert [lesson.cue for lesson in L.all_lessons()] == ["a results list opens sorted by relevance"]


def test_a_read_only_phone_run_writes_nothing_and_starts_nothing(bank, fake_embeddings, monkeypatch):  # noqa: F811
    _install(monkeypatch)
    with L.read_only(), _phone():
        result = pn.evaluator(_state())
    assert pn.wait_for_learning(10)
    assert result.goto == END
    assert "learning from this run in the background" not in result.update["board"]
    with L.bind_kind(L.PHONE_KIND):
        assert L.all_lessons() == []
    assert L.all_lessons() == []


def test_closing_a_session_waits_for_the_lesson(monkeypatch):
    from agent import embed

    waited: list = []
    monkeypatch.setattr(pn, "wait_for_learning", lambda timeout=None: waited.append(timeout) or True)

    class _Session:
        closed = False

        def close(self):
            self.closed = True

    session = _Session()
    embed.SessionHandle(session).close()
    assert waited == [10] and session.closed
