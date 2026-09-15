"""The lesson lookup a fresh run's seed needs runs beside its rubric call
(agent/pipeline/nodes.py _look_up_lessons_early). Offline."""
from __future__ import annotations

import threading

from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.memory import lessons as L
from agent.phone import JsonBackend, phone_tools
from agent.pipeline import nodes as pn
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import FakePhone

BLOCK = "FROM AN EARLIER RUN (might not apply -- ignore it if it does not):\n- When the test is flaky: rerun it once [worked]"


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self._replies.pop(0) if self._replies else "FINAL:\ndone")


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("fix the failing test")],
        "board": [], "node": None, "feedback": "", "output": None, "context": "", "node_error": None,
        "pending_question": None, "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [], "model_calls": 0, "rejections": 0,
    }
    base.update(overrides)
    return base


def _lookups(monkeypatch):
    calls: list[tuple[str, str]] = []
    started = threading.Event()

    def lessons_block(task_text):
        calls.append((task_text, pn._lesson_kind()))
        started.set()
        return BLOCK

    monkeypatch.setattr(pn, "_lessons_block", lessons_block)
    return calls, started


def test_the_lookup_runs_while_the_rubric_call_does_and_reaches_the_seed(monkeypatch):
    calls, started = _lookups(monkeypatch)
    during: list[bool] = []

    def rubric(llm, task):
        # Returns only once the lookup has begun: were it still run after
        # the rubric, this would wait out its timeout and record False.
        during.append(started.wait(5))
        return pn.Rubric(["the test passes"])

    fake = _Scripted(["FINAL:\nthe test passes"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", rubric)
    with L.bind_bank(None):
        pn.agent(_state())

    assert during == [True]
    assert calls == [("fix the failing test", L.KIND)]
    seed_body = next(m.content for m in fake.seen[0] if str(m.content).startswith("TASK:"))
    assert BLOCK in seed_body


def test_the_lookup_sees_the_runs_own_bindings(monkeypatch):
    calls, _ = _lookups(monkeypatch)
    fake = _Scripted(["FINAL:\ndone"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["milk is in the cart"]))
    with L.bind_bank(None), bind_extra_tools(phone_tools(JsonBackend(FakePhone([])))):
        pn.agent(_state(messages=[HumanMessage("add milk")]))
    assert calls == [("add milk", L.PHONE_KIND)]


def test_a_turn_answered_on_the_chat_fast_path_seeds_nothing(monkeypatch):
    calls, _ = _lookups(monkeypatch)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _Scripted([]))
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric([], conversational=True))
    monkeypatch.setattr(pn, "_chat_reply", lambda state, task_text: "hello!")
    with L.bind_bank(None):
        result = pn.agent(_state(messages=[HumanMessage("hi")]))
    assert result.update["final_output"] == "hello!"
    assert len(calls) <= 1


def test_a_resumed_run_does_not_look_lessons_up(monkeypatch):
    calls, _ = _lookups(monkeypatch)
    fake = _Scripted(["FINAL:\nfirst"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["the test passes"]))
    with L.bind_bank(None):
        first = pn.agent(_state())
        calls.clear()
        pn.agent(_state(transcript=first.update["transcript"], checklist=first.update.get("checklist"),
                        feedback="run it again"))
    assert calls == []
