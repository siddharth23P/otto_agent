"""Coverage for evaluator() (agent/pipeline/nodes.py) -- judges the
dispatched specialist's answer with the same ACTION/FINAL tool-calling
loop every role node has (_tool_loop), then either ends the graph (approve,
or gives up after MAX_DISPATCH_ROUNDS) or sends rejection feedback back to
router() -- never straight back to the same specialist (2026-09-10 design
call: "router re-dispatches").

_parse_approval defaults to approve=False whenever _tool_loop's reply
never contained an "APPROVE:" line at all (e.g. it exhausted on
unparseable replies) -- fails CLOSED by construction. That default is
exercised here too: an evaluator that never rendered a real verdict must
not be mistaken for one that approved.
"""
from langgraph.graph import END
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.pipeline import nodes as pn


class _FakeModel:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[list[str]] = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._reply)


def _install(monkeypatch, fake):
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("is 17 prime?")],
        "board": [],
        "round": 1,
        "node": "solver",
        "feedback": "",
        "output": "yes, 17 is prime",
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_evaluator_approves_and_ends_the_graph(monkeypatch):
    fake = _FakeModel("FINAL:\nAPPROVE: yes\nWHY: correct and verified")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state())

    assert result.goto == END
    assert result.update["final_output"] == "yes, 17 is prime"


def test_evaluator_rejects_and_routes_feedback_back_to_router(monkeypatch):
    fake = _FakeModel("FINAL:\nAPPROVE: no\nWHY: never checked divisibility")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(round=1))

    assert result.goto == "router"
    assert result.update["feedback"] == "never checked divisibility"
    assert "final_output" not in result.update


def test_evaluator_gives_up_after_max_dispatch_rounds_and_uses_the_last_answer(monkeypatch):
    fake = _FakeModel("FINAL:\nAPPROVE: no\nWHY: still not great")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(round=pn.MAX_DISPATCH_ROUNDS))

    assert result.goto == END
    assert result.update["final_output"] == "yes, 17 is prime"


def test_evaluator_treats_a_verdict_less_reply_as_a_rejection_not_an_approval(monkeypatch):
    # _tool_loop exhausts on unparseable replies -> its return value never
    # contains "APPROVE:" -> _parse_approval must default to False, not True.
    replies = [f"unparseable attempt {i}" for i in range(pn.MAX_TOOL_ITERATIONS)]

    class _Multi:
        def __init__(self, replies):
            self._replies = list(replies)
            self.calls = []

        def stream(self, messages):
            self.calls.append([m.content for m in messages])
            yield AIMessageChunk(content=self._replies.pop(0))

    multi = _Multi(replies)
    _install(monkeypatch, multi)

    result = pn.evaluator(_state(round=1))

    assert result.goto == "router"
    assert "final_output" not in result.update


def test_evaluator_can_self_check_via_a_tool_before_rendering_its_verdict(monkeypatch):
    class _Multi:
        def __init__(self, replies):
            self._replies = list(replies)
            self.calls = []

        def stream(self, messages):
            self.calls.append([m.content for m in messages])
            yield AIMessageChunk(content=self._replies.pop(0))

    multi = _Multi([
        "ACTION: execute_python\nCODE:\nprint(17 % 2, 17 % 3)",
        "FINAL:\nAPPROVE: yes\nWHY: verified no small factors divide it",
    ])
    _install(monkeypatch, multi)

    result = pn.evaluator(_state())

    assert result.goto == END
    tool_result = multi.calls[1][-1]
    assert "TOOL RESULT" in tool_result
