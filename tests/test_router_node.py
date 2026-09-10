"""Coverage for router() itself (agent/pipeline/nodes.py) -- the single
classification call that replaces the swarm pipeline's whole plan-
negotiation loop (orchestrator_propose/orchestrator_review/
orchestrator_consensus). No vote, no tool use: one call in, one Command
dispatching to exactly one specialist out.

Round 1 (no feedback yet) and a retry round (feedback from a prior
evaluator rejection) use different prompts (ROUTER_PROMPT vs.
ROUTER_RETRY_PROMPT) -- both are exercised here, since picking the wrong
one silently would still produce a syntactically valid Command and only
show up as worse routing decisions in practice, never a crash.
"""
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
        "messages": [HumanMessage("write a function that checks if a number is prime")],
        "board": [],
        "round": 0,
        "node": None,
        "feedback": "",
        "output": None,
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_router_dispatches_to_the_node_the_model_named(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: needs code")
    _install(monkeypatch, fake)

    result = pn.router(_state())

    assert result.goto == "solver"
    assert result.update["node"] == "solver"
    assert result.update["round"] == 1
    assert "solver" in result.update["board"][0]


def test_router_falls_back_to_solver_on_an_unparseable_reply(monkeypatch):
    fake = _FakeModel("uh, I'm not sure, just try something")
    _install(monkeypatch, fake)

    result = pn.router(_state())

    assert result.goto == "solver"


def test_round_one_uses_the_plain_router_prompt_with_just_the_task(monkeypatch):
    fake = _FakeModel("NODE: planner\nWHY: multi-step")
    _install(monkeypatch, fake)

    pn.router(_state())

    system, human = fake.calls[0]
    assert system == pn.ROUTER_PROMPT
    assert human == "write a function that checks if a number is prime"


def test_retry_round_uses_the_retry_prompt_with_feedback_and_previous_attempt(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: try again with fixes")
    _install(monkeypatch, fake)

    result = pn.router(_state(
        round=1, node="planner", output="1. do a thing",
        feedback="the plan never actually solves the problem",
    ))

    system, human = fake.calls[0]
    assert system == pn.ROUTER_RETRY_PROMPT
    assert "planner" in human
    assert "1. do a thing" in human
    assert "the plan never actually solves the problem" in human
    assert result.update["round"] == 2
    assert result.goto == "solver"
