"""Coverage for router() itself (agent/pipeline/nodes.py) -- the overseer.
Re-invoked after every node (not just after a rejection), one call, no
tools: it reads everything accumulated so far (context/plan/pending output
or feedback, via _router_body) and picks exactly one of five targets
(planner, solver, summarizer, finder, evaluator) to run next.

Unlike the first revision, there is only ONE router prompt now (ROUTER_PROMPT)
-- round 1 and a retry round differ only in what _router_body puts in the
human message, not in which system prompt is used. router() also no longer
predicts `state["node"]` ahead of time; whichever node runs next self-reports
its own identity (see test_role_nodes.py).
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
        "context": "",
        "plan": None,
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_router_dispatches_to_the_node_the_model_named(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: needs code")
    _install(monkeypatch, fake)

    result = pn.router(_state())

    assert result.goto == "solver"
    assert result.update["round"] == 1
    assert "solver" in result.update["board"][0]
    # router() no longer predicts `node` ahead of time -- whichever node
    # runs next self-reports its own identity (see test_role_nodes.py).
    assert "node" not in result.update


def test_router_falls_back_to_solver_on_an_unparseable_reply(monkeypatch):
    fake = _FakeModel("uh, I'm not sure, just try something")
    _install(monkeypatch, fake)

    result = pn.router(_state())

    assert result.goto == "solver"


def test_round_one_uses_just_the_task_with_no_context_plan_or_feedback(monkeypatch):
    fake = _FakeModel("NODE: planner\nWHY: multi-step")
    _install(monkeypatch, fake)

    pn.router(_state())

    system, human = fake.calls[0]
    assert system == pn.ROUTER_PROMPT
    assert human == "TASK:\nwrite a function that checks if a number is prime"


def test_a_rejection_shows_the_same_prompt_with_the_previous_attempt_and_feedback(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: try again with fixes")
    _install(monkeypatch, fake)

    result = pn.router(_state(
        round=1, node="planner", output="1. do a thing",
        feedback="the plan never actually solves the problem",
    ))

    system, human = fake.calls[0]
    assert system == pn.ROUTER_PROMPT
    assert "planner" in human
    assert "1. do a thing" in human
    assert "the plan never actually solves the problem" in human
    assert result.update["round"] == 2
    assert result.goto == "solver"


def test_gathered_context_and_an_approved_plan_are_both_shown_to_the_overseer(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: plan and context are ready")
    _install(monkeypatch, fake)

    pn.router(_state(context="found: the repo uses pytest", plan="1. write the function\n2. test it"))

    _, human = fake.calls[0]
    assert "found: the repo uses pytest" in human
    assert "1. write the function\n2. test it" in human


def test_a_pending_unjudged_output_is_shown_but_not_confused_with_a_rejection(monkeypatch):
    fake = _FakeModel("NODE: evaluator\nWHY: ready to judge")
    _install(monkeypatch, fake)

    pn.router(_state(node="solver", output="def f(): return 1", feedback=""))

    _, human = fake.calls[0]
    assert "PENDING OUTPUT (from solver, not yet judged)" in human
    assert "def f(): return 1" in human
    assert "REJECTED ATTEMPT" not in human


def test_choosing_evaluator_with_nothing_pending_judgment_falls_back_to_solver(monkeypatch):
    fake = _FakeModel("NODE: evaluator\nWHY: seems done")
    _install(monkeypatch, fake)

    result = pn.router(_state(node=None, output=None, feedback=""))

    assert result.goto == "solver"
    assert "nothing pending judgment" in result.update["board"][0]


def test_choosing_evaluator_is_honored_when_something_is_actually_pending(monkeypatch):
    fake = _FakeModel("NODE: evaluator\nWHY: plan is ready to judge")
    _install(monkeypatch, fake)

    result = pn.router(_state(node="planner", output="1. step one", feedback=""))

    assert result.goto == "evaluator"
