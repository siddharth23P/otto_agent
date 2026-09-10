"""Coverage for the four specialist nodes (planner/solver/summarizer/
finder, agent/pipeline/nodes.py) -- all four are thin wrappers around one
shared implementation (_run_role), so this tests _run_role's own two
branches (fresh attempt vs. revise-with-feedback) once, plus a thin check
that each named wrapper actually reaches _run_role with its own role/task/
prompt rather than silently sharing another role's.
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
        "messages": [HumanMessage("summarize this: blah blah blah")],
        "board": [],
        "round": 1,
        "node": "summarizer",
        "feedback": "",
        "output": None,
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_run_role_fresh_attempt_uses_the_role_prompt_and_the_raw_task(monkeypatch):
    fake = _FakeModel("FINAL:\na short summary")
    _install(monkeypatch, fake)

    result = pn._run_role(_state(), role="summarizer", task=pn.Task.SUMMARIZE,
                           temperature=0.2, prompt=pn.SUMMARIZER_PROMPT)

    system, human = fake.calls[0]
    assert system == pn.SUMMARIZER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert human == "summarize this: blah blah blah"
    assert result.update["output"] == "a short summary"
    assert result.goto == "evaluator"
    assert "summarizer" in result.update["board"][0]


def test_run_role_revise_branch_uses_the_shared_revise_prompt_with_context(monkeypatch):
    fake = _FakeModel("FINAL:\na better summary")
    _install(monkeypatch, fake)

    result = pn._run_role(
        _state(feedback="too long, cut it in half", output="a first, too-long summary"),
        role="summarizer", task=pn.Task.SUMMARIZE, temperature=0.2, prompt=pn.SUMMARIZER_PROMPT,
    )

    system, human = fake.calls[0]
    assert system == pn.ROLE_REVISE_PROMPT.format(role_upper="SUMMARIZER", max_iter=pn.MAX_TOOL_ITERATIONS)
    assert "a first, too-long summary" in human
    assert "too long, cut it in half" in human
    assert result.update["output"] == "a better summary"


def test_planner_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\n1. step one\n2. step two")
    _install(monkeypatch, fake)

    result = pn.planner(_state(node="planner"))

    system, _ = fake.calls[0]
    assert system == pn.PLANNER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "1. step one\n2. step two"
    assert "planner" in result.update["board"][0]


def test_solver_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake)

    result = pn.solver(_state(node="solver"))

    system, _ = fake.calls[0]
    assert system == pn.SOLVER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "def f(): return 1"


def test_finder_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\naccording to my own knowledge, the answer is x")
    _install(monkeypatch, fake)

    result = pn.finder(_state(node="finder"))

    system, _ = fake.calls[0]
    assert system == pn.FINDER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "according to my own knowledge, the answer is x"
