"""Coverage for the four specialist nodes (planner/solver/summarizer/
finder, agent/pipeline/nodes.py) -- all four are thin wrappers around one
shared implementation (_run_role), so this tests _run_role's branches once,
plus a thin check that each named wrapper actually reaches _run_role with
its own role/task/prompt rather than silently sharing another role's.

Every specialist now always returns to "router" (the overseer decides what
runs next, not the specialist itself) and self-reports its own identity
into state["node"] -- router() no longer predicts that ahead of time (see
test_router_node.py). The fresh-vs-revise signal is `state["node"] == role`
at dispatch time (I am the one whose attempt was rejected), not merely
"feedback is present" -- a DIFFERENT specialist picked up after a rejection
(the overseer re-dispatching, e.g. planner after a plan-less solver
failure) gets a fresh attempt with the rejected attempt shown only as
background, not the revise prompt.
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
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
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
    assert human == "TASK:\nsummarize this: blah blah blah"
    assert result.update["output"] == "a short summary"
    assert result.update["node"] == "summarizer"
    assert result.update["feedback"] == ""
    assert result.goto == "router"
    assert "summarizer" in result.update["board"][0]


def test_run_role_revise_branch_when_the_same_specialist_was_rejected(monkeypatch):
    fake = _FakeModel("FINAL:\na better summary")
    _install(monkeypatch, fake)

    result = pn._run_role(
        _state(node="summarizer", feedback="too long, cut it in half", output="a first, too-long summary"),
        role="summarizer", task=pn.Task.SUMMARIZE, temperature=0.2, prompt=pn.SUMMARIZER_PROMPT,
    )

    system, human = fake.calls[0]
    assert system == pn.ROLE_REVISE_PROMPT.format(role_upper="SUMMARIZER", max_iter=pn.MAX_TOOL_ITERATIONS)
    assert "YOUR PREVIOUS ATTEMPT" in human
    assert "a first, too-long summary" in human
    assert "too long, cut it in half" in human
    assert result.update["output"] == "a better summary"
    assert result.update["node"] == "summarizer"


def test_run_role_fresh_attempt_with_a_different_specialists_rejection_as_background(monkeypatch):
    # The overseer escalated from solver to planner after a rejection --
    # planner gets a FRESH attempt (its own role prompt, not the revise
    # prompt), but still sees the rejected attempt and feedback as context.
    fake = _FakeModel("FINAL:\n1. step one\n2. step two")
    _install(monkeypatch, fake)

    result = pn._run_role(
        _state(node="solver", feedback="jumped straight to code with no plan", output="def f(): ..."),
        role="planner", task=pn.Task.PLAN, temperature=0.4, prompt=pn.PLANNER_PROMPT,
    )

    system, human = fake.calls[0]
    assert system == pn.PLANNER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert "PREVIOUS ATTEMPT BY SOLVER" in human
    assert "def f(): ..." in human
    assert "jumped straight to code with no plan" in human
    assert result.update["node"] == "planner"


def test_run_role_shows_gathered_context_and_an_approved_plan_when_present(monkeypatch):
    fake = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake)

    pn._run_role(
        _state(context="found: uses pytest", plan="1. write it\n2. test it"),
        role="solver", task=pn.Task.REASON, temperature=0.5, prompt=pn.SOLVER_PROMPT,
    )

    _, human = fake.calls[0]
    assert "found: uses pytest" in human
    assert "1. write it\n2. test it" in human


def test_finder_appends_its_output_onto_existing_context(monkeypatch):
    fake = _FakeModel("FINAL:\nfound: the repo uses pytest")
    _install(monkeypatch, fake)

    result = pn.finder(_state(context="earlier: the repo is named otto"))

    assert result.update["context"] == "earlier: the repo is named otto\n\nfound: the repo uses pytest"


def test_finder_context_with_nothing_prior_is_just_its_own_output(monkeypatch):
    fake = _FakeModel("FINAL:\nfound: the repo uses pytest")
    _install(monkeypatch, fake)

    result = pn.finder(_state(context=""))

    assert result.update["context"] == "found: the repo uses pytest"


def test_summarizer_replaces_context_with_its_condensed_output(monkeypatch):
    fake = _FakeModel("FINAL:\na condensed version")
    _install(monkeypatch, fake)

    result = pn.summarizer(_state(context="a very long pile of gathered material"))

    assert result.update["context"] == "a condensed version"


def test_planner_and_solver_leave_context_untouched(monkeypatch):
    fake = _FakeModel("FINAL:\n1. step one")
    _install(monkeypatch, fake)
    result = pn.planner(_state(context="some context"))
    assert "context" not in result.update

    fake2 = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake2)
    result2 = pn.solver(_state(context="some context"))
    assert "context" not in result2.update


def test_planner_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\n1. step one\n2. step two")
    _install(monkeypatch, fake)

    result = pn.planner(_state())

    system, _ = fake.calls[0]
    assert system == pn.PLANNER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "1. step one\n2. step two"
    assert result.update["node"] == "planner"
    assert result.goto == "router"
    assert "planner" in result.update["board"][0]


def test_solver_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake)

    result = pn.solver(_state())

    system, _ = fake.calls[0]
    assert system == pn.SOLVER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "def f(): return 1"
    assert result.update["node"] == "solver"


def test_finder_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\naccording to my own knowledge, the answer is x")
    _install(monkeypatch, fake)

    result = pn.finder(_state())

    system, _ = fake.calls[0]
    assert system == pn.FINDER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == "according to my own knowledge, the answer is x"
    assert result.update["node"] == "finder"
