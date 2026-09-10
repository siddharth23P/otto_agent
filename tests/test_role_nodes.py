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

A separate branch -- "executing a plan step" (state["active_step"] points
into state["plan"]) -- always writes the output back into THAT step (not
just the flat state["output"]) and always APPENDS a labeled step summary
onto context, regardless of the role's own normal context_op (summarizer's
usual "replace" would erase earlier steps' results mid-plan, defeating the
whole point of sequencing them).
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
        "active_step": None,
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
    fake = _FakeModel("FINAL:\n[{\"task\": \"step one\"}, {\"task\": \"step two\"}]")
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


def test_run_role_shows_the_plan_when_present(monkeypatch):
    fake = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake)

    plan = [{"task": "write it", "route_to": "solver", "output": None}]
    pn._run_role(
        _state(context="found: uses pytest", plan=plan, active_step=0),
        role="solver", task=pn.Task.REASON, temperature=0.5, prompt=pn.SOLVER_PROMPT,
    )

    _, human = fake.calls[0]
    assert "found: uses pytest" in human
    assert "write it" in human
    assert "YOUR CURRENT STEP (step 1)" in human


def test_finder_appends_its_output_onto_existing_context_outside_a_plan(monkeypatch):
    fake = _FakeModel("FINAL:\nfound: the repo uses pytest")
    _install(monkeypatch, fake)

    result = pn.finder(_state(context="earlier: the repo is named otto"))

    assert result.update["context"] == "earlier: the repo is named otto\n\nfound: the repo uses pytest"
    assert "plan" not in result.update


def test_finder_context_with_nothing_prior_is_just_its_own_output(monkeypatch):
    fake = _FakeModel("FINAL:\nfound: the repo uses pytest")
    _install(monkeypatch, fake)

    result = pn.finder(_state(context=""))

    assert result.update["context"] == "found: the repo uses pytest"


def test_summarizer_replaces_context_with_its_condensed_output_outside_a_plan(monkeypatch):
    fake = _FakeModel("FINAL:\na condensed version")
    _install(monkeypatch, fake)

    result = pn.summarizer(_state(context="a very long pile of gathered material"))

    assert result.update["context"] == "a condensed version"


def test_planner_and_solver_leave_context_untouched_outside_a_plan(monkeypatch):
    fake = _FakeModel("FINAL:\n[{\"task\": \"step one\"}]")
    _install(monkeypatch, fake)
    result = pn.planner(_state(context="some context"))
    assert "context" not in result.update

    fake2 = _FakeModel("FINAL:\ndef f(): return 1")
    _install(monkeypatch, fake2)
    result2 = pn.solver(_state(context="some context"))
    assert "context" not in result2.update


def test_planner_wrapper_uses_its_own_role_task_and_prompt(monkeypatch):
    fake = _FakeModel("FINAL:\n[{\"task\": \"step one\"}, {\"task\": \"step two\"}]")
    _install(monkeypatch, fake)

    result = pn.planner(_state())

    system, _ = fake.calls[0]
    assert system == pn.PLANNER_PROMPT.format(max_iter=pn.MAX_TOOL_ITERATIONS)
    assert result.update["output"] == '[{"task": "step one"}, {"task": "step two"}]'
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


# --------------------------------------------------------------------------
# Executing a plan step (state["active_step"] points into state["plan"]) --
# writes back into that step, always APPENDS to context regardless of the
# role's own normal context_op.
# --------------------------------------------------------------------------

def test_executing_a_step_writes_the_output_into_that_steps_slot(monkeypatch):
    fake = _FakeModel("FINAL:\nfound it: pytest")
    _install(monkeypatch, fake)

    plan = [
        {"task": "look up the test runner", "route_to": "finder", "output": None},
        {"task": "write the test", "route_to": None, "output": None},
    ]
    result = pn.finder(_state(plan=plan, active_step=0, node="finder"))

    assert result.update["plan"][0]["output"] == "found it: pytest"
    assert result.update["plan"][1]["output"] is None  # untouched
    # the original list/dicts are not mutated in place
    assert plan[0]["output"] is None


def test_executing_a_step_appends_a_labeled_summary_to_context(monkeypatch):
    fake = _FakeModel("FINAL:\nfound it: pytest")
    _install(monkeypatch, fake)

    plan = [{"task": "look up the test runner", "route_to": "finder", "output": None}]
    result = pn.finder(_state(plan=plan, active_step=0, context="earlier: repo is named otto"))

    assert "earlier: repo is named otto" in result.update["context"]
    assert "step 1 (finder)" in result.update["context"]
    assert "look up the test runner" in result.update["context"]
    assert "found it: pytest" in result.update["context"]


def test_a_summarizer_step_still_appends_rather_than_replacing_context(monkeypatch):
    # summarizer's own normal context_op is "replace" -- but while executing
    # a plan step, every role always APPENDS, so earlier steps' results
    # survive a summarizer step instead of being wiped out.
    fake = _FakeModel("FINAL:\ncondensed: uses pytest, no CI configured")
    _install(monkeypatch, fake)

    plan = [
        {"task": "look things up", "route_to": "finder", "output": "raw: uses pytest, raw: no CI configured, ..."},
        {"task": "condense the findings", "route_to": "summarizer", "output": None},
    ]
    result = pn.summarizer(_state(
        plan=plan, active_step=1,
        context="step 1 (finder): look things up\n-> raw: uses pytest, raw: no CI configured, ...",
    ))

    assert "step 1 (finder)" in result.update["context"]
    assert "step 2 (summarizer)" in result.update["context"]
    assert "condensed: uses pytest, no CI configured" in result.update["context"]


def test_revising_the_active_step_after_a_rejection_overwrites_its_output(monkeypatch):
    fake = _FakeModel("FINAL:\ndef f(): return 2  # fixed")
    _install(monkeypatch, fake)

    plan = [{"task": "write the function", "route_to": "solver", "output": "def f(): return 1  # buggy"}]
    result = pn.solver(_state(
        plan=plan, active_step=0, node="solver", output="def f(): return 1  # buggy",
        feedback="off by one",
    ))

    system, human = fake.calls[0]
    assert system == pn.ROLE_REVISE_PROMPT.format(role_upper="SOLVER", max_iter=pn.MAX_TOOL_ITERATIONS)
    assert "def f(): return 1  # buggy" in human
    assert "off by one" in human
    assert result.update["plan"][0]["output"] == "def f(): return 2  # fixed"


def test_active_step_out_of_range_or_without_a_list_plan_is_not_treated_as_executing_a_step(monkeypatch):
    fake = _FakeModel("FINAL:\na short summary")
    _install(monkeypatch, fake)

    # No plan at all, but active_step is (stale/defensive) set -- must not
    # crash trying to index into a None plan.
    result = pn.summarizer(_state(plan=None, active_step=0))

    assert "plan" not in result.update
    assert result.update["context"] == "a short summary"  # normal "replace" behavior applies
