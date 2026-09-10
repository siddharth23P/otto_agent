"""End-to-end coverage for the structured plan-execution model (2026-09-10,
third refinement to agent/pipeline/nodes.py): once a plan is approved, it
is a list of `{"task", "route_to", "output"}` step dicts (state.py's
PlanStep), `task` written by the planner and `route_to` filled in by the
overseer one step at a time as it assigns each step -- not the router
picking the whole task's specialist once, and not free-text plans the next
specialist had to interpret whole.

This drives router() -> a role node -> router() -> ... across multiple
turns (unlike test_router_node.py/test_role_nodes.py, which test each node
function in isolation) to verify the handoff actually works end to end:
does a later step really see an earlier step's result, does the plan
converge to evaluator once every step is done, does a rejection revise the
right step.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.pipeline import nodes as pn


class _FakeModel:
    """Replies with one fixed string every call, recording the messages it
    was shown -- same shape as the other node test files' fake, used here
    to drive several nodes across several turns without a real LLM.
    """

    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[list[str]] = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._reply)


def _install(monkeypatch, reply: str) -> _FakeModel:
    fake = _FakeModel(reply)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    return fake


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("look up the test runner, then write a test for it")],
        "board": [],
        "round": 2,
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
        "active_step": None,
        "node_error": None,
        "pending_question": None,
        "pending_choices": None,
        "asking_role": None,
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_a_freshly_approved_plan_gets_its_first_step_assigned_and_executed(monkeypatch):
    plan = [
        {"task": "find the repo's test runner", "route_to": None, "output": None},
        {"task": "write a test using it", "route_to": None, "output": None},
    ]
    state = _state(plan=plan)

    # router() assigns step 1 to a specialist -- no route_to yet, so it
    # makes the narrower STEP_ROUTE_PROMPT call.
    route_fake = _install(monkeypatch, "NODE: finder\nWHY: needs a lookup")
    routed = pn.router(state)
    assert routed.goto == "finder"
    assert routed.update["active_step"] == 0
    state = {**state, **routed.update}

    # finder executes step 1 -- writes into plan[0]["output"], appends to
    # context, self-reports node="finder", goes back to router.
    exec_fake = _install(monkeypatch, "FINAL:\nfound it: pytest")
    ran = pn.finder(state)
    assert ran.goto == "router"
    state = {**state, **ran.update}
    assert state["plan"][0]["output"] == "found it: pytest"
    assert state["plan"][1]["output"] is None
    assert "step 1 (finder)" in state["context"]

    # router() now assigns step 2 -- and the step-assignment prompt shows
    # step 1's result via the formatted plan.
    route_fake2 = _install(monkeypatch, "NODE: solver\nWHY: write the test")
    routed2 = pn.router(state)
    assert routed2.goto == "solver"
    assert routed2.update["active_step"] == 1
    _, human = route_fake2.calls[0]
    assert "found it: pytest" in human  # step 1's result is visible
    state = {**state, **routed2.update}

    # solver executes step 2, using step 1's result via context.
    exec_fake2 = _install(monkeypatch, "FINAL:\ndef test_thing(): assert True")
    _, _ = exec_fake, route_fake  # (silence unused-var lint; kept for clarity of the sequence)
    ran2 = pn.solver(state)
    _, human2 = exec_fake2.calls[0]
    assert "found it: pytest" in human2  # step 2's specialist sees step 1's result via context
    state = {**state, **ran2.update}
    assert state["plan"][1]["output"] == "def test_thing(): assert True"

    # every step now has output -- router() dispatches straight to
    # evaluator with no further LLM call.
    no_call_fake = _install(monkeypatch, "should never be read")
    final_route = pn.router(state)
    assert final_route.goto == "evaluator"
    assert no_call_fake.calls == []


def test_a_rejected_final_answer_revises_the_last_step_not_a_fresh_attempt(monkeypatch):
    plan = [{"task": "write the function", "route_to": "solver", "output": "def f(): return 1  # buggy"}]
    state = _state(plan=plan, active_step=0, node="solver", output="def f(): return 1  # buggy")

    # evaluator rejects the final answer.
    _install(monkeypatch, "FINAL:\nAPPROVE: no\nWHY: off by one")
    rejected = pn.evaluator(state)
    assert rejected.goto == "router"
    state = {**state, **rejected.update}
    assert state["feedback"] == "off by one"

    # router(), with feedback set, uses the general 5-way prompt (not the
    # deterministic step-assignment path) and defaults to retrying solver.
    route_fake = _install(monkeypatch, "NODE: solver\nWHY: retry with the fix in mind")
    routed = pn.router(state)
    assert routed.goto == "solver"
    assert "plan" not in routed.update or routed.update.get("plan") == plan  # not reset
    state = {**state, **routed.update}

    # solver revises -- since active_step still points at the same step
    # solver produced, the fix overwrites that step's output in place.
    revise_fake = _install(monkeypatch, "FINAL:\ndef f(): return 2  # fixed")
    revised = pn.solver(state)
    system, human = revise_fake.calls[0]
    assert system == pn.ROLE_REVISE_PROMPT.format(role_upper="SOLVER", max_iter=pn.MAX_TOOL_ITERATIONS)
    assert "off by one" in human
    assert revised.update["plan"][0]["output"] == "def f(): return 2  # fixed"


def test_evaluator_approving_a_plan_hands_a_ready_to_execute_plan_to_the_router(monkeypatch):
    state = _state(node="planner", output='[{"task": "step one"}, {"task": "step two"}]')

    _install(monkeypatch, "FINAL:\nAPPROVE: yes\nWHY: sound and complete")
    approved = pn.evaluator(state)
    assert approved.goto == "router"
    state = {**state, **approved.update}
    assert state["plan"] == [
        {"task": "step one", "route_to": None, "output": None},
        {"task": "step two", "route_to": None, "output": None},
    ]
    assert state["active_step"] is None
    assert state["output"] is None

    # router() picks up the fresh plan and assigns its first step.
    route_fake = _install(monkeypatch, "NODE: solver\nWHY: works out step one")
    routed = pn.router(state)
    assert routed.goto == "solver"
    assert routed.update["active_step"] == 0
    assert routed.update["plan"][0]["route_to"] == "solver"
