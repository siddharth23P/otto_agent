"""Coverage for router() itself (agent/pipeline/nodes.py) -- the overseer.
Re-invoked after every node, and makes its decision via _decide() (shared
by both of the prompts below), which retries its OWN call in place --
_MAX_ROUTER_PARSE_RETRIES times -- if a reply doesn't parse at all, before
falling back to "solver". See test_the_router_retries_an_unparseable_reply_*
below for that specifically; most tests here use a single-shot fake that
parses cleanly, so only one call happens.

  * ROUTER_PROMPT (the general, 5-way decision) -- used before any plan
    exists, or after a rejection (retry judgment). _router_body shows
    context/plan/pending-output-or-feedback.
  * STEP_ROUTE_PROMPT (a narrower, 3-way decision -- STEP_TARGETS) -- used
    while an approved, feedback-free plan is actively executing, to assign
    the next pending step to solver/summarizer/finder. When every step
    already has an output, router() skips the LLM call entirely and
    dispatches straight to evaluator -- that decision is unambiguous.

router() never predicts `state["node"]` ahead of time; whichever node runs
next self-reports its own identity (see test_role_nodes.py).

2026-09-10, fifth refinement: `state["node_error"]` is checked FIRST, ahead
of both the plan-execution shortcut and the general 5-way decision --
a provider/network failure means an LLM call just failed, so router()
does not make another one to decide what to do about it. It escalates to
planner deterministically instead, the same way the general prompt's own
"needed a plan after all" branch resets a stale plan.
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


class _MultiFakeModel:
    """Returns each reply in `replies` in turn, one per .stream() call --
    for exercising _decide()'s in-place retry loop, where the model's
    reply changes across attempts (unlike _FakeModel's fixed single reply).
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[str]] = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._replies.pop(0))


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
        "active_step": None,
        "node_error": None,
        "final_output": None,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# A provider/network failure -- checked before anything else, no LLM call.
# --------------------------------------------------------------------------

def test_a_node_error_escalates_straight_to_planner_with_no_llm_call(monkeypatch):
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    result = pn.router(_state(node_error="solver: inception: The read operation timed out"))

    assert result.goto == "planner"
    assert fake.calls == []
    assert result.update["node_error"] == ""
    assert "provider failure" in result.update["board"][0]
    assert "timed out" in result.update["board"][0]


def test_a_node_error_discards_an_active_plan(monkeypatch):
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    plan = [{"task": "step one", "route_to": "solver", "output": None}]
    result = pn.router(_state(node_error="evaluator: boom", plan=plan, active_step=0))

    assert result.goto == "planner"
    assert result.update["plan"] is None
    assert result.update["active_step"] is None


def test_a_node_error_with_no_plan_does_not_touch_plan_fields(monkeypatch):
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    result = pn.router(_state(node_error="solver: boom", plan=None))

    assert result.goto == "planner"
    assert "plan" not in result.update
    assert "active_step" not in result.update


def test_a_node_error_takes_priority_over_an_active_plans_step_assignment(monkeypatch):
    # Without the node_error check, this state (an active, feedback-free
    # plan with a pending step) would hit the plan-execution branch instead
    # -- node_error must win regardless of what else is going on.
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    plan = [{"task": "some step", "route_to": None, "output": None}]
    result = pn.router(_state(node_error="finder: boom", plan=plan, feedback=""))

    assert result.goto == "planner"
    assert fake.calls == []


# --------------------------------------------------------------------------
# General 5-way dispatch -- no active plan yet.
# --------------------------------------------------------------------------

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


def test_gathered_context_alone_is_shown_to_the_overseer_on_a_fresh_dispatch(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: context is ready")
    _install(monkeypatch, fake)

    pn.router(_state(context="found: the repo uses pytest"))

    _, human = fake.calls[0]
    assert "found: the repo uses pytest" in human


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

    result = pn.router(_state(node="planner", output='[{"task": "step one"}]', feedback=""))

    assert result.goto == "evaluator"


# --------------------------------------------------------------------------
# Rejection branch -- feedback is set, so the general 5-way call runs even
# if a plan is active (a rejection needs real judgment, not just the
# deterministic step-assignment path -- see the plan-execution section
# below).
# --------------------------------------------------------------------------

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


def test_a_rejection_while_a_plan_is_active_still_shows_the_plan_to_the_overseer(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: retry the last step")
    _install(monkeypatch, fake)

    plan = [
        {"task": "step one", "route_to": "finder", "output": "found it"},
        {"task": "step two", "route_to": "solver", "output": "def f(): return 1"},
    ]
    pn.router(_state(node="solver", output="def f(): return 1", plan=plan,
                      feedback="off by one", active_step=1))

    _, human = fake.calls[0]
    assert "step one" in human
    assert "step two" in human
    assert "off by one" in human


def test_escalating_to_planner_after_a_rejection_resets_a_stale_plan(monkeypatch):
    fake = _FakeModel("NODE: planner\nWHY: this needed a plan after all")
    _install(monkeypatch, fake)

    plan = [{"task": "step one", "route_to": "solver", "output": "bad answer"}]
    result = pn.router(_state(node="solver", output="bad answer", plan=plan,
                               feedback="jumped to code with no real plan", active_step=0))

    assert result.goto == "planner"
    assert result.update["plan"] is None
    assert result.update["active_step"] is None


def test_a_fresh_dispatch_to_planner_with_no_prior_plan_does_not_touch_plan_fields(monkeypatch):
    fake = _FakeModel("NODE: planner\nWHY: multi-step task")
    _install(monkeypatch, fake)

    result = pn.router(_state(plan=None))

    assert result.goto == "planner"
    assert "plan" not in result.update
    assert "active_step" not in result.update


# --------------------------------------------------------------------------
# Plan-step execution -- an approved, feedback-free plan is active.
# Deterministic except for the one narrow (STEP_TARGETS) choice.
# --------------------------------------------------------------------------

def test_plan_complete_dispatches_straight_to_evaluator_with_no_llm_call(monkeypatch):
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    plan = [
        {"task": "step one", "route_to": "finder", "output": "found it"},
        {"task": "step two", "route_to": "solver", "output": "def f(): return 1"},
    ]
    result = pn.router(_state(plan=plan, feedback=""))

    assert result.goto == "evaluator"
    assert fake.calls == []
    assert "plan complete" in result.update["board"][0]


def test_a_pending_step_with_no_route_to_gets_assigned_via_the_step_prompt(monkeypatch):
    fake = _FakeModel("NODE: finder\nWHY: needs a lookup first")
    _install(monkeypatch, fake)

    plan = [{"task": "look up the repo's test runner", "route_to": None, "output": None}]
    result = pn.router(_state(plan=plan, feedback=""))

    system, human = fake.calls[0]
    assert system == pn.STEP_ROUTE_PROMPT
    assert "look up the repo's test runner" in human
    assert "STEP TO ASSIGN (step 1)" in human

    assert result.goto == "finder"
    assert result.update["active_step"] == 0
    assert result.update["plan"][0]["route_to"] == "finder"
    # the original step dict is not mutated in place
    assert plan[0]["route_to"] is None


def test_the_step_prompt_only_offers_the_three_execution_specialists(monkeypatch):
    # A garbled/out-of-range reply falls back to "solver" -- STEP_TARGETS
    # never includes "planner" or "evaluator".
    fake = _FakeModel("NODE: evaluator\nWHY: looks done")
    _install(monkeypatch, fake)

    plan = [{"task": "some step", "route_to": None, "output": None}]
    result = pn.router(_state(plan=plan, feedback=""))

    assert result.goto == "solver"
    assert result.update["plan"][0]["route_to"] == "solver"


def test_a_step_already_assigned_a_route_to_is_dispatched_to_directly(monkeypatch):
    fake = _FakeModel("this should never be read")
    _install(monkeypatch, fake)

    plan = [{"task": "some step", "route_to": "summarizer", "output": None}]
    result = pn.router(_state(plan=plan, feedback=""))

    assert result.goto == "summarizer"
    assert result.update["active_step"] == 0
    assert fake.calls == []


def test_context_gathered_so_far_is_shown_during_step_assignment(monkeypatch):
    fake = _FakeModel("NODE: solver\nWHY: ready")
    _install(monkeypatch, fake)

    plan = [{"task": "some step", "route_to": None, "output": None}]
    pn.router(_state(plan=plan, feedback="", context="earlier finding: uses pytest"))

    _, human = fake.calls[0]
    assert "earlier finding: uses pytest" in human


# --------------------------------------------------------------------------
# _decide()'s in-place retry -- observed live (2026-09-10): the model
# occasionally rambles instead of the required NODE:/WHY: format, most
# often right when it should say "evaluator" for the first time. Retrying
# the SAME small call recovers this far more cheaply than silently
# defaulting to solver and burning a whole extra specialist round would.
# --------------------------------------------------------------------------

def test_an_unparseable_reply_is_retried_in_place_and_can_recover(monkeypatch):
    fake = _MultiFakeModel([
        "The candidate output looks complete and I'm now deciding if the task is finished.",
        "NODE: evaluator\nWHY: ready to judge",
    ])
    _install(monkeypatch, fake)

    result = pn.router(_state(node="solver", output="def f(): return 1", feedback=""))

    assert len(fake.calls) == 2
    assert result.goto == "evaluator"
    assert "evaluator" in result.update["board"][0]
    assert "could not parse" not in result.update["board"][0]


def test_the_retry_feedback_asks_for_exactly_two_lines_and_lists_valid_targets(monkeypatch):
    fake = _MultiFakeModel([
        "hmm, let me think about this for a moment",
        "NODE: solver\nWHY: retry succeeded",
    ])
    _install(monkeypatch, fake)

    pn.router(_state())

    retry_human = fake.calls[1][-1]  # the corrective HumanMessage appended before the 2nd call
    assert "EXACTLY two lines" in retry_human
    assert "planner" in retry_human and "evaluator" in retry_human


def test_exhausting_all_retries_falls_back_to_solver_with_an_explanatory_why(monkeypatch):
    fake = _MultiFakeModel([
        "rambling attempt one",
        "rambling attempt two",
        "rambling attempt three",
    ])
    _install(monkeypatch, fake)

    result = pn.router(_state())

    assert len(fake.calls) == pn._MAX_ROUTER_PARSE_RETRIES + 1
    assert result.goto == "solver"
    assert "3 attempt" in result.update["board"][0]


def test_step_assignment_retries_are_restricted_to_step_targets_in_the_feedback(monkeypatch):
    fake = _MultiFakeModel([
        "not sure who should do this",
        "NODE: finder\nWHY: needs a lookup",
    ])
    _install(monkeypatch, fake)

    plan = [{"task": "some step", "route_to": None, "output": None}]
    pn.router(_state(plan=plan, feedback=""))

    retry_human = fake.calls[1][-1]
    assert "solver, summarizer, finder" in retry_human
    assert "planner" not in retry_human
    assert "evaluator" not in retry_human
