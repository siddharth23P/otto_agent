"""Coverage for the seventh refinement (agent/pipeline/nodes.py's module
docstring): a role node or the evaluator can pause the whole run and ask
the person something it's genuinely stuck without, via LangGraph's own
interrupt()/Command(resume=...) mechanism, instead of guessing or looping
("Solve N Queens with brute force" -> "improve above solution" getting
stuck trying to guess what "above solution" meant, live-tested the same
day -- see nodes.py's sixth refinement for the OTHER half of that fix).

Three pieces, tested separately here:

  * _parse_ask_user_body -- splitting an ask_user ACTION's CODE: body into
    (question, choices). Covered by test_role_nodes.py/test_evaluator_
    node.py too (via the full _run_role/evaluator path), but the parsing
    rules themselves (a CHOICES: line anywhere, case-insensitively, pipe-
    separated, empty entries dropped) are tested directly here.

  * _tool_loop raising NeedsUserInput on ACTION: ask_user, instead of
    dispatching it like any other tool through TOOL_DISPATCH -- the
    signal that unwinds all the way out to the dedicated ask_user node.

  * ask_user() itself -- the ONLY node that calls interrupt(), monkeypatched
    here (it needs a real LangGraph task context otherwise, which a bare
    unit test calling the node function directly doesn't have) so its own
    logic -- reading pending_question/pending_choices/asking_role off
    state, writing the Q&A into `context` (NOT `messages` -- see the
    node's own docstring for why), clearing all three pending fields, and
    handing back to whichever role asked -- can be checked without a real
    graph run.
"""
import pytest
from langchain_core.messages import HumanMessage

from agent.pipeline import nodes as pn


# --------------------------------------------------------------------------
# _parse_ask_user_body
# --------------------------------------------------------------------------

def test_parse_ask_user_body_with_no_choices_is_just_the_question():
    question, choices = pn._parse_ask_user_body("what output format do you want?")
    assert question == "what output format do you want?"
    assert choices == []


def test_parse_ask_user_body_splits_out_a_choices_line():
    question, choices = pn._parse_ask_user_body(
        "which language should this be in?\nCHOICES: python | rust | go"
    )
    assert question == "which language should this be in?"
    assert choices == ["python", "rust", "go"]


def test_parse_ask_user_body_choices_line_is_case_insensitive():
    question, choices = pn._parse_ask_user_body("pick one\nchoices: a | b")
    assert question == "pick one"
    assert choices == ["a", "b"]


def test_parse_ask_user_body_drops_empty_choice_entries():
    question, choices = pn._parse_ask_user_body("pick one\nCHOICES: a | | b |  ")
    assert choices == ["a", "b"]


def test_parse_ask_user_body_multiline_question_before_choices():
    question, choices = pn._parse_ask_user_body(
        "here is some context.\nwhat should I do next?\nCHOICES: x | y"
    )
    assert question == "here is some context.\nwhat should I do next?"
    assert choices == ["x", "y"]


# --------------------------------------------------------------------------
# _tool_loop raising NeedsUserInput on ACTION: ask_user
# --------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, reply: str):
        self._reply = reply
        self.calls: list[list[str]] = []

    def stream(self, messages):
        from langchain_core.messages import AIMessageChunk
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._reply)


def test_tool_loop_raises_needs_user_input_on_ask_user_action():
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhat should I call the function?")

    with pytest.raises(pn.NeedsUserInput) as excinfo:
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert excinfo.value.question == "what should I call the function?"
    assert excinfo.value.choices == []


def test_tool_loop_ask_user_only_makes_one_call_not_the_full_iteration_budget():
    # Confirms it's treated as an immediate unwind, not fed back into the
    # loop as a "tool result" and retried -- ask_user is not in
    # TOOL_DISPATCH, so falling through to the generic branch would have
    # produced a "tool not available" TOOL RESULT and kept looping instead.
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhich one?")

    with pytest.raises(pn.NeedsUserInput):
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert len(fake.calls) == 1


def test_tool_loop_ask_user_with_no_question_falls_back_to_a_placeholder():
    fake = _FakeModel("ACTION: ask_user\nCODE:\n")

    with pytest.raises(pn.NeedsUserInput) as excinfo:
        pn._tool_loop(fake, [HumanMessage("do the thing")])

    assert excinfo.value.question  # never empty -- something is always shown


# --------------------------------------------------------------------------
# ask_user() -- the dedicated node. interrupt() itself is monkeypatched:
# calling it for real needs a live LangGraph task context (get_config()
# raises outside one), which a bare unit test calling the node function
# directly doesn't have -- exactly like every other node test in this repo
# calls _run_role/router()/evaluator() directly rather than through a
# compiled graph.
# --------------------------------------------------------------------------

def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("solve N queens with brute force")],
        "board": [],
        "round": 3,
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
        "active_step": None,
        "node_error": None,
        "pending_question": "which N?",
        "pending_choices": [],
        "asking_role": "solver",
        "final_output": None,
    }
    base.update(overrides)
    return base


def test_ask_user_calls_interrupt_with_the_pending_question_and_choices(monkeypatch):
    seen = {}

    def fake_interrupt(payload):
        seen.update(payload)
        return "8"

    monkeypatch.setattr(pn, "interrupt", fake_interrupt)

    pn.ask_user(_state(pending_question="which N?", pending_choices=["4", "8"]))

    assert seen == {"question": "which N?", "choices": ["4", "8"]}


def test_ask_user_routes_back_to_whichever_role_asked(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state(asking_role="finder"))

    assert result.goto == "finder"


def test_ask_user_defaults_to_solver_if_asking_role_is_somehow_unset(monkeypatch):
    # Defensive fallback, same "fail toward a safe default" spirit as
    # _parse_router's own -- should never happen in practice (only
    # ask_user() itself clears asking_role, and only after reading it).
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8")

    result = pn.ask_user(_state(asking_role=None))

    assert result.goto == "solver"


def test_ask_user_writes_the_qa_into_context_not_messages(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "8x8")

    result = pn.ask_user(_state(pending_question="which N?", context=""))

    assert "which N?" in result.update["context"]
    assert "8x8" in result.update["context"]
    assert "messages" not in result.update  # task text must stay untouched


def test_ask_user_appends_onto_existing_context_rather_than_replacing_it(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "recursive")

    result = pn.ask_user(_state(context="earlier: the repo uses pytest", pending_question="recursive or iterative?"))

    assert "earlier: the repo uses pytest" in result.update["context"]
    assert "recursive or iterative?" in result.update["context"]
    assert "recursive" in result.update["context"]


def test_ask_user_clears_all_three_pending_fields(monkeypatch):
    monkeypatch.setattr(pn, "interrupt", lambda payload: "yes")

    result = pn.ask_user(_state())

    assert result.update["pending_question"] is None
    assert result.update["pending_choices"] is None
    assert result.update["asking_role"] is None


def test_ask_user_leaves_plan_and_active_step_untouched(monkeypatch):
    # A specialist that got stuck mid-plan-step must resume that same
    # step once answered -- ask_user() itself has no reason to touch
    # plan/active_step, only the role it hands back to does.
    monkeypatch.setattr(pn, "interrupt", lambda payload: "yes")
    plan = [{"task": "write it", "route_to": "solver", "output": None}]

    result = pn.ask_user(_state(plan=plan, active_step=0, asking_role="solver"))

    assert "plan" not in result.update
    assert "active_step" not in result.update


# --------------------------------------------------------------------------
# End to end through the REAL compiled graph (pn.app) -- interrupt()/
# Command(resume=...) is genuine LangGraph checkpointed-pause runtime
# machinery, not something a bare node-function call (everything above)
# can exercise on its own. No live LLM needed: every ROUTER.chat_model()
# call is monkeypatched to pop the next reply off a scripted queue, in the
# exact order this particular run is expected to make them -- the same
# "no live LLM" spirit as the rest of the offline suite, just also
# confirming the graph's own wiring (StateGraph edges, the checkpointer,
# app.compile()) actually pauses and resumes correctly, which nothing else
# in this suite touches.
# --------------------------------------------------------------------------

def _initial_state(text: str) -> dict:
    return {
        "messages": [HumanMessage(text)],
        "board": [],
        "round": 0,
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


def test_a_real_graph_run_pauses_on_ask_user_and_resumes_via_command(monkeypatch):
    import uuid

    queue = [
        "NODE: solver\nWHY: simple task",              # router's first 5-way decision
        "ACTION: ask_user\nCODE:\nhow many queens?",   # solver's _tool_loop -> pauses
        "FINAL:\n8-queens solution here",              # solver's fresh retry after resume
        "NODE: evaluator\nWHY: judge it",              # router again
        "FINAL:\nAPPROVE: yes\nWHY: looks right",      # evaluator
    ]

    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: _FakeModel(queue.pop(0)))

    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    initial = _initial_state("solve N queens with brute force")

    updates = list(pn.app.stream(initial, config, stream_mode="updates"))

    assert "__interrupt__" in updates[-1]
    triggered = updates[-1]["__interrupt__"][0]
    assert triggered.value == {"question": "how many queens?", "choices": []}
    # Nothing finished yet -- the checkpointed state still shows the pause.
    assert pn.app.get_state(config).values["final_output"] is None

    list(pn.app.stream(pn.Command(resume="8"), config, stream_mode="updates"))

    final_state = pn.app.get_state(config).values
    assert final_state["final_output"] == "8-queens solution here"
    # The pause itself leaves no trace once resumed.
    assert final_state["pending_question"] is None
    assert final_state["asking_role"] is None
    assert not queue  # every scripted reply was actually consumed, in order
