"""Coverage for evaluator() (agent/pipeline/nodes.py) -- dual-mode judge,
with the same ACTION/FINAL tool-calling loop every role node has
(_tool_loop). It judges a PLAN (state["node"] == "planner") or a candidate
FINAL ANSWER (anything else) -- different question, different prompt.

Approving a plan PARSES it (_parse_plan_steps) into a list of PlanStep
dicts (agent/pipeline/state.py) and hands back to the overseer with
active_step cleared -- there's more work left, the plan hasn't been
executed yet. Approving a final answer ends the run. Rejecting either
always goes back to the overseer with the reason -- there is no round cap
and no exhaustion branch anymore (2026-09-10 design call: "we dont need
any variable to limit number of rounds a agent runs for").

_parse_approval defaults to approve=False whenever _tool_loop's reply
never contained an "APPROVE:" line at all (e.g. it exhausted on
unparseable replies) -- fails CLOSED by construction. That default is
exercised here too: an evaluator that never rendered a real verdict must
not be mistaken for one that approved.

2026-09-10, fifth refinement: a provider/network failure (_tool_loop
raising ProviderError instead of returning) is a THIRD outcome, distinct
from approve/reject -- evaluator() returns to router() with
state["node_error"] set rather than crashing or being mistaken for either
verdict.
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


class _FailingModel:
    """Raises instead of ever producing a reply -- models a provider/
    network failure already normalised to ProviderError by the provider
    layer (agent/router/llm_provider/inception_provider.py).
    """

    def __init__(self, exc: Exception):
        self._exc = exc
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        raise self._exc


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


def test_evaluator_approves_a_final_answer_and_ends_the_graph(monkeypatch):
    fake = _FakeModel("FINAL:\nAPPROVE: yes\nWHY: correct and verified")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state())

    assert result.goto == END
    assert result.update["final_output"] == "yes, 17 is prime"


def test_evaluator_judges_a_final_answer_using_the_final_answer_framing(monkeypatch):
    # The framing used to name the role that produced the output, because five
    # different nodes could. One loop produces every answer now, so there is one
    # framing and the label is just ANSWER.
    fake = _FakeModel("FINAL:\nAPPROVE: yes\nWHY: fine")
    _install(monkeypatch, fake)

    pn.evaluator(_state(node="agent", output="def f(): return 1"))

    system, human = fake.calls[0]
    assert "ANSWER" in system
    assert "as a finished answer" in system
    assert "ANSWER:" in human
    assert "def f(): return 1" in human


def test_the_evaluator_no_longer_judges_blind(monkeypatch):
    """It used to see the conversation, the request and the answer, and nothing
    else -- so "I cannot verify this" came back as a rejection, and every
    rejection cost a whole extra round. Everything below already existed and
    was already bounded; it was simply never shown to the judge."""
    fake = _FakeModel("FINAL:\nAPPROVE: yes\nWHY: fine")
    _install(monkeypatch, fake)

    pn.evaluator(_state(
        node="agent",
        output="the suite passes",
        actions=["solve: execute_bash pytest -> ok (0 failed)"],
        mode_log=["call 3: solve -> plan (needs ordering)"],
        transcript=[{"kind": "human", "content": "TOOL RESULT:\nstdout:\n0 failed"}],
        context="the failing test was test_auth",
    ))

    _, human = fake.calls[0]
    assert "pytest -> ok" in human, "the judge cannot see what was run"
    assert "solve -> plan" in human, "the judge cannot see how the run worked"
    assert "0 failed" in human, "the judge cannot see the evidence"
    assert "test_auth" in human, "the judge cannot see the gathered context"


def test_a_third_rejection_lets_the_answer_stand_rather_than_spending_the_budget(monkeypatch):
    """Judgment is worth paying for; judgment without a bound is a way to spend
    a whole run re-reading one answer. The board says it was not verified."""
    fake = _FakeModel("FINAL:\nAPPROVE: no\nWHY: still not right")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(node="agent", output="best effort", rejections=2))

    assert result.goto == END
    assert result.update["final_output"] == "best effort"
    assert "unverified" in " ".join(result.update["board"])


def test_evaluator_shows_prior_conversation_ahead_of_the_original_request(monkeypatch):
    # 2026-09-10, sixth refinement -- the evaluator needs the same
    # conversation context as everyone else to judge whether an "improve
    # above solution"-shaped answer actually improved the right thing.
    from langchain_core.messages import AIMessage

    fake = _FakeModel("FINAL:\nAPPROVE: yes\nWHY: fine")
    _install(monkeypatch, fake)

    state = _state(node="solver", output="def f(): return 2  # improved", messages=[
        HumanMessage("solve N queens with brute force"),
        AIMessage("def solve(n): ..."),
        HumanMessage("improve above solution"),
    ])
    pn.evaluator(state)

    _, human = fake.calls[0]
    assert human.startswith("CONVERSATION SO FAR:\n")
    assert "you: solve N queens with brute force" in human
    assert "otto: def solve(n): ..." in human
    assert "ORIGINAL REQUEST:\nimprove above solution" in human


def test_evaluator_rejects_a_final_answer_and_routes_feedback_back_to_the_loop(monkeypatch):
    fake = _FakeModel("FINAL:\nAPPROVE: no\nWHY: never checked divisibility")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(round=1))

    assert result.goto == "agent"
    assert result.update["feedback"] == "never checked divisibility"
    assert "final_output" not in result.update


def test_evaluator_rejects_repeatedly_with_no_round_cap(monkeypatch):
    # No MAX_DISPATCH_ROUNDS anymore -- a rejection always goes back to the
    # overseer regardless of how many rounds have already happened.
    fake = _FakeModel("FINAL:\nAPPROVE: no\nWHY: still not great")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(round=500))

    assert result.goto == "agent"
    assert "final_output" not in result.update


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

    assert result.goto == "agent"
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


# --------------------------------------------------------------------------
# Getting stuck and asking the user (2026-09-10, seventh refinement) --
# _tool_loop raises NeedsUserInput when a reply's ACTION: is ask_user.
# evaluator() hands off to the dedicated ask_user node -- a THIRD outcome
# again, distinct from approve/reject/node_error, and `output` (still
# pending judgment) is left untouched.
# --------------------------------------------------------------------------

def test_evaluator_asks_the_user_and_pauses_instead_of_guessing_a_verdict(monkeypatch):
    fake = _FakeModel("ACTION: ask_user\nCODE:\ndid you want it recursive or iterative?")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(node="solver", output="def f(): return 1"))

    assert result.goto == "ask_user"
    assert result.update["pending_question"] == "did you want it recursive or iterative?"
    assert result.update["pending_choices"] == []
    assert result.update["asking_role"] == "evaluator"
    assert "final_output" not in result.update
    assert "feedback" not in result.update


def test_evaluator_asking_with_choices_parses_them_out(monkeypatch):
    fake = _FakeModel("ACTION: ask_user\nCODE:\nwhich one is right?\nCHOICES: option a | option b")
    _install(monkeypatch, fake)

    result = pn.evaluator(_state())

    assert result.update["pending_choices"] == ["option a", "option b"]


# --------------------------------------------------------------------------
# A provider/network failure mid-judgment (2026-09-10, fifth refinement) --
# _tool_loop raises ProviderError instead of returning. evaluator() returns
# to router() with node_error set rather than crashing, or being mistaken
# for either an approval or a rejection.
# --------------------------------------------------------------------------

def test_a_provider_failure_returns_to_the_loop_with_node_error_set_instead_of_crashing(monkeypatch):
    fake = _FailingModel(pn.ProviderError("inception: The read operation timed out"))
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(node="solver", output="def f(): return 1"))

    assert result.goto == "agent"
    assert "evaluator" in result.update["node_error"]
    assert "timed out" in result.update["node_error"]
    assert result.goto != END
    assert "final_output" not in result.update


def test_a_provider_failure_writes_feedback_naming_whose_output_it_was_judging(monkeypatch):
    fake = _FailingModel(pn.ProviderError("inception: The read operation timed out"))
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(node="solver", output="def f(): return 1"))

    assert "solver" in result.update["feedback"]
    assert "not a real rejection" in result.update["feedback"]


def test_a_provider_failure_judging_a_plan_says_plan_not_output_in_the_feedback(monkeypatch):
    fake = _FailingModel(pn.ProviderError("boom"))
    _install(monkeypatch, fake)

    result = pn.evaluator(_state(node="planner", output='[{"task": "step one"}]'))

    assert "planner" in result.update["feedback"]
    assert "plan" in result.update["feedback"]
    assert "output" not in result.update  # untouched, still pending judgment
