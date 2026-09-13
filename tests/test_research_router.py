"""The entry router: one rubric reply says chat, research or agent.

The research route is the expensive one and the one with no rejection to
recover from a misroute cheaply, so every ambiguity here reads as agent.
"""

from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.pipeline import nodes as pn
from agent.pipeline.workspace import bind_workspace
from agent.router.llm_provider.base import ProviderError


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "FINAL:\ndone"
        yield AIMessageChunk(content=reply)


class _Failing:
    def __init__(self, exc):
        self._exc = exc

    def stream(self, messages):
        raise self._exc


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("write a ten-part history of the empire")],
        "board": [], "node": None, "feedback": "", "output": None,
        "context": "", "node_error": None, "pending_question": None,
        "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [],
        "model_calls": 0, "rejections": 0, "checklist": None,
        "route": None, "document_path": None,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# parsing the rubric reply
# --------------------------------------------------------------------------

def test_a_kind_research_line_is_read_as_research():
    rubric = pn._rubric(_Scripted([
        "KIND: research\n- ten generations, each written out\n- each narrative at least 500 words"
    ]), "the task")

    assert rubric.kind == "research"
    assert rubric.route == "research"
    assert len(rubric.criteria) == 2
    assert not rubric.conversational


def test_a_missing_kind_line_defaults_to_agent():
    """Every scripted rubric reply in the older tests has no KIND line. They
    must keep meaning what they meant."""
    rubric = pn._rubric(_Scripted(["- the sum is correct"]), "add 2 and 2")

    assert rubric.kind == "agent"
    assert rubric.route == "agent"
    assert rubric.criteria == ["the sum is correct"]


def test_kind_research_with_no_criteria_is_agent():
    """Nothing to build a document from. The safe reading of a contradiction
    is the path a rejection can recover."""
    rubric = pn._rubric(_Scripted(["KIND: research"]), "the task")

    assert rubric.kind == "agent"
    assert rubric.criteria == []
    assert not rubric.conversational


def test_a_failed_rubric_call_is_never_research():
    rubric = pn._rubric(_Failing(ProviderError("down")), "write a book")

    assert rubric == pn.Rubric([])
    assert rubric.kind == "agent"


def test_a_bulleted_kind_line_is_not_a_criterion():
    rubric = pn._rubric(_Scripted(["- KIND: research\n- every section is present"]), "t")

    assert rubric.kind == "research"
    assert rubric.criteria == ["every section is present"]


def test_the_kind_is_read_case_insensitively_from_any_line():
    rubric = pn._rubric(_Scripted(["- covers it\nkind: Research"]), "t")

    assert rubric.kind == "research"


def test_no_task_still_wins():
    rubric = pn._rubric(_Scripted(["NO TASK"]), "hi")

    assert rubric.conversational
    assert rubric.route == "chat"


def test_a_criterion_mentioning_research_does_not_switch_the_route():
    rubric = pn._rubric(_Scripted(["- the research notes are cited"]), "t")

    assert rubric.kind == "agent"


def test_the_rubric_prompt_asks_for_the_kind_and_for_counts():
    assert "KIND: research" in pn.RUBRIC_PROMPT
    assert "KIND: agent" in pn.RUBRIC_PROMPT
    assert "countable" in pn.RUBRIC_PROMPT
    # The older rule survives: coverage is part of the answer.
    assert "COVERAGE" in pn.RUBRIC_PROMPT


# --------------------------------------------------------------------------
# dispatch from the agent node
# --------------------------------------------------------------------------

def test_a_research_task_is_dispatched_to_the_research_node(monkeypatch, tmp_path):
    fake = _Scripted([])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["ten parts"], kind="research"))

    with bind_workspace(str(tmp_path)):
        result = pn.agent(_state())

    assert result.goto == "research"
    assert result.update["route"] == "research"
    assert result.update["checklist"][0]["text"] == "ten parts"
    assert fake.calls == 0, "the loop must not have started"


def test_research_needs_a_workspace(monkeypatch):
    """No workspace, nowhere for a document to live: the loop takes it."""
    fake = _Scripted(["FINAL:\nhere it is"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["ten parts"], kind="research"))

    result = pn.agent(_state())

    assert result.goto == "evaluator"
    assert result.update["route"] == "agent"
    assert fake.calls == 1


def test_a_resumed_turn_never_reroutes(monkeypatch, tmp_path):
    """A turn with an agent conversation behind it stays with the agent, even
    if the rubric (rewritten after an answer) now says research."""
    fake = _Scripted(["FINAL:\ncarried on"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["ten parts"], kind="research"))
    stored = pn._plain([HumanMessage("TASK:\nwrite it"), pn.AIMessage("ACTION: ask_user\nCODE:\nhow long?")])

    with bind_workspace(str(tmp_path)):
        result = pn.agent(_state(transcript=stored, user_answer="ten parts",
                                 checklist=None))

    assert result.goto == "evaluator"
    assert result.update["route"] == "agent"


def test_the_chat_fast_path_is_labelled_chat(monkeypatch):
    fake = _Scripted(["hello!"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric([], conversational=True))

    result = pn.agent(_state(messages=[HumanMessage("hi")]))

    assert result.update["route"] == "chat"
    assert result.update["final_output"] == "hello!"


# --------------------------------------------------------------------------
# the evaluator on the research route
# --------------------------------------------------------------------------

def _judged(monkeypatch, reply: str, **state):
    fake = _Scripted([reply])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    result = pn.evaluator(_state(
        node="research", output="Document written to x -- 3 sections, 900 words.",
        checklist=[{"text": "three sections", "status": "pending", "evidence": ""}],
        **state,
    ))
    return fake, result


def test_the_evaluator_sends_a_research_rejection_back_to_research(monkeypatch):
    _, result = _judged(monkeypatch, "FINAL:\nMET: 0/1\nBLOCKED: no\nAPPROVE: no\nWHY: thin",
                        route="research")

    assert result.goto == "research"
    assert result.update["feedback"]


def test_the_evaluator_sends_an_agent_rejection_back_to_the_agent(monkeypatch):
    _, result = _judged(monkeypatch, "FINAL:\nMET: 0/1\nBLOCKED: no\nAPPROVE: no\nWHY: thin",
                        route="agent")

    assert result.goto == "agent"


def test_the_evaluator_judges_a_document_from_its_report(monkeypatch):
    fake, _ = _judged(monkeypatch, "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
                      route="research")

    system, human = fake.seen[0][0].content, fake.seen[0][1].content
    assert "DOCUMENT" in system
    assert "REPORT ON THE DOCUMENT" in human


def test_a_research_run_leaves_no_lesson_and_credits_no_seat(monkeypatch):
    recorded = []
    monkeypatch.setattr(pn.seat_outcomes, "record", lambda *a, **kw: recorded.append(a))
    monkeypatch.setattr(pn, "learning_enabled", lambda: True)
    fake, result = _judged(monkeypatch, "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
                           route="research", actions=["research: wrote sections/01-a.md"])

    assert result.goto == "__end__"
    assert fake.calls == 1, "the verdict, and no distilling call after it"
    assert recorded == []
    assert not [line for line in result.update["board"] if line.startswith("learned:")]
