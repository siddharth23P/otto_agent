"""A message that asked for no work costs two model calls, not thirteen.

Live, "hi otto!" went through the whole engineering pipeline -- criteria, an
agent loop on the reasoning seat, a judge, a distilled lesson -- for 13 model
calls and about five minutes. The loop is told to run something that would
fail if the task were not done; handed a greeting, it invented one, grepping
the workspace four times and writing a test file to disk to have something to
verify.

What decides between the two paths is the rubric call that was already
happening first (agent/pipeline/nodes.py's `_rubric`), so a real task pays
nothing for the fast path existing. These tests pin both halves: a greeting
takes it, and every way of being unsure does not.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.graph import END

from agent.pipeline import nodes as pn
from agent.router.llm_provider.base import ProviderError


class _Scripted:
    """One reply per .stream() call, recording what it was handed."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.seen: list[list] = []

    def stream(self, messages):
        self.seen.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "FINAL:\ndone"
        yield AIMessageChunk(content=reply)


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("hi otto!")],
        "board": [], "node": None, "feedback": "", "output": None,
        "context": "", "node_error": None, "pending_question": None,
        "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [],
        "model_calls": 0, "rejections": 0, "asked_qa": [], "user_answer": None,
    }
    base.update(overrides)
    return base


def _chatty(monkeypatch, fake, *, conversational=True):
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(
        pn, "_rubric",
        lambda llm, task: pn.Rubric([], conversational=conversational),
    )


# --------------------------------------------------------------------------
# The reply the rubric call gives back
# --------------------------------------------------------------------------

def test_a_no_task_reply_is_read_as_conversational(monkeypatch):
    monkeypatch.setattr(pn, "_call", lambda llm, messages: "NO TASK")

    rubric = pn._rubric(object(), "hi otto!")

    assert rubric.conversational
    assert rubric.criteria == []


def test_criteria_are_never_conversational(monkeypatch):
    monkeypatch.setattr(pn, "_call",
                        lambda llm, messages: "- the suite passes\n- nothing else broke")

    rubric = pn._rubric(object(), "fix the failing test")

    assert not rubric.conversational
    assert rubric.criteria == ["the suite passes", "nothing else broke"]


def test_a_reply_that_says_both_is_treated_as_work(monkeypatch):
    """The fast path skips the judge, so it is the one decision here a
    rejection cannot undo. A contradiction is read the safe way."""
    monkeypatch.setattr(pn, "_call",
                        lambda llm, messages: "NO TASK\n- the file exists")

    assert not pn._rubric(object(), "write a file").conversational


def test_a_failed_rubric_call_is_never_conversational(monkeypatch):
    def boom(llm, messages):
        raise ProviderError("the vendor is down")

    monkeypatch.setattr(pn, "_call", boom)
    rubric = pn._rubric(object(), "fix the failing test")

    assert rubric == pn.Rubric([], conversational=False)


# --------------------------------------------------------------------------
# What the agent node does with it
# --------------------------------------------------------------------------

def test_a_greeting_is_answered_without_the_loop(monkeypatch):
    fake = _Scripted(["Hi! I'm otto -- what can I help you with today?"])
    _chatty(monkeypatch, fake)

    result = pn.agent(_state())

    assert result.goto == END, "a greeting still went on to the loop"
    assert len(fake.seen) == 1, (
        f"{len(fake.seen)} calls past the rubric to say hello"
    )
    assert result.update["final_output"].startswith("Hi!")
    # Nothing to judge, so nothing is handed to the judge -- and nothing is
    # left half-finished for the next turn to resume into.
    assert result.update["output"] == result.update["final_output"]


def test_the_fast_path_never_uses_the_action_protocol(monkeypatch):
    """The reply is the answer itself. A prompt carrying ACTION:/FINAL: is how
    a one-line greeting turns back into a tool loop."""
    fake = _Scripted(["Hi there."])
    _chatty(monkeypatch, fake)

    pn.agent(_state())

    sent = "\n".join(str(m.content) for m in fake.seen[0])
    assert "ACTION:" not in sent
    assert "FINAL:" not in sent


def test_what_was_said_earlier_reaches_the_reply(monkeypatch):
    """"what did I just ask you" is conversation, not work -- but it is only
    answerable with the conversation in front of it."""
    fake = _Scripted(["You asked me to fix the failing test."])
    _chatty(monkeypatch, fake)

    pn.agent(_state(messages=[HumanMessage("fix the failing test"),
                              HumanMessage("what did I just ask you?")]))

    assert "fix the failing test" in "\n".join(str(m.content) for m in fake.seen[0])


def test_a_task_is_untouched(monkeypatch):
    fake = _Scripted(["FINAL:\nfixed"])
    _chatty(monkeypatch, fake, conversational=False)

    result = pn.agent(_state(messages=[HumanMessage("fix the failing test")]))

    assert result.goto == "evaluator"


def test_a_resumed_run_never_takes_the_fast_path(monkeypatch):
    """A turn with work behind it -- a question it asked, a rejection to fix --
    is not conversational just because the last sentence is."""
    fake = _Scripted(["FINAL:\nthe report is written"])
    _chatty(monkeypatch, fake)

    result = pn.agent(_state(
        transcript=[{"type": "human", "content": "write the report"}],
        user_answer="thanks!",
    ))

    assert result.goto == "evaluator"


def test_an_empty_reply_falls_back_to_the_task_path(monkeypatch):
    """A fast path that can lose a turn is worse than no fast path."""
    fake = _Scripted(["", "FINAL:\nhello"])
    _chatty(monkeypatch, fake)

    result = pn.agent(_state())

    assert result.goto == "evaluator"
    assert result.update["output"] == "hello"


def test_a_provider_failure_falls_back_to_the_task_path(monkeypatch):
    calls = []

    def failing_chat(state, task_text):
        calls.append(task_text)
        raise ProviderError("the cheap seat is down")

    fake = _Scripted(["FINAL:\nhello"])
    _chatty(monkeypatch, fake)
    monkeypatch.setattr(pn, "_chat_reply", failing_chat)

    result = pn.agent(_state())

    assert calls, "the fast path was never tried"
    assert result.goto == "evaluator"
    assert result.update["output"] == "hello"
