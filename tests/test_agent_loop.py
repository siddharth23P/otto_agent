"""Coverage for agent/pipeline/nodes.py's single agent loop.

This replaces tests/test_router_node.py, tests/test_parse_router.py,
tests/test_plan_execution.py and most of tests/test_role_nodes.py, all deleted
with the machinery they covered. What is being tested is one property those
four files could not express, because the design made it impossible: the
conversation survives.

The old graph was router -> role -> router -> evaluator -> router, and every
node rebuilt `[SystemMessage, HumanMessage]` from scratch while the tool
conversation died with the node. Measured on Claw-Eval traces, those boundaries
cost 20 to 126 seconds each and were 45 to 69% of a run's wall time, against
0.1 to 0.4 seconds of actual tool execution per task. The agent forgot what it
had just done AND paid to be reminded -- one boundary, not two problems.

Three tests here guard failures that would be silent rather than loud, and they
are the reason this file exists at all:

  * no SystemMessage after the opening run -- langchain_anthropic raises on
    non-consecutive system messages and langchain_google_genai hoists a mid-list
    one out of position or drops it at a bare `else: pass`;
  * no empty AIMessage -- `_call` returns "" on an empty stream, Anthropic
    rejects empty text blocks, and in a list that never resets one of those
    would break every later Anthropic call for the rest of the run;
  * the transcript is persisted on EVERY exit -- a run that pauses for a
    question without saving would resume having forgotten the work it paused
    in the middle of.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from agent.pipeline import nodes as pn
from agent.pipeline.budget import Budget, bind_budget
from agent.router.llm_provider.base import ProviderError


class _Scripted:
    """Returns each reply in order, one per .stream() call, and records the
    exact message list it was handed each time."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.seen: list[list] = []

    def stream(self, messages):
        self.seen.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "FINAL:\ndone"
        yield AIMessageChunk(content=reply)


class _Failing:
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
        "messages": [HumanMessage("fix the failing test")],
        "board": [], "node": None, "feedback": "", "output": None,
        "context": "", "node_error": None, "pending_question": None,
        "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [],
        "model_calls": 0, "rejections": 0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# The shape of the conversation
# --------------------------------------------------------------------------

def test_a_fresh_run_seeds_one_conversation_and_answers_from_it(monkeypatch):
    fake = _Scripted(["FINAL:\nthe test passes now"])
    _install(monkeypatch, fake)

    result = pn.agent(_state())

    assert result.goto == "evaluator"
    assert result.update["output"] == "the test passes now"


def test_the_task_reaches_the_model(monkeypatch):
    fake = _Scripted(["FINAL:\ndone"])
    _install(monkeypatch, fake)
    pn.agent(_state())
    assert "fix the failing test" in fake.seen[0][-2].content


def test_the_run_starts_in_the_default_mode(monkeypatch):
    fake = _Scripted(["FINAL:\ndone"])
    _install(monkeypatch, fake)
    result = pn.agent(_state())
    assert result.update["mode"] == pn.DEFAULT_MODE
    assert f"MODE: {pn.DEFAULT_MODE}" in fake.seen[0][-1].content


def test_system_messages_only_ever_sit_at_the_front(monkeypatch):
    """THE portability invariant. A SystemMessage after the opening run is a
    hard 400 on Anthropic and a silent drop on Gemini, and three routed tasks
    are pinned to Anthropic."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan",
        "ACTION: execute_python\nCODE:\nprint(2)",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    pn.agent(_state())

    for sent in fake.seen:
        systems = [i for i, m in enumerate(sent) if isinstance(m, SystemMessage)]
        assert systems == list(range(len(systems))), f"system message out of the opening run: {systems}"


def test_an_empty_reply_never_becomes_an_empty_assistant_turn(monkeypatch):
    """`_call` returns "" on an empty stream. Anthropic rejects empty text
    blocks, and in a transcript that never resets one of those breaks every
    later Anthropic call in the run."""
    fake = _Scripted(["", "FINAL:\ndone"])
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    for entry in result.update["transcript"]:
        assert entry["content"].strip(), "an empty message reached the transcript"


def test_what_a_tool_returned_is_still_there_many_calls_later(monkeypatch):
    """The property the whole rewrite exists for. Under the old graph this was
    impossible: the tool conversation died when the node returned."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('the marker value')",
        *["ACTION: execute_python\nCODE:\nprint('filler')"] * 4,
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    pn.agent(_state())

    last_sent = "\n".join(m.content for m in fake.seen[-1])
    assert "the marker value" in last_sent


# --------------------------------------------------------------------------
# Persisting the conversation across a return
# --------------------------------------------------------------------------

def test_every_exit_persists_the_transcript(monkeypatch):
    fake = _Scripted(["FINAL:\ndone"])
    _install(monkeypatch, fake)
    assert pn.agent(_state()).update["transcript"]


def test_a_rejection_resumes_the_same_conversation_rather_than_restarting(monkeypatch):
    first = _Scripted(["ACTION: execute_python\nCODE:\nprint('earlier finding')",
                       "FINAL:\nfirst answer"])
    _install(monkeypatch, first)
    one = pn.agent(_state())

    second = _Scripted(["FINAL:\nsecond answer"])
    _install(monkeypatch, second)
    pn.agent(_state(transcript=one.update["transcript"], feedback="that is wrong"))

    resumed = "\n".join(m.content for m in second.seen[0])
    assert "earlier finding" in resumed
    assert "that is wrong" in resumed


def test_a_persisted_transcript_carries_no_system_messages(monkeypatch):
    """They are rebuilt on resume. Reviving one into the middle of a list is
    the exact failure the opening-run invariant exists to prevent."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)", "FINAL:\ndone"])
    _install(monkeypatch, fake)
    stored = pn.agent(_state()).update["transcript"]
    assert all(entry["kind"] != "system" for entry in stored)


# --------------------------------------------------------------------------
# Pausing for a person -- ported from the deleted tests/test_role_nodes.py
# --------------------------------------------------------------------------

def test_the_loop_asks_the_user_and_pauses_instead_of_guessing(monkeypatch):
    fake = _Scripted(["ACTION: ask_user\nCODE:\nwhich database?"])
    _install(monkeypatch, fake)

    result = pn.agent(_state())

    assert result.goto == "ask_user"
    assert result.update["pending_question"] == "which database?"
    assert result.update["asking_role"] == "agent"


def test_asking_with_choices_parses_them_out(monkeypatch):
    fake = _Scripted(["ACTION: ask_user\nCODE:\nwhich one?\nCHOICES: postgres | mysql"])
    _install(monkeypatch, fake)
    result = pn.agent(_state())
    assert result.update["pending_choices"] == ["postgres", "mysql"]


def test_a_pause_saves_the_work_it_paused_in_the_middle_of(monkeypatch):
    """Without this the answer to the question arrives at a run that has
    forgotten why it asked."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('found the config')",
        "ACTION: ask_user\nCODE:\nwhich environment?",
    ])
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    stored = "\n".join(e["content"] for e in result.update["transcript"])
    assert "found the config" in stored


# --------------------------------------------------------------------------
# Provider failures -- ported from the deleted tests/test_role_nodes.py
# --------------------------------------------------------------------------

def test_a_provider_failure_hands_over_instead_of_crashing(monkeypatch):
    _install(monkeypatch, _Failing(ProviderError("read timeout")))
    result = pn.agent(_state())
    assert result.goto == "evaluator"
    assert "read timeout" in result.update["node_error"]


def test_a_provider_failure_says_it_was_not_a_rejection(monkeypatch):
    _install(monkeypatch, _Failing(ProviderError("boom")))
    result = pn.agent(_state())
    assert "provider" in result.update["feedback"]


def test_a_provider_failure_still_persists_the_transcript(monkeypatch):
    _install(monkeypatch, _Failing(ProviderError("boom")))
    assert "transcript" in pn.agent(_state()).update


# --------------------------------------------------------------------------
# The budget
# --------------------------------------------------------------------------

def test_a_spent_budget_ends_the_run_rather_than_handing_on(monkeypatch):
    """Never raises, and goes straight to the end rather than to the judge.

    There is nothing left to pay a judgment with, and handing on would loop:
    the evaluator rejects for lack of evidence, the loop returns immediately
    because it is still out of budget, and the two bounce until the recursion
    limit. A run killed mid-command reports nothing; one that stops and answers
    with what it has is still gradable."""
    from langgraph.graph import END

    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 10)
    _install(monkeypatch, fake)

    with bind_budget(Budget(max_model_calls=3)):
        result = pn.agent(_state())

    assert result.goto == END
    assert result.update["model_calls"] == 3
    assert "final_output" in result.update


def test_the_wrap_up_note_reaches_the_model_before_the_budget_is_gone(monkeypatch):
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 6 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)

    # Wrap-up opens at 0.8 of the budget, so a ceiling of 5 opens it at call 4
    # -- while the script still has work left to be interrupted in.
    with bind_budget(Budget(max_model_calls=5)):
        pn.agent(_state())

    assert any("FINAL" in m.content and "nearly up" in m.content
               for sent in fake.seen for m in sent)


def test_an_unbound_budget_changes_nothing(monkeypatch):
    fake = _Scripted(["FINAL:\ndone"])
    _install(monkeypatch, fake)
    assert pn.agent(_state()).update["output"] == "done"
