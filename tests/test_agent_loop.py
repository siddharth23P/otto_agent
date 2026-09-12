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
    # The loop writes its checklist from the task before it starts. Tests that
    # care about that call it explicitly; the rest are about what the loop does
    # afterwards, so they get a fixed one and keep their scripted replies
    # aligned with the exchanges they are actually asserting on.
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["the task is done"])


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
    # Four, not three: a run that spent its budget without producing an answer
    # now pays one more call to write down what it has, because reporting
    # nothing is the worst outcome available. See OUT_OF_BUDGET_NOTE.
    assert result.update["model_calls"] == 4
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


# --------------------------------------------------------------------------
# switch_mode, and the guards against thrashing
# --------------------------------------------------------------------------
#
# A model choosing its own model is a new failure mode. The guards are ordered
# cheapest first, and the first one is free: a swap costs a model call and
# yields no tool result, so it already competes for budget against real work.

def test_a_swap_changes_which_task_the_next_call_routes_to(monkeypatch):
    asked: list = []

    def chat_model(task, *a, **kw):
        asked.append(task)
        return fake

    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan",
        "FINAL:\ndone",
    ])
    monkeypatch.setattr(pn.ROUTER, "chat_model", chat_model)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["done"])

    result = pn.agent(_state())

    assert result.update["mode"] == "plan"
    assert asked[-1] is pn.MODES["plan"].task
    # asked[0] is the checklist, written before the loop starts and routed to
    # the judging seat rather than the working one. The loop's own first call
    # is the one after it.
    assert asked[1] is pn.MODES[pn.DEFAULT_MODE].task


def test_a_swap_keeps_everything_that_came_before_it(monkeypatch):
    """The whole point. Changing role used to mean a node boundary and a
    discarded conversation."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('the earlier finding')",
        "ACTION: switch_mode\nCODE:\nplan",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    pn.agent(_state())

    after_swap = "\n".join(m.content for m in fake.seen[-1])
    assert "the earlier finding" in after_swap


def test_the_swap_is_recorded_for_the_board_and_the_trace(monkeypatch):
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan\nneeds ordering first",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    [line] = result.update["mode_log"]
    assert "solve -> plan" in line
    assert "needs ordering first" in line


def test_an_unknown_mode_is_refused_and_the_run_carries_on(monkeypatch):
    """A refusal is a message the model reads and acts on, exactly like a
    failed tool result -- never an exception."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nrefactor",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    assert result.update["mode"] == pn.DEFAULT_MODE
    assert "not a mode" in "\n".join(m.content for m in fake.seen[-1])


def test_switching_to_the_mode_you_are_in_does_not_repeat_the_guidance(monkeypatch):
    """Repeating it would grow the prompt every time the model asked for what
    it already has."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        f"ACTION: switch_mode\nCODE:\n{pn.DEFAULT_MODE}",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    pn.agent(_state())

    sent = [m.content for m in fake.seen[-1]]
    assert sum(1 for m in sent if m.startswith(f"MODE: {pn.DEFAULT_MODE}")) == 1
    assert any("already in" in m for m in sent)


def test_a_swap_straight_after_a_swap_is_refused(monkeypatch):
    """Switching is not progress. This is the guard that catches ping-ponging,
    and it is deliberately shaped like REPEATED_CALL_NOTE, which measurement
    already showed this model acts on."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan",
        "ACTION: switch_mode\nCODE:\nfind",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    assert result.update["mode"] == "plan", "the second swap should not have taken"
    assert "nothing done in between" in "\n".join(m.content for m in fake.seen[-1])


def test_doing_work_between_swaps_makes_the_next_one_allowed(monkeypatch):
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan",
        "ACTION: execute_python\nCODE:\nprint(2)",
        "ACTION: switch_mode\nCODE:\nfind",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    assert pn.agent(_state()).update["mode"] == "find"


def test_a_run_that_will_not_stop_switching_is_capped(monkeypatch, ):
    """The backstop, not the main guard -- the work-between-swaps rule catches
    ping-ponging much earlier."""
    monkeypatch.setattr(pn, "MAX_MODE_SWAPS", 2)
    swap_then_work = [
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nplan",
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nfind",
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nsummarize",
        "FINAL:\ndone",
    ]
    fake = _Scripted(swap_then_work)
    _install(monkeypatch, fake)
    result = pn.agent(_state())

    assert result.update["mode"] == "find", "the third swap should have been capped"
    assert "Finish in the mode" in "\n".join(m.content for m in fake.seen[-1])


def test_the_mode_survives_a_rejection(monkeypatch):
    """A run that switched to plan and got rejected should resume planning,
    not silently revert to the default."""
    fake = _Scripted(["FINAL:\nsecond attempt"])
    _install(monkeypatch, fake)

    result = pn.agent(_state(mode="plan", transcript=[{"kind": "human", "content": "earlier"}],
                             feedback="not good enough"))

    assert result.update["mode"] == "plan"


# --------------------------------------------------------------------------
# Compaction -- what one never-resetting conversation needs
# --------------------------------------------------------------------------

def test_a_long_run_compacts_its_old_tool_results(monkeypatch):
    """Costs no model call, which is why it is the first thing tried. On a
    tool-heavy run the transcript is mostly tool output by volume."""
    monkeypatch.setattr(pn, "LOOP_COMPACT_AT", 2000)
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 2)
    monkeypatch.setitem(
        pn.dispatch_table.__globals__["TOOL_DISPATCH"], "execute_python",
        lambda body: type("R", (), {"stdout": "y" * 3000, "stderr": "", "returncode": 0})(),
    )
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 6 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    sent = [m.content for m in fake.seen[-1]]
    assert any("compacted" in m for m in sent), "nothing was compacted"


def test_compaction_never_touches_the_task(monkeypatch):
    monkeypatch.setattr(pn, "LOOP_COMPACT_AT", 2000)
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 2)
    monkeypatch.setitem(
        pn.dispatch_table.__globals__["TOOL_DISPATCH"], "execute_python",
        lambda body: type("R", (), {"stdout": "y" * 3000, "stderr": "", "returncode": 0})(),
    )
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 6 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    assert any("fix the failing test" in m.content for m in fake.seen[-1])


def test_compaction_bounds_the_transcript(monkeypatch):
    """The property that makes one loop survivable: size stops growing with
    the number of tool calls."""
    monkeypatch.setattr(pn, "LOOP_COMPACT_AT", 2000)
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 2)
    monkeypatch.setitem(
        pn.dispatch_table.__globals__["TOOL_DISPATCH"], "execute_python",
        lambda body: type("R", (), {"stdout": "y" * 3000, "stderr": "", "returncode": 0})(),
    )
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 12 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    sizes = [sum(len(m.content) for m in sent) for sent in fake.seen]
    assert sizes[-1] < sizes[len(sizes) // 2] * 2, f"transcript still growing: {sizes}"


def test_a_short_run_is_left_alone(monkeypatch):
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)", "FINAL:\ndone"])
    _install(monkeypatch, fake)
    pn.agent(_state())
    assert not any("dropped to make room" in m.content for m in fake.seen[-1])


# --------------------------------------------------------------------------
# The gate before an irreversible action
# --------------------------------------------------------------------------
#
# A single MUTATING deviation cuts a task's success odds 55-96%; a non-mutating
# one costs 7-21%. Mutating actions are only 14-18% of steps, so the gate is
# cheap and precisely aimed. Separately, 82.5% of analysed failures are the
# agent failing to compare against evidence it already holds.
#
# Claw-Eval T026 is that failure exactly: three contacts matched "Manager
# Zhang", all three were in the transcript, and the agent sent to the first.

def test_a_read_only_tool_runs_without_a_hold(monkeypatch):
    """The gate must not tax the 85% of steps that change nothing."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)", "FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    assert len(fake.seen) == 2, "a read-only tool was held"


def test_a_workspace_write_is_not_held(monkeypatch, tmp_path):
    """Writing into a throwaway workspace or a task container is recoverable --
    write it again. Gating it cost a model call per new file and bought
    nothing; measured live, a two-file task went from 6 calls to 9."""
    from agent.pipeline.workspace import bind_workspace

    fake = _Scripted(["ACTION: write_file\nCODE:\nout.txt\nhello", "FINAL:\ndone"])
    _install(monkeypatch, fake)

    with bind_workspace(tmp_path):
        pn.agent(_state())

    assert len(fake.seen) == 2, "a workspace write was held"
    assert (tmp_path / "out.txt").exists()


def test_the_hold_names_the_ambiguity_rule(monkeypatch, tmp_path):
    """The T026 case in one sentence: more than one candidate means you do not
    know which is meant."""
    assert "more than one candidate" in pn.MUTATION_GATE_NOTE
    assert "ask_user" in pn.MUTATION_GATE_NOTE


def test_the_hold_offers_a_way_forward_rather_than_a_refusal(monkeypatch):
    """One enforcement study blocked 94% of non-compliant actions and still had
    safe completion below 5%, because the agent fabricated credentials to route
    around the block."""
    assert "issue the same call again" in pn.MUTATION_GATE_NOTE


def test_the_same_target_is_held_only_once(monkeypatch):
    """A gate that fired every time would loop forever or teach the model to
    ignore it."""
    from agent.pipeline.toolkit import ExtraTool, bind_extra_tools
    from agent.pipeline.tools import ToolResult

    ran = []
    send = ExtraTool(name="send_message", description="Send it.",
                     call=lambda b: (ran.append(b), ToolResult("ok", "", 0))[1],
                     schema={"type": "object", "properties": {"to": {"type": "string"}}})

    fake = _Scripted([
        'ACTION: send_message\nCODE:\n{"to": "a@b.c"}',
        'ACTION: send_message\nCODE:\n{"to": "a@b.c"}',
        'ACTION: send_message\nCODE:\n{"to": "a@b.c"}',
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    with bind_extra_tools([send]):
        pn.agent(_state())

    holds = sum(1 for sent in fake.seen for m in sent if "HOLD" in str(m.content))
    assert holds >= 1
    assert len(ran) == 2, "the confirmed sends did not both run"


def test_a_run_scoped_tool_is_gated_unless_it_says_otherwise(monkeypatch):
    """Safe direction: an unmarked benchmark tool is treated as irreversible.
    `gmail_send_message` is exactly the case that matters."""
    from agent.pipeline.toolkit import ExtraTool, bind_extra_tools
    from agent.pipeline.tools import ToolResult

    sent = []
    send = ExtraTool(name="send_message", description="Send it.",
                     call=lambda b: (sent.append(b), ToolResult("ok", "", 0))[1],
                     schema={"type": "object", "properties": {"to": {"type": "string"}}})

    fake = _Scripted([
        'ACTION: send_message\nCODE:\n{"to": "a@b.c"}',
        'ACTION: send_message\nCODE:\n{"to": "a@b.c"}',
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)
    with bind_extra_tools([send]):
        pn.agent(_state())

    assert any("HOLD" in str(m.content) for sent_msgs in fake.seen for m in sent_msgs)
    assert len(sent) == 1, "the send ran without being held, or ran twice"


def test_a_run_scoped_tool_that_only_reads_is_not_gated(monkeypatch):
    from agent.pipeline.toolkit import ExtraTool, bind_extra_tools
    from agent.pipeline.tools import ToolResult

    look = ExtraTool(name="list_messages", description="Read them.",
                     call=lambda b: ToolResult("[]", "", 0), mutates=False,
                     schema={"type": "object", "properties": {}})

    fake = _Scripted(['ACTION: list_messages\nCODE:\n{}', "FINAL:\ndone"])
    _install(monkeypatch, fake)
    with bind_extra_tools([look]):
        pn.agent(_state())

    assert len(fake.seen) == 2, "a read-only run-scoped tool was held"


# --------------------------------------------------------------------------
# The checklist: the run's working state, not its conversation
# --------------------------------------------------------------------------
#
# In the ablation this comes from, a verified working state was worth +24
# points where an experience library over the same tasks was worth +2 -- and
# injecting more library text with no state signal scored 16 points BELOW
# state alone. What matters is knowing what is still open.

def test_the_loop_is_told_what_has_to_be_true(monkeypatch):
    fake = _Scripted(["FINAL:\ndone"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["fib(10) prints 55", "fib.py exists"])

    result = pn.agent(_state())

    seeded = "\n".join(m.content for m in fake.seen[0])
    assert "fib(10) prints 55" in seeded
    assert "fib.py exists" in seeded
    assert len(result.update["checklist"]) == 2


def test_the_checklist_is_written_before_any_attempt_exists(monkeypatch):
    """Stronger than writing it after the fact: nothing in it can be shaped by
    an attempt trying to satisfy it."""
    seen_task = []
    fake = _Scripted(["FINAL:\ndone"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: seen_task.append(task) or ["c"])

    pn.agent(_state())

    assert seen_task == ["fix the failing test"], "the checklist saw more than the task"


def test_an_existing_checklist_is_not_rewritten(monkeypatch):
    """A rejection must not move the bar the attempt is being judged against."""
    calls = []
    fake = _Scripted(["FINAL:\nsecond attempt"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: calls.append(1) or ["new"])

    existing = [{"text": "the original bar", "status": "pending", "evidence": ""}]
    result = pn.agent(_state(checklist=existing, transcript=[{"kind": "human", "content": "earlier"}],
                             feedback="not yet"))

    assert calls == [], "the checklist was rewritten mid-run"
    assert result.update["checklist"][0]["text"] == "the original bar"


def test_a_rejected_run_is_shown_what_is_still_open(monkeypatch):
    fake = _Scripted(["FINAL:\nsecond attempt"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)

    pn.agent(_state(
        checklist=[{"text": "the file exists", "status": "met", "evidence": "saw it"},
                   {"text": "the test passes", "status": "pending", "evidence": ""}],
        transcript=[{"kind": "human", "content": "earlier"}],
        feedback="the test still fails",
    ))

    resumed = "\n".join(m.content for m in fake.seen[0])
    assert "[done] the file exists" in resumed
    assert "[  ] the test passes" in resumed


def test_only_the_judgment_may_change_a_status():
    """An executor's claim about its own work is not evidence -- that
    separation is the whole reason the state layer exists."""
    checklist = [{"text": "a", "status": "pending", "evidence": ""}]
    approved = pn._settle(checklist, pn._parse_verdict(
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: verified"))
    assert approved[0]["status"] == "met"
    assert approved[0]["evidence"] == "verified"


def test_a_blocked_record_is_not_the_same_as_an_open_one():
    """An obstacle outside the agent's control should not be retried
    identically, and a plain rejection invites exactly that."""
    checklist = [{"text": "a", "status": "pending", "evidence": ""}]
    blocked = pn._settle(checklist, pn._parse_verdict(
        "FINAL:\nMET: 0/1\nBLOCKED: yes\nAPPROVE: no\nWHY: the service was down"))
    assert blocked[0]["status"] == "blocked"


def test_a_rejection_leaves_records_open_rather_than_guessing():
    """The verdict says how many criteria were met, not which. A wrong `met`
    is worse than an honest `pending`, because the next attempt would skip the
    thing that is actually missing."""
    checklist = [{"text": "a", "status": "pending", "evidence": ""},
                 {"text": "b", "status": "pending", "evidence": ""}]
    settled = pn._settle(checklist, pn._parse_verdict(
        "FINAL:\nMET: 1/2\nBLOCKED: no\nAPPROVE: no\nWHY: b is missing"))
    assert [i["status"] for i in settled] == ["pending", "pending"]


def test_compaction_replaces_a_result_with_its_summary_not_its_first_bytes(monkeypatch):
    """`actions` already holds one line per call, written when it ran. Keeping
    the first 240 characters keeps whatever came first, which for a failing
    command is usually the banner and not the error."""
    messages = [
        HumanMessage("TOOL RESULT:\n" + "x" * 5000),
        HumanMessage("recent, untouched"),
    ]
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 1)

    pn._compact(messages, ["solve: execute_bash pytest -> FAILED (exit 1): E0433"])

    assert "E0433" in messages[0].content
    assert "x" * 100 not in messages[0].content


def test_compaction_never_touches_what_the_run_is_for(monkeypatch):
    """Type-blind compaction takes constraint recall from 53% to 10% over five
    rounds. Losing a tool result costs a re-run; losing a constraint means the
    agent does the wrong thing confidently for the rest of the session."""
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 0)
    protected = [
        HumanMessage("TOOL RESULT:\nTASK:\n" + "x" * 5000),
        HumanMessage("TOOL RESULT:\nTHIS IS WHAT HAS TO BE TRUE WHEN YOU ARE DONE:\n" + "y" * 5000),
        HumanMessage("TOOL RESULT:\nHOLD. send_message changes something\n" + "z" * 5000),
    ]
    before = [m.content for m in protected]

    pn._compact(protected, [])

    assert [m.content for m in protected] == before


def test_the_summary_still_lines_up_after_some_are_compacted(monkeypatch):
    """Off-by-one here would attach the wrong summary to the wrong call, which
    is worse than not compacting at all."""
    monkeypatch.setattr(pn, "KEEP_VERBATIM", 1)
    messages = [
        HumanMessage("TOOL RESULT:\n" + "a" * 5000),
        HumanMessage("TOOL RESULT:\nshort"),
        HumanMessage("TOOL RESULT:\n" + "c" * 5000),
        HumanMessage("recent"),
    ]

    pn._compact(messages, ["first call", "second call", "third call"])

    assert "first call" in messages[0].content
    assert messages[1].content == "TOOL RESULT:\nshort"
    assert "third call" in messages[2].content


# --------------------------------------------------------------------------
# The handoff is asymmetric
# --------------------------------------------------------------------------
#
# Handing a stronger model the weaker one's trajectory recovers under half the
# quality it should, at four to six times the cost; DISCARDING that trajectory
# takes recovery from 47% to 64%. The reverse is not true -- removing a strong
# model's trajectory before handing down hurts. Strong trajectories guide weak
# receivers; weak trajectories burden strong ones.

def test_escalating_drops_the_working_conversation(monkeypatch):
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('a weaker model was here')",
        "ACTION: switch_mode\nCODE:\nsolve",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="find"))

    after = "\n".join(m.content for m in fake.seen[-1])
    assert "a weaker model was here" not in after, "the weak trajectory was carried up"


def test_escalating_keeps_what_the_run_established(monkeypatch):
    """The restart is only affordable because the checklist survives it: what
    is established is state, not conversation."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nsolve",
        "FINAL:\ndone",
    ])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["the thing is true"])

    pn.agent(_state(mode="find"))

    after = "\n".join(m.content for m in fake.seen[-1])
    assert "the thing is true" in after, "the checklist did not survive the restart"
    assert "fix the failing test" in after, "the task did not survive the restart"


def test_de_escalating_carries_everything(monkeypatch):
    """Removing a strong model's trajectory before handing down measurably
    hurts, so this direction keeps it."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('a stronger model found this')",
        "ACTION: switch_mode\nCODE:\nsummarize",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    after = "\n".join(m.content for m in fake.seen[-1])
    assert "a stronger model found this" in after, "the strong trajectory was dropped"


def test_a_sideways_move_is_not_an_escalation(monkeypatch):
    """Two modes at the same depth carry context; only moving UP restarts."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('sideways evidence')",
        "ACTION: switch_mode\nCODE:\nsummarize",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="find"))

    after = "\n".join(m.content for m in fake.seen[-1])
    assert "sideways evidence" in after


def test_the_restart_is_recorded(monkeypatch):
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint(1)",
        "ACTION: switch_mode\nCODE:\nsolve",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    result = pn.agent(_state(mode="find"))

    assert any("restarted" in line for line in result.update["mode_log"])


# --------------------------------------------------------------------------
# Reminders: what gets re-said, and what it costs
# --------------------------------------------------------------------------
#
# Long runs suffer instruction fade-out. The reflection here comes from the
# strongest cheap result in the survey, whose ablation is the whole point: an
# agent that COULD write itself tools scored 62% -> 64%; adding the question
# after each step took it to 76%. Deciding WHEN is the mechanism.

def test_reminders_cost_no_model_call(monkeypatch):
    """They ride on a tool result that was being sent regardless."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 12 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)
    monkeypatch.setattr(pn, "REMINDER_EVERY", 2)

    pn.agent(_state())

    # One call per scripted reply and nothing extra.
    assert len(fake.seen) == 13


def test_the_tool_building_question_is_asked_on_a_long_run(monkeypatch):
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 8 + ["FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    assert any(pn.TOOL_BUILDING_NOTE in str(m.content)
               for sent in fake.seen for m in sent)


def test_a_short_run_is_not_nagged(monkeypatch):
    """Said constantly these become wallpaper, which is the failure mode they
    exist to fix."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)", "FINAL:\ndone"])
    _install(monkeypatch, fake)

    pn.agent(_state())

    assert not any(pn.TOOL_BUILDING_NOTE in str(m.content)
                   for sent in fake.seen for m in sent)


def test_the_reminder_restates_what_is_still_open(monkeypatch):
    """Counters fade-out on the thing that matters most: what the run is for."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 8 + ["FINAL:\ndone"])
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["the suite passes"])

    pn.agent(_state())

    assert any("Still open" in str(m.content) and "the suite passes" in str(m.content)
               for sent in fake.seen for m in sent)


def test_a_settled_checklist_is_not_restated():
    assert "Still open" not in pn._reminders(
        pn.REMINDER_EVERY, [{"text": "done thing", "status": "met", "evidence": "saw it"}])


def test_reminders_are_periodic_not_constant():
    assert pn._reminders(1, None) == ""
    assert pn._reminders(pn.REMINDER_EVERY, None) != ""
    assert pn._reminders(pn.REMINDER_EVERY + 1, None) == ""


# --------------------------------------------------------------------------
# Delegation: the one shape of multi-agent the evidence supports
# --------------------------------------------------------------------------
#
# Where every agent shares a model, a single agent role-playing the workflow
# matches or beats the multi-agent version at lower cost. A sub-agent earns its
# keep only when the model genuinely differs or the context must be isolated.
# And reasoning belongs at the orchestrator: +18.2 and +36.7 points at 8% added
# latency there, marginal-to-negative at +77% in the sub-agents.

def test_a_subtask_runs_on_the_other_modes_model(monkeypatch):
    asked = []

    def chat_model(task, *a, **kw):
        asked.append(task)
        return fake

    fake = _Scripted([
        "ACTION: delegate\nCODE:\nfind\nlook up the config value",
        "FINAL:\nthe value is 42",
        "FINAL:\nall done",
    ])
    monkeypatch.setattr(pn.ROUTER, "chat_model", chat_model)
    monkeypatch.setattr(pn, "_criteria", lambda llm, task: ["done"])

    result = pn.agent(_state(mode="solve"))

    assert pn.MODES["find"].task in asked
    assert result.update["output"] == "all done"


def test_the_subtask_does_not_get_the_parents_conversation(monkeypatch):
    """A contract goes down, never a transcript."""
    fake = _Scripted([
        "ACTION: execute_python\nCODE:\nprint('parent private working')",
        "ACTION: delegate\nCODE:\nfind\nlook something up",
        "FINAL:\nthe child answer",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    # The third call is the child's first exchange.
    child_saw = "\n".join(str(m.content) for m in fake.seen[2])
    assert "parent private working" not in child_saw
    assert "look something up" in child_saw


def test_the_parent_gets_a_report_not_a_trajectory(monkeypatch):
    """What comes back up is the answer. The child's working is discarded,
    which is what stops a long delegation costing the parent its context."""
    fake = _Scripted([
        "ACTION: delegate\nCODE:\nfind\nlook it up",
        "ACTION: execute_python\nCODE:\nprint('child private working')",
        "FINAL:\nthe child answer",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    parent_saw = "\n".join(m.content for m in fake.seen[-1])
    assert "the child answer" in parent_saw
    assert "child private working" not in parent_saw


def test_delegating_to_your_own_mode_is_refused(monkeypatch):
    """The degenerate case the literature warns about: sub-agents used purely
    as context-isolation threads, paying coordination for what the parent could
    do itself."""
    fake = _Scripted([
        "ACTION: delegate\nCODE:\nsolve\ndo the thing",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    assert any("already in solve" in str(m.content) for sent in fake.seen for m in sent)


def test_a_subtask_cannot_delegate(monkeypatch):
    """One level. A sub-agent that delegates is a subtask that was never
    bounded, and the depth would compound silently."""
    fake = _Scripted([
        "ACTION: delegate\nCODE:\nfind\nouter job",
        "ACTION: delegate\nCODE:\nplan\ninner job",
        "FINAL:\nchild answer",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    assert any("not available inside a delegated" in str(m.content)
               for sent in fake.seen for m in sent)


def test_the_subtask_is_bounded(monkeypatch):
    """A sub-agent that needs a long conversation is a subtask that was not
    bounded properly, and the parent is better placed to notice than the child."""
    monkeypatch.setattr(pn, "MAX_DELEGATE_ITERATIONS", 2)
    fake = _Scripted(
        ["ACTION: delegate\nCODE:\nfind\nlook it up"]
        + ["ACTION: execute_python\nCODE:\nprint(1)"] * 20
        + ["FINAL:\ndone"]
    )
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    # 1 parent delegate call + 2 child exchanges, then the parent carries on.
    # Without the cap the child would consume every scripted reply.
    assert any("finished without an answer" in str(m.content)
               for sent in fake.seen for m in sent), "the subtask was not bounded"


def test_what_the_subtask_did_joins_the_parents_record(monkeypatch):
    """Its conversation is discarded; the account of what it ran is not."""
    fake = _Scripted([
        "ACTION: delegate\nCODE:\nfind\nlook it up",
        "ACTION: execute_python\nCODE:\nprint('evidence')",
        "FINAL:\nfound it",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    result = pn.agent(_state(mode="solve"))

    assert any("delegated" in line for line in result.update["actions"])


def test_a_malformed_delegation_says_what_was_wrong(monkeypatch):
    fake = _Scripted([
        "ACTION: delegate\nCODE:\nnot-a-mode\ndo something",
        "FINAL:\ndone",
    ])
    _install(monkeypatch, fake)

    pn.agent(_state(mode="solve"))

    assert any("first line must be a mode" in str(m.content)
               for sent in fake.seen for m in sent)


# --------------------------------------------------------------------------
# What a mode switch must not carry, and what it must
# --------------------------------------------------------------------------

def test_an_escalating_switch_forgets_that_a_tool_was_already_held():
    """The gate's memory belongs to the conversation it was recorded in.

    `confirmed` says "this target has been held once, let the reissue
    through". That is only true while the model can REMEMBER being asked. An
    escalating switch wipes the transcript back to the seed, so the model that
    arrives next has no record of the hold -- and would have found the gate
    already satisfied and run the irreversible call unchecked.

    Holding the same target twice costs one exchange. Not holding it costs the
    thing the gate exists for.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage("prompt"), HumanMessage("TASK: do it")]
    seed = list(messages)
    confirmed = {"execute_bash:rm -rf /data"}

    pn._switch_mode(messages, "solve", mode="summarize", swaps=0,
                    did_work=True, mode_log=[], calls=0, seed=seed,
                    confirmed=confirmed)

    assert confirmed == set(), "the gate would have been skipped after the wipe"


def test_a_de_escalating_switch_keeps_it():
    """Nothing was wiped, so the model still remembers being asked and a
    second hold would be pure tax."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage("prompt"), HumanMessage("TASK: do it")]
    confirmed = {"execute_bash:rm -rf /data"}

    pn._switch_mode(messages, "summarize", mode="solve", swaps=0,
                    did_work=True, mode_log=[], calls=0, seed=list(messages),
                    confirmed=confirmed)

    assert confirmed == {"execute_bash:rm -rf /data"}


def test_the_plan_survives_switching_back_to_solve():
    """`plan` mode's own guidance says to "switch back and carry the steps out
    yourself". `solve` is the deepest mode, so switching back always escalates,
    and escalating wipes the transcript -- which destroyed the plan at the
    moment it was needed. Nothing else held it: `context` is only written by
    ask_user and the checklist is fixed at run start."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    plan = "1. read config.yaml\n2. change the timeout\n3. run the tests"
    messages = [SystemMessage("prompt"), HumanMessage("TASK: fix it"),
                AIMessage(f"FINAL:\n{plan}")]
    seed = messages[:2]

    pn._switch_mode(messages, "solve", mode="plan", swaps=1, did_work=True,
                    mode_log=[], calls=5, seed=seed, confirmed=set())

    carried = "\n".join(pn._content_text(m.content) for m in messages)
    assert "change the timeout" in carried, "the plan was thrown away"


def test_what_is_carried_is_bounded():
    """One message, not the whole conversation. Escalation exists to drop the
    weaker model's account of getting somewhere."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    messages = [SystemMessage("p"), HumanMessage("TASK: t")]
    seed = list(messages)
    for i in range(20):
        messages.append(AIMessage(f"step {i} " + "x" * 500))

    pn._switch_mode(messages, "solve", mode="plan", swaps=1, did_work=True,
                    mode_log=[], calls=5, seed=seed, confirmed=set())

    carried = "\n".join(pn._content_text(m.content) for m in messages)
    assert "step 19" in carried
    assert "step 0 " not in carried
    assert len(carried) < 20 * 500


# --------------------------------------------------------------------------
# Settling criteria from the record, not only from the verdict
# --------------------------------------------------------------------------

def test_a_criterion_whose_artefact_was_written_stops_being_open():
    """The checklist is re-stated every REMINDER_EVERY iterations while items
    are open. Without this a run keeps being nagged about something it did
    twenty actions ago, which is how a reminder becomes wallpaper -- the exact
    failure the reminder exists to prevent."""
    checklist = [
        {"text": "a report exists at report.md", "status": "pending", "evidence": ""},
        {"text": "the totals are correct", "status": "pending", "evidence": ""},
    ]

    settled = pn._note_evidence(checklist, ["solve: write_file report.md ok (42 lines)"])

    assert [i["status"] for i in settled] == ["seen", "pending"]
    assert "report.md" in settled[0]["evidence"]


def test_it_stops_short_of_saying_the_criterion_is_met():
    """A successful write is environment-grounded -- it is a returncode, not
    the agent's account of itself -- but it says the artefact EXISTS, not that
    its contents satisfy anything. A wrong `seen` costs a missing nag; a wrong
    `met` would cost the next attempt skipping what is actually absent."""
    settled = pn._note_evidence(
        [{"text": "report.md holds the Q3 totals", "status": "pending", "evidence": ""}],
        ["solve: write_file report.md ok"],
    )

    assert settled[0]["status"] == "seen"
    assert settled[0]["status"] != "met"


def test_a_failed_write_grounds_nothing():
    settled = pn._note_evidence(
        [{"text": "a report exists at report.md", "status": "pending", "evidence": ""}],
        ["solve: write_file report.md failed: no such directory"],
    )

    assert settled[0]["status"] == "pending"


def test_reading_a_file_is_not_evidence_that_it_was_produced():
    """A criterion about a report is not satisfied by having looked at one."""
    settled = pn._note_evidence(
        [{"text": "a report exists at report.md", "status": "pending", "evidence": ""}],
        ["solve: read_file report.md ok", "solve: list_files . ok"],
    )

    assert settled[0]["status"] == "pending"


def test_an_unrelated_path_does_not_settle_a_criterion():
    settled = pn._note_evidence(
        [{"text": "a report exists at report.md", "status": "pending", "evidence": ""}],
        ["solve: write_file scratch/notes.txt ok"],
    )

    assert settled[0]["status"] == "pending"


def test_a_criterion_naming_no_path_is_left_alone():
    """Most criteria are about values and behaviour, not files. Matching them
    on prose would mark work done that nobody did."""
    settled = pn._note_evidence(
        [{"text": "the reported total is what the code prints", "status": "pending",
          "evidence": ""}],
        ["solve: write_file report.md ok"],
    )

    assert settled[0]["status"] == "pending"


def test_a_settled_criterion_is_not_re_opened():
    already = [{"text": "x at a.py", "status": "met", "evidence": "judge said so"}]

    assert pn._note_evidence(already, ["solve: write_file a.py ok"]) == already


def test_an_acted_on_criterion_is_not_listed_as_still_open():
    """The point of the whole thing: the reminder names what is left."""
    checklist = [
        {"text": "a report exists at report.md", "status": "seen", "evidence": "wrote report.md"},
        {"text": "the totals are correct", "status": "pending", "evidence": ""},
    ]

    reminder = pn._reminders(pn.REMINDER_EVERY, checklist)

    assert "the totals are correct" in reminder
    assert "report.md" not in reminder


def test_the_loop_settles_the_checklist_as_it_goes():
    """Not a unit of `_note_evidence`: the loop has to actually call it, or
    the mechanism exists and nothing drives it."""
    import inspect

    source = inspect.getsource(pn._agent_loop)
    assert "_note_evidence(checklist, actions)" in source


# --------------------------------------------------------------------------
# A spent run answers with what it has
# --------------------------------------------------------------------------

def test_a_run_that_runs_out_still_produces_an_answer(monkeypatch):
    """Out of budget with nothing to show is the worst outcome available, and
    it used to be a common one: the loop returned "" and the run reported
    nothing. One more call buys an answer from what it already has."""
    fake = _Scripted(["ACTION: execute_python\nCODE:\nprint(1)"] * 3
                     + ["FINAL:\npartial, but here is what I found"])
    _install(monkeypatch, fake)

    with bind_budget(Budget(max_model_calls=3)):
        result = pn.agent(_state())

    assert result.update["final_output"] == "partial, but here is what I found"


def test_the_salvage_never_hands_back_a_tool_call_as_the_answer():
    """Returning the raw reply would put "ACTION: execute_bash..." in front of
    the person as the result of their run -- the ACTION-protocol leak this
    loop has a rule against, reintroduced by the one path that exists to
    salvage something."""
    class _Action:
        def stream(self, messages):
            from langchain_core.messages import AIMessageChunk
            yield AIMessageChunk(content="ACTION: execute_bash\nCODE:\nls")

    assert pn._answer_from_what_is_here(_Action(), []) == ""


def test_prose_that_forgot_the_marker_is_still_an_answer():
    class _Prose:
        def stream(self, messages):
            from langchain_core.messages import AIMessageChunk
            yield AIMessageChunk(content="I found three files and two of them parse.")

    answer = pn._answer_from_what_is_here(_Prose(), [])

    assert "three files" in answer


def test_a_failure_while_salvaging_leaves_the_run_no_worse():
    """It returns the same nothing the caller already had, so this path can
    never make the outcome worse than it was."""
    class _Broken:
        def stream(self, messages):
            raise RuntimeError("provider down")

    assert pn._answer_from_what_is_here(_Broken(), []) == ""


def test_a_run_that_already_answered_pays_nothing_extra(monkeypatch):
    """The salvage fires only when there is no answer. A run that finished
    normally must not be charged for it."""
    fake = _Scripted(["FINAL:\nthe answer"])
    _install(monkeypatch, fake)

    with bind_budget(Budget(max_model_calls=20)):
        result = pn.agent(_state())

    # Goes to the judge with an answer in hand, rather than ending on a
    # salvage -- `output`, not `final_output`, which the evaluator sets.
    assert result.update["output"] == "the answer"
    assert result.update["model_calls"] <= 3


# --------------------------------------------------------------------------
# The one edge with no bound on it
# --------------------------------------------------------------------------
#
# The evaluator handed a provider failure back to the agent with no counter
# anywhere on the path. The agent reworked its answer, handed it back, the
# same dead provider refused again, and the two bounced until LangGraph's
# recursion limit ended the run outright -- losing the answer the agent had
# been holding the whole time. Measured live with an exhausted Anthropic key:
# 68 model requests and 190 seconds for a task that answers in three, ending
# in GraphRecursionError with nothing.

def test_a_dead_judge_ends_the_run_instead_of_bouncing(monkeypatch):
    def refuses(llm, messages, **kw):
        raise pn.ProviderError("credit balance is too low")

    monkeypatch.setattr(pn, "_tool_loop", refuses)
    state = {
        "messages": [HumanMessage("do the thing")],
        "output": "here is the thing",
        "node": "agent",
        "judge_errors": pn.MAX_JUDGE_ERRORS,
    }

    command = pn.evaluator(state)

    assert command.goto == "__end__", "the judge bounced it back again"
    assert command.update["final_output"] == "here is the thing", (
        "the answer the agent was holding was thrown away"
    )
    assert "unverified" in command.update["board"][0]


def test_the_first_provider_failure_is_still_retried(monkeypatch):
    """One retry, not none. A network blip between two calls is real, and a
    run that gave up on the first one would stop being judged over nothing."""
    def refuses(llm, messages, **kw):
        raise pn.ProviderError("connection reset")

    monkeypatch.setattr(pn, "_tool_loop", refuses)
    command = pn.evaluator({
        "messages": [HumanMessage("do the thing")],
        "output": "here is the thing",
        "node": "agent",
    })

    assert command.goto == "evaluator", (
        "the agent was sent to rework an answer nobody rejected"
    )
    assert command.update["judge_errors"] == 1
    assert "feedback" not in command.update, (
        "a provider failure was handed to the agent as if it were a verdict"
    )


def test_a_failed_rubric_is_not_retried_on_every_pass(monkeypatch):
    """carry() writes a None checklist back over whatever the evaluator
    settled, so without this the next pass finds None again and pays for
    another rubric call against the provider that just refused it. Twenty of
    them in one turn, on the run that found this."""
    tried = []

    def dead(llm, task_text):
        tried.append(task_text)
        return None

    monkeypatch.setattr(pn, "_criteria", dead)
    monkeypatch.setattr(pn, "_agent_loop", lambda *a, **kw: ("an answer", "final", "solve"))

    resuming = {
        "messages": [HumanMessage("do the thing")],
        "transcript": [{"kind": "human", "content": "TASK:\ndo the thing"}],
    }
    pn.agent(resuming)

    assert tried == [], "a resuming pass paid for the rubric all over again"
