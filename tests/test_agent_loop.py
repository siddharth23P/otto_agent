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
